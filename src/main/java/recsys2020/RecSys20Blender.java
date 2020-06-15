package recsys2020;

import common.core.linalg.MLDenseVector;
import common.core.linalg.MLSparseMatrixAOO;
import common.core.linalg.MLSparseVector;
import common.core.utils.MLIOUtils;
import common.core.utils.MLTimer;
import common.xgb.MLXGBoost;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import recsys2020.RecSys20Data.EngageType;
import recsys2020.RecSys20Model.RecSys20Config;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys20Blender {

    public static EngageType[] ACTIONS = EngageType.values();
    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20Blender");
        timer.tic();
    }

    public RecSys20Data data;
    public RecSys20TextData textData;
    public RecSys20Split split;
    public RecSys20Config config;
    public RecSys20FeatExtractor featExtractor;

    public Map<String, Integer> userTweetToIndex;
    public float[][][] modelPreds;

    public RecSys20Blender(final RecSys20Data dataP,
                           final RecSys20TextData textDataP,
                           final RecSys20Config configP,
                           final long trainDate,
                           final long validDate) throws Exception {
        this.data = dataP;
        this.textData = textDataP;
        this.config = configP;

        //generate split
        this.split = new RecSys20Split(this.config, trainDate, validDate);
        this.split.splitByCreateDateBlend(this.data);

        //init feat extractor
        this.featExtractor = new RecSys20FeatExtractor(
                this.data,
                this.textData,
                this.split,
                this.config);
    }

    public static String getId(final int tweetIndex,
                               final int userIndex) {
        return tweetIndex + "-" + userIndex;
    }

    public void loadModelPreds(final String prefix) throws Exception {
        //create index map
        this.userTweetToIndex = new HashMap<>();
        AtomicInteger index = new AtomicInteger(0);
        try (BufferedReader reader =
                     new BufferedReader(new FileReader(this.config.modelPredFiles[0][0] + prefix))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] split = line.split(",");
                int tweetIndex = this.data.tweetToIndex.get(split[0]);
                int userIndex = this.data.userToIndex.get(split[1]);

                String id = getId(tweetIndex, userIndex);
                if (this.userTweetToIndex.containsKey(id) == false) {
                    this.userTweetToIndex.put(id, index.getAndIncrement());
                }
            }
        }
        timer.toc("loadModelPreds map done " + this.userTweetToIndex.size());

        this.modelPreds = new float[ACTIONS.length][][];
        for (EngageType action : ACTIONS) {
            String[] actionFiles = this.config.modelPredFiles[action.index];
            float[][] actionPreds =
                    new float[this.userTweetToIndex.size()][actionFiles.length];
            for (int i = 0; i < actionFiles.length; i++) {
                final int iFinal = i;
                try (BufferedReader reader =
                             new BufferedReader(new FileReader(actionFiles[i] + prefix))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] split = line.split(",");
                        int tweetIndex = this.data.tweetToIndex.get(split[0]);
                        int userIndex = this.data.userToIndex.get(split[1]);
                        float score = Float.parseFloat(split[2]);

                        String id = getId(tweetIndex, userIndex);
                        actionPreds[this.userTweetToIndex.get(id)][iFinal] =
                                score;
                    }
                }
                this.modelPreds[action.index] = actionPreds;
                timer.toc("loadModelPreds " + actionFiles[i] + prefix);
            }
        }
    }

    public void train(final String outPath) throws Exception {

        this.loadModelPreds("_valid");

        Integer[] trainIndexes = new Integer[this.split.trainIndexSet.size()];
        this.split.trainIndexSet.toArray(trainIndexes);
        Arrays.sort(trainIndexes);

        Integer[] validIndexes = new Integer[this.split.validIndexSet.size()];
        this.split.validIndexSet.toArray(validIndexes);
        Arrays.sort(validIndexes);

        BufferedWriter[] trainWriters = new BufferedWriter[ACTIONS.length];
        BufferedWriter[] validWriters = new BufferedWriter[ACTIONS.length];
        for (EngageType action : ACTIONS) {
            trainWriters[action.index] =
                    new BufferedWriter(new FileWriter(
                            outPath + "TrainXGBBlend" + action,
                            this.config.appendToFile));
            validWriters[action.index] =
                    new BufferedWriter(new FileWriter(
                            outPath + "ValidXGBBlend" + action,
                            this.config.appendToFile));
        }

        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(0, trainIndexes.length + validIndexes.length).parallel().forEach(index -> {
            int engageIndex;
            boolean isValid = false;
            if (index < trainIndexes.length) {
                engageIndex = trainIndexes[index];
            } else {
                engageIndex = validIndexes[index - trainIndexes.length];
                isValid = true;
                if (this.config.skipValid == true) {
                    return;
                }
            }

            int curCount = counter.incrementAndGet();
            if (curCount % 1_000_000 == 0) {
                timer.tocLoop("train", curCount);
            }

            //features
//            List<MLSparseVector> featureList =
//                    this.featExtractor.extractFeatures(engageIndex);
            List<MLSparseVector> featureList = new LinkedList<>();

            int[] engage = this.data.engage[engageIndex];
            int tweetIndex = RecSys20Helper.getTweetIndex(engage);
            int userIndex = RecSys20Helper.getUserIndex(engage);
            int predsIndex = this.userTweetToIndex.get(
                    getId(tweetIndex, userIndex));

            for (int i = 0; i < this.modelPreds.length; i++) {
                featureList.add(new MLDenseVector(this.modelPreds[i][predsIndex]).toSparse());
            }
            String libSVM = RecSys20Helper.toLIBSVM(featureList, null);

            long[] engageAction = this.data.engageAction[engageIndex];
            for (EngageType action : ACTIONS) {
                int target = 0;
                if (engageAction != null && engageAction[action.index] > 0) {
                    target = 1;
                }
                if (isValid == true) {
                    try {
                        validWriters[action.index].write(target + libSVM +
                                "\n");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        trainWriters[action.index].write(target + libSVM +
                                "\n");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        timer.tocLoop("train", counter.get());

        for (EngageType action : ACTIONS) {
            trainWriters[action.index].close();
            validWriters[action.index].close();
        }
    }

    public static void trainXGB(final EngageType targetAction,
                                final String path,
                                final boolean highL2) throws Exception {
        //load libsvm
        DMatrix dmatTrain =
                new DMatrix(path + "TrainXGBBlend" + targetAction);
        DMatrix dmatValid =
                new DMatrix(path + "ValidXGBBlend" + targetAction);

        timer.toc("train rows " + dmatTrain.rowNum());
        timer.toc("valid rows " + dmatValid.rowNum());

        //set XGB parameters
        Map<String, Object> params = new HashMap<>();
        params.put("booster", "gbtree");
        params.put("verbosity", 2);
        params.put("eta", 0.1);
        params.put("gamma", 0);
        params.put("min_child_weight", 5);
        params.put("max_depth", 5);
        params.put("subsample", 1);
        params.put("colsample_bytree", 0.8);
        params.put("alpha", 0);
        if (highL2 == true) {
            params.put("lambda", 10000);
        } else {
            params.put("lambda", 1);
        }
        params.put("tree_method", "hist");
        params.put("max_bin", 256);
        params.put("seed", 5);
        params.put("objective", "binary:logistic");
        params.put("eval_metric", "aucpr");
        params.put("base_score", 0.1);
        params.put("use_buffer", 0);
        timer.toc("xbg params " + params.toString());

        //number of trees
        int nRounds = 50;
        if (highL2 == true) {
            nRounds = 2000;
        }
        timer.toc("nRounds " + nRounds);

        //set watches
        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("valid", dmatValid);

        //train
        Booster booster = XGBoost.train(
                dmatTrain,
                params,
                nRounds,
                watches,
                null,
                null);

        //save
        booster.saveModel(path + nRounds + ".modelBlend" + targetAction);

        //clean up
        booster.dispose();
        dmatTrain.dispose();
        dmatValid.dispose();
    }

    public void submitAVG(final String[] xgbModels,
                          final String lbFile) throws Exception {
        this.loadModelPreds("_submitLB");

        //load LB data
        RecSys20DataParser parser = new RecSys20DataParser(this.data);
        parser.parse(lbFile, 1);

        //submit xgb model for a specific action
        String[] indexToUser = new String[this.data.userToIndex.size()];
        for (Map.Entry<String, Integer> entry :
                this.data.userToIndex.entrySet()) {
            indexToUser[entry.getValue()] = entry.getKey();
        }

        String[] indexToTweet = new String[this.data.tweetToIndex.size()];
        for (Map.Entry<String, Integer> entry :
                this.data.tweetToIndex.entrySet()) {
            indexToTweet[entry.getValue()] = entry.getKey();
        }

        for (int i = 0; i < ACTIONS.length; i++) {
            try (BufferedWriter writer =
                         new BufferedWriter(new FileWriter(xgbModels[ACTIONS[i].index] + "_submit"))) {

                for (int engageIndex = this.data.lbEngageIndex; engageIndex < this.data.testEngageIndex; engageIndex++) {
                    int[] engage = this.data.engage[engageIndex];
                    int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                    int userIndex = RecSys20Helper.getUserIndex(engage);
                    int predsIndex = this.userTweetToIndex.get(
                            getId(tweetIndex, userIndex));

                    double score = 0;
                    float[] modelScores =
                            this.modelPreds[ACTIONS[i].index][predsIndex];
                    for (float modelScore : modelScores) {
                        score += modelScore;
                    }
                    if (modelScores.length > 1) {
                        score = score / modelScores.length;
                    }

                    writer.write(String.format("%s,%s,%.8f\n",
                            indexToTweet[tweetIndex],
                            indexToUser[userIndex],
                            score));
                }

            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("submit failed");
            }
        }
    }

    public void submit(final String[] xgbModels,
                       final String lbFile) throws Exception {
        this.loadModelPreds("_submitLB");

        //load LB data
        RecSys20DataParser parser = new RecSys20DataParser(this.data);
        parser.parse(lbFile, 1);

        //submit xgb model for a specific action
        String[] indexToUser = new String[this.data.userToIndex.size()];
        for (Map.Entry<String, Integer> entry :
                this.data.userToIndex.entrySet()) {
            indexToUser[entry.getValue()] = entry.getKey();
        }

        String[] indexToTweet = new String[this.data.tweetToIndex.size()];
        for (Map.Entry<String, Integer> entry :
                this.data.tweetToIndex.entrySet()) {
            indexToTweet[entry.getValue()] = entry.getKey();
        }

        MLSparseVector[] features =
                new MLSparseVector[this.data.testEngageIndex - this.data.lbEngageIndex];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(this.data.lbEngageIndex,
                this.data.testEngageIndex).parallel().forEach(engageIndex -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("submit", count);
            }

//            List<MLSparseVector> featureList =
//                    this.featExtractor.extractFeatures(engageIndex);
            List<MLSparseVector> featureList = new LinkedList<>();

            int[] engage = this.data.engage[engageIndex];
            int tweetIndex = RecSys20Helper.getTweetIndex(engage);
            int userIndex = RecSys20Helper.getUserIndex(engage);
            int predsIndex = this.userTweetToIndex.get(
                    getId(tweetIndex, userIndex));
            for (int i = 0; i < this.modelPreds.length; i++) {
                featureList.add(new MLDenseVector(this.modelPreds[i][predsIndex]).toSparse());
            }

            features[engageIndex - this.data.lbEngageIndex] =
                    RecSys20Helper.concat(featureList, null);
        });
        timer.tocLoop("submit", counter.get());

        for (int i = 0; i < ACTIONS.length; i++) {
            Booster xgb = null;
            DMatrix xgbMat = null;
            try (BufferedWriter writer =
                         new BufferedWriter(new FileWriter(xgbModels[ACTIONS[i].index] + "_submit"))) {

                xgb = XGBoost.loadModel(xgbModels[ACTIONS[i].index]);
                timer.toc("model loaded " + xgbModels[ACTIONS[i].index]);

                xgbMat = MLXGBoost.toDMatrix(new MLSparseMatrixAOO(
                        features, features[0].getLength()));
                float[][] xgbPreds = xgb.predict(xgbMat);
                timer.toc("submit xgb inference done " + ACTIONS[i]);

                for (int j = 0; j < xgbPreds.length; j++) {
                    int[] engage =
                            this.data.engage[this.data.lbEngageIndex + j];
                    writer.write(String.format("%s,%s,%.8f\n",
                            indexToTweet[RecSys20Helper.getTweetIndex(engage)],
                            indexToUser[RecSys20Helper.getUserIndex(engage)],
                            xgbPreds[j][0]));
                }

            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("xgb failed");
            } finally {
                xgb.dispose();
                xgbMat.dispose();
            }
        }
    }

    public static void main(final String[] args) {
        try {
            String path = "/media/mvolkovs/external4TB/Data/recsys2020/";
            if (args.length > 1) {
                path = args[1];
            }
            String xgbPath = path + "Models/XGB/";
            String predsPath = xgbPath + "preds/";
            String textFile = path + "Data/parsed_tweet_text.out";

            String modelPrefix = "50.modelBlend";

            RecSys20Config config = new RecSys20Config();
            config.path = path;
            config.modelPredFiles = new String[ACTIONS.length][];
            config.modelPredFiles[EngageType.REPLY.index] = new String[]{
                    predsPath + "200.modelREPLY",
                    predsPath + "3000.modelREPLY"
            };
            config.modelPredFiles[EngageType.RETWEET.index] = new String[]{
                    predsPath + "200.modelRETWEET",
                    predsPath + "3000.modelRETWEET"
            };
            config.modelPredFiles[EngageType.COMMENT.index] = new String[]{
                    predsPath + "200.modelCOMMENT",
                    predsPath + "3000.modelCOMMENT"
            };
            config.modelPredFiles[EngageType.LIKE.index] = new String[]{
                    predsPath + "200.modelLIKE",
                    predsPath + "3000.modelLIKE"
            };

            //saba split
            long[][] dates = new long[1][];
            dates[0] = new long[]{
                    RecSys20Split.VALID_DATE_CUTOFF_4H,
                    RecSys20Split.MAX_DATE - 1 * 60 * 60
            };

            //TRAIN XGB model from java
            if (args[0].startsWith("trainXGB") == true) {
                String[] split = args[0].split(":");
                boolean highL2 = Boolean.parseBoolean(split[2]);
                timer.toc("starting blend XGB TRAIN for ALL actions with " +
                        "highL2=" + highL2);
                for (EngageType targetAction : ACTIONS) {
                    timer.toc("starting blend XGB TRAIN for " + targetAction.toString());
                    RecSys20Blender.trainXGB(targetAction, xgbPath, highL2);
                }
            }

            //get blend libsvm
            if (args[0].startsWith("trainLIBSVM") == true) {
                timer.toc("starting blend LIBSVM TRAIN");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        path + "Data/parsed_transformed_1M.out",
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                config.removeTrain = false;
                config.removeValid = true;
                config.nChunks = dates.length;
                config.chunkIndex = 0;

                RecSys20Blender blender = new RecSys20Blender(
                        data,
                        textData,
                        config,
                        dates[0][0],
                        dates[0][1]);
                blender.train(xgbPath);
            }

            //SUBMIT
            if (args[0].startsWith("submit") == true) {
                timer.toc("starting blend SUBMIT");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        path + "Data/parsed_transformed_1M.out",
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                config.removeTrain = false;
                config.removeValid = false;
                config.nChunks = dates.length;
                config.chunkIndex = 0;

                RecSys20Blender blender = new RecSys20Blender(
                        data,
                        textData,
                        config,
                        dates[0][0],
                        dates[0][1]);
                timer.toc("submitting blend for ALL actions");

                String[] models = new String[ACTIONS.length];
                models[EngageType.REPLY.index] =
                        xgbPath + modelPrefix + EngageType.REPLY;
                models[EngageType.RETWEET.index] =
                        xgbPath + modelPrefix + EngageType.RETWEET;
                models[EngageType.COMMENT.index] =
                        xgbPath + modelPrefix + EngageType.COMMENT;
                models[EngageType.LIKE.index] =
                        xgbPath + modelPrefix + EngageType.LIKE;
                blender.submit(models,
                        path + "Data/val.tsv");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

//cat ValidXGBBlendCOMMENT >> TrainXGBBlendCOMMENT
//cat ValidXGBBlendLIKE >> TrainXGBBlendLIKE
//cat ValidXGBBlendREPLY >> TrainXGBBlendREPLY
//cat ValidXGBBlendRETWEET >> TrainXGBBlendRETWEET