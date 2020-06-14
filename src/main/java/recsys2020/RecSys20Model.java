package recsys2020;

import common.core.linalg.FloatElement;
import common.core.linalg.MLSparseVector;
import common.core.utils.MLIOUtils;
import common.core.utils.MLTimer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import recsys2020.RecSys20Data.EngageType;

public class RecSys20Model {

    public static EngageType[] ACTIONS = EngageType.values();
    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20Model");
        timer.tic();
    }

    public static class RecSys20Config {

        public boolean removeTrain = true;
        public boolean removeValid = true;

        public boolean skipValid = false;
        public boolean appendToFile = false;
        public int nChunks = 0;
        public int chunkIndex = 0;

        public String path;

        public String[][] modelPredFiles;

//        public int[][] groupIndexesRemove = new int[][]{
//                {14, 15, 23, 24}, // REPLY
//                {5, 6, 10, 11, 22, 24}, // RETWEET
//                {25}, // COMMENT
//                {6, 9, 10, 14, 15, 17, 21, 22, 23, 24}, // LIKE
//        };

        public int[][] groupIndexesRemove = new int[][]{
                null, // REPLY
                null, // RETWEET
                null, // COMMENT
                null, // LIKE
        };

        public RecSys20Config() {
        }
    }

    public RecSys20Data data;
    public RecSys20TextData textData;
    public RecSys20Split split;
    public RecSys20Config config;
    public RecSys20FeatExtractor featExtractor;

    public RecSys20Model(final RecSys20Data dataP,
                         final RecSys20TextData textDataP,
                         final RecSys20Config configP,
                         final long trainDate,
                         final long validDate) throws Exception {

        this.data = dataP;
        this.textData = textDataP;
        this.config = configP;

        //generate split
        this.split = new RecSys20Split(this.config, trainDate, validDate);
//        this.split.splitWithCache(this.data);
        this.split.splitByCreateDate(this.data);

        //init feat extractor
        this.featExtractor = new RecSys20FeatExtractor(
                this.data,
                this.textData,
                this.split,
                this.config);
    }

    public void train(final String outPath) throws Exception {

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
                            outPath + "TrainXGB" + action,
                            this.config.appendToFile));
            validWriters[action.index] =
                    new BufferedWriter(new FileWriter(
                            outPath + "ValidXGB" + action,
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
            if (curCount % 2_000_000 == 0) {
                timer.tocLoop("train", curCount);
            }

            //features
            List<MLSparseVector> features =
                    this.featExtractor.extractFeatures(engageIndex);
            String libSVM = RecSys20Helper.toLIBSVM(features, null);

            long[] engageAction = this.data.engageAction[engageIndex];
            for (EngageType action : ACTIONS) {
                String actionLIBSVM = libSVM;
                if (this.config.groupIndexesRemove[action.index] != null) {
                    actionLIBSVM = RecSys20Helper.toLIBSVM(features,
                            this.config.groupIndexesRemove[action.index]);
                }

                int target = 0;
                if (engageAction != null && engageAction[action.index] > 0) {
                    target = 1;
                }
                if (isValid == true) {
                    try {
                        validWriters[action.index].write(target + actionLIBSVM +
                                "\n");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        trainWriters[action.index].write(target + actionLIBSVM +
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
                new DMatrix(path + "TrainXGB" + targetAction);
        DMatrix dmatValid =
                new DMatrix(path + "ValidXGB" + targetAction);

        timer.toc("train rows " + dmatTrain.rowNum());
        timer.toc("valid rows " + dmatValid.rowNum());

        //set XGB parameters
        Map<String, Object> params = new HashMap<>();
        params.put("booster", "gbtree");
        params.put("verbosity", 2);
        params.put("eta", 0.1);
        params.put("gamma", 0);
        if (highL2 == true) {
            params.put("min_child_weight", 5);
        } else {
            params.put("min_child_weight", 20);
        }
        params.put("max_depth", 15);
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
        int nRounds = 200;
        if (highL2 == true) {
            nRounds = 3000;
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
        booster.saveModel(path + nRounds + ".model" + targetAction);

        //clean up
        booster.dispose();
        dmatTrain.dispose();
        dmatValid.dispose();
    }

    public void validate(final String[] xgbModels) {
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

        Integer[] validIndexes = new Integer[this.split.validIndexSet.size()];
        this.split.validIndexSet.toArray(validIndexes);
        Arrays.sort(validIndexes);

        List<MLSparseVector>[] features = new List[validIndexes.length];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(0, validIndexes.length).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("validate", count);
            }

            int validIndex = validIndexes[index];
            features[index] = this.featExtractor.extractFeatures(validIndex);
        });
        timer.tocLoop("validate", counter.get());

        for (int i = 0; i < ACTIONS.length; i++) {
            Booster xgb = null;
            DMatrix xgbMat = null;
            FloatElement[] preds =
                    new FloatElement[validIndexes.length];
            try (BufferedWriter writer =
                         new BufferedWriter(new FileWriter(xgbModels[ACTIONS[i].index] + "_valid"))) {

                xgb = XGBoost.loadModel(xgbModels[ACTIONS[i].index]);
                xgbMat = RecSys20Helper.toDMatrix(features,
                        this.config.groupIndexesRemove[ACTIONS[i].index],
                        null,
                        -1);
                float[][] xgbPreds = xgb.predict(xgbMat);
                timer.toc("validate xgb inference done " + ACTIONS[i]);

                for (int j = 0; j < validIndexes.length; j++) {
                    preds[j] = new FloatElement(validIndexes[j],
                            xgbPreds[j][0]);

                    int[] engage = this.data.engage[validIndexes[j]];
                    writer.write(String.format("%s,%s,%.8f\n",
                            indexToTweet[RecSys20Helper.getTweetIndex(engage)],
                            indexToUser[RecSys20Helper.getUserIndex(engage)],
                            xgbPreds[j][0]));
                }
                Arrays.sort(preds, new FloatElement.ValueComparator(true));

            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("xgb failed");
            } finally {
                xgb.dispose();
                xgbMat.dispose();
            }

            //evaluate
            RecSys20Eval eval = new RecSys20Eval();
            eval.evaluate(ACTIONS[i], preds, this.data);
        }
    }

    public void submitLB(final EngageType targetAction,
                         final String xgbModel) throws Exception {
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

        List<MLSparseVector>[] features =
                new List[this.data.testEngageIndex - this.data.lbEngageIndex];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(this.data.lbEngageIndex,
                this.data.testEngageIndex).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("submit", count);
            }

            features[index - this.data.lbEngageIndex] =
                    this.featExtractor.extractFeatures(index);
        });
        timer.tocLoop("submit", counter.get());

        Booster xgb = null;
        DMatrix xgbMat = null;
        try (BufferedWriter writer =
                     new BufferedWriter(new FileWriter(xgbModel + "_submit"))) {

            xgb = XGBoost.loadModel(xgbModel);
            xgbMat = RecSys20Helper.toDMatrix(features,
                    this.config.groupIndexesRemove[targetAction.index],
                    null,
                    -1);

            float[][] xgbPreds = xgb.predict(xgbMat);
            timer.toc("submit xgb inference done " + targetAction);

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


    public void submitLB(final String[] xgbModels) {
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

        List<MLSparseVector>[] features =
                new List[this.data.testEngageIndex - this.data.lbEngageIndex];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(this.data.lbEngageIndex,
                this.data.testEngageIndex).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("submitLB", count);
            }
            features[index - this.data.lbEngageIndex] =
                    this.featExtractor.extractFeatures(index);
        });
        timer.tocLoop("submitLB", counter.get());

        for (int i = 0; i < ACTIONS.length; i++) {
            Booster xgb = null;
            DMatrix xgbMat = null;
            try (BufferedWriter writer =
                         new BufferedWriter(new FileWriter(xgbModels[ACTIONS[i].index] + "_submitLB"))) {

                xgb = XGBoost.loadModel(xgbModels[ACTIONS[i].index]);
                xgbMat = RecSys20Helper.toDMatrix(features,
                        this.config.groupIndexesRemove[ACTIONS[i].index],
                        null,
                        -1);
                float[][] xgbPreds = xgb.predict(xgbMat);
                timer.toc("submitLB xgb inference done " + ACTIONS[i]);

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

    public void submitTest(final String[] xgbModels) {
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

        List<MLSparseVector>[] features =
                new List[this.data.testEngageIndexEnd - this.data.testEngageIndex];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(this.data.testEngageIndex,
                this.data.testEngageIndexEnd).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("submitTest", count);
            }
            features[index - this.data.testEngageIndex] =
                    this.featExtractor.extractFeatures(index);
        });
        timer.tocLoop("submitTest", counter.get());

        for (int i = 0; i < ACTIONS.length; i++) {
            Booster xgb = null;
            DMatrix xgbMat = null;
            try (BufferedWriter writer =
                         new BufferedWriter(new FileWriter(xgbModels[ACTIONS[i].index] + "_submitTest"))) {

                xgb = XGBoost.loadModel(xgbModels[ACTIONS[i].index]);
                xgbMat = RecSys20Helper.toDMatrix(features,
                        this.config.groupIndexesRemove[ACTIONS[i].index],
                        null,
                        -1);
                float[][] xgbPreds = xgb.predict(xgbMat);
                timer.toc("submitTest xgb inference done " + ACTIONS[i]);

                for (int j = 0; j < xgbPreds.length; j++) {
                    int[] engage =
                            this.data.engage[this.data.testEngageIndex + j];
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
            String path = "/data/recsys2020/Data/";
            if (args.length > 1) {
                path = args[1];
            }
            String xgbPath = path + "Models/XGB/";
            String modelPrefix = "200.model";
            String dataFile = path + "Data/parsed_transformed_1M.out";
            String textFile = path + "Data/parsed_tweet_text.out";

//            //to test features
//            long[][] dates = RecSys20Helper.getTrainValidDates(
//                    24 * 60 * 60,
//                    1,
//                    RecSys20Split.VALID_DATE_CUTOFF_4H);

//            //best chunk
//            long[][] dates = RecSys20Helper.getTrainValidDates(
//                    8 * 60 * 60,
//                    15,
//                    RecSys20Split.VALID_DATE_CUTOFF_4H);

            //saba split
            long[][] dates = RecSys20Helper.getTrainValidDates(
                    24 * 60 * 60,
                    5,
                    RecSys20Split.VALID_DATE_CUTOFF_4H);

//            //blind training
//            long[][] dates = RecSys20Helper.getTrainValidDates(
//                    8 * 60 * 60,
//                    15,
//                    RecSys20Split.MAX_DATE);

            RecSys20Config config = new RecSys20Config();
            config.path = path;

            //TRAIN XGB model from java
            if (args[0].startsWith("trainXGB") == true) {
                String[] split = args[0].split(":");

                boolean highL2 = Boolean.parseBoolean(split[2]);
                if (split[1].equals("ALL") == true) {
                    timer.toc("starting XGB TRAIN for ALL actions with " +
                            "highL2=" + highL2);
                    for (EngageType targetAction : ACTIONS) {
                        timer.toc("starting XGB TRAIN for " + targetAction.toString());
                        RecSys20Model.trainXGB(targetAction, xgbPath,
                                highL2);
                    }
                } else {
                    EngageType targetAction =
                            EngageType.fromString(split[1]);
                    timer.toc("starting XGB TRAIN for " + targetAction.toString() +
                            " with highL2=" + highL2);
                    RecSys20Model.trainXGB(targetAction, xgbPath, highL2);
                }
            }

            //TRAIN libsvm with multiple date intervals
            if (args[0].startsWith("trainLIBSVM") == true) {
                timer.toc("starting LIBSVM TRAIN");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        dataFile,
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                for (int i = 0; i < dates.length; i++) {
                    if (i == 0) {
                        config.skipValid = false;
                        config.appendToFile = false;
                    } else {
                        config.skipValid = true;
                        config.appendToFile = true;
                    }
                    config.nChunks = dates.length;
                    config.chunkIndex = i;

                    RecSys20Model model = new RecSys20Model(
                            data,
                            textData,
                            config,
                            dates[i][0],
                            dates[i][1]);
                    model.train(xgbPath);
                }
            }

            //VALIDATE
            if (args[0].startsWith("valid") == true) {
                timer.toc("starting VALID for ALL actions");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        dataFile,
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                config.removeTrain = false;
                config.removeValid = true;
                config.nChunks = dates.length;
                config.chunkIndex = 0;

                RecSys20Model model = new RecSys20Model(
                        data,
                        textData,
                        config,
                        dates[0][0],
                        dates[0][1]);

                String[] models = new String[ACTIONS.length];
                models[EngageType.REPLY.index] =
                        xgbPath + modelPrefix + EngageType.REPLY;
                models[EngageType.RETWEET.index] =
                        xgbPath + modelPrefix + EngageType.RETWEET;
                models[EngageType.COMMENT.index] =
                        xgbPath + modelPrefix + EngageType.COMMENT;
                models[EngageType.LIKE.index] =
                        xgbPath + modelPrefix + EngageType.LIKE;
                model.validate(models);
            }

            //SUBMIT
            if (args[0].startsWith("submit") == true) {
                timer.toc("starting SUBMIT LB");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        dataFile,
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                config.removeTrain = false;
                config.removeValid = false;
                config.nChunks = dates.length;
                config.chunkIndex = 0;

                String[] split = args[0].split(":");
                RecSys20Model model = new RecSys20Model(
                        data,
                        textData,
                        config,
                        dates[0][0],
                        dates[0][1]);
                if (split[1].equals("ALL") == true) {
                    timer.toc("submitting LB for ALL actions");

                    String[] models = new String[ACTIONS.length];
                    models[EngageType.REPLY.index] =
                            xgbPath + modelPrefix + EngageType.REPLY;
                    models[EngageType.RETWEET.index] =
                            xgbPath + modelPrefix + EngageType.RETWEET;
                    models[EngageType.COMMENT.index] =
                            xgbPath + modelPrefix + EngageType.COMMENT;
                    models[EngageType.LIKE.index] =
                            xgbPath + modelPrefix + EngageType.LIKE;
                    model.submitLB(models);

                } else {
                    EngageType targetAction = EngageType.fromString(split[1]);
                    timer.toc("submitting for " + targetAction);
                    model.submitLB(targetAction,
                            xgbPath + modelPrefix + targetAction);
                }
            }

            //TEST
            if (args[0].startsWith("test") == true) {
                timer.toc("starting SUBMIT TEST");
                RecSys20Data data = MLIOUtils.readObjectFromFile(
                        dataFile,
                        RecSys20Data.class);
                RecSys20TextData textData = MLIOUtils.readObjectFromFile(
                        textFile,
                        RecSys20TextData.class);
                timer.toc("data loaded");

                config.removeTrain = false;
                config.removeValid = false;
                config.nChunks = dates.length;
                config.chunkIndex = 0;

                RecSys20Model model = new RecSys20Model(
                        data,
                        textData,
                        config,
                        dates[0][0],
                        dates[0][1]);
                timer.toc("submitting test for ALL actions");

                String[] models = new String[ACTIONS.length];
                models[EngageType.REPLY.index] =
                        xgbPath + modelPrefix + EngageType.REPLY;
                models[EngageType.RETWEET.index] =
                        xgbPath + modelPrefix + EngageType.RETWEET;
                models[EngageType.COMMENT.index] =
                        xgbPath + modelPrefix + EngageType.COMMENT;
                models[EngageType.LIKE.index] =
                        xgbPath + modelPrefix + EngageType.LIKE;
                model.submitTest(models);
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}