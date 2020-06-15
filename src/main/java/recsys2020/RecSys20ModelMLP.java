package recsys2020;

import ai.layer6.ml.core.linalg.FloatElement;
import ai.layer6.ml.core.linalg.MLSparseVector;
import ai.layer6.ml.core.utils.MLIOUtils;
import ai.layer6.ml.core.utils.MLTimer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

//import java.util.Set;
//import ai.layer6.ml.xgb.MLXGBoost;

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

    public static String toCSV(final List<MLSparseVector> feats) {

        StringBuilder builder = new StringBuilder();
        for (MLSparseVector feat : feats) {
            if (feat.isEmpty() == true) {
                for(int i = 0; i < feat.getLength(); i++){
                    builder.append("0,");
                }
                continue;
            }

            int[] indexes = feat.getIndexes();
            int prevIndex = 0;
            float[] values = feat.getValues();
            for (int i = 0; i < indexes.length; i++) {
                for(int j = prevIndex; j < indexes[i]; j++) {
                    builder.append("0,");
                }
                float val = values[i];
                if (val == Math.round(val)) {
                    builder.append(((int) val));
                } else {
                    builder.append(String.format("%.5f", val));
                }
                builder.append(",");
                prevIndex=indexes[i]+1;
            }
            for (int i=prevIndex; i<feat.getLength(); i++){
                builder.append("0,");
            }
        }
        return builder.toString();
    }

    public void train(final String outPath) throws Exception {

        Integer[] trainIndexes = new Integer[this.split.trainIndexSet.size()];
        this.split.trainIndexSet.toArray(trainIndexes);
        Arrays.sort(trainIndexes);

        Integer[] validIndexes = new Integer[this.split.validIndexSet.size()];
        this.split.validIndexSet.toArray(validIndexes);
        Arrays.sort(validIndexes);

        String[] indexToTweet = new String[this.data.tweetToIndex.size()];
        for (Map.Entry<String, Integer> entry :
                this.data.tweetToIndex.entrySet()) {
            indexToTweet[entry.getValue()] = entry.getKey();
        }
        AtomicInteger featLength = new AtomicInteger(-1);
        AtomicInteger counter = new AtomicInteger(0);

        try (BufferedWriter trainWriter =
                     new BufferedWriter(new FileWriter(outPath +
                             "TrainXGB.csv", this.config.appendToFile));
             BufferedWriter validWriter =
                     new BufferedWriter(new FileWriter(outPath +
                             "ValidXGB.csv", this.config.appendToFile))) {
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

                long[] engageAction = this.data.engageAction[engageIndex];
                List<MLSparseVector> features =
                        this.featExtractor.extractFeatures(engageIndex);

                //get original tweet id
                String tweet = indexToTweet[RecSys20Helper.getTweetIndex(this.data.engage[engageIndex])];
                int cId = RecSys20Helper.getCreatorIndex(this.data.engage[engageIndex]);
                int uId = RecSys20Helper.getUserIndex(this.data.engage[engageIndex]);

                if (featLength.get() < 0) {
                    int length = 0;
                    for (MLSparseVector feature : features) {
                        length += feature.getLength();
                    }
                    featLength.set(length);
                }

                //String featLIBSVM = RecSys20Helper.toCSV(features);
                String featCSV = this.toCSV(features);
                StringBuilder targetAll = new StringBuilder();

                for (int i = 0; i < ACTIONS.length; i++) {
                    int target = 0;
                    if (engageAction != null && engageAction[ACTIONS[i].index] > 0) {
                        target = 1;
                    }
                    targetAll.append(target);
                    targetAll.append(",");
                }
                /*
                Set<Integer>[] targetUserEngage = this.featExtractor.userToEngage[uId];
                StringBuilder tweetHistory = new StringBuilder();

                for (int i = 0; i < ACTIONS.length; i++) {
                    String tidHis;
                    if (targetUserEngage[i] != null) {
                        for (int engageId : targetUserEngage[i]) {
                            tidHis = indexToTweet[RecSys20Helper.getTweetIndex(this.data.engage[engageId])];
                            tweetHistory.append(tidHis);
                            tweetHistory.append("|");
                        }
                    }
                    tweetHistory.append(",");
                }
                */

                //String csv =
                //        targetAll + featCSV + tweet + ',' + tweetHistory + uId + ',' + cId + "\n";

                String csv =
                        targetAll + featCSV + tweet + ',' + uId + ',' + cId + "\n";

                if (isValid == true) {
                    try {
                        validWriter.write(csv);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    try {
                        trainWriter.write(csv);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }
    }


    public void validate(final String outPath) {
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

        AtomicInteger counter = new AtomicInteger(0);

        try (BufferedWriter submitWriter =
                     new BufferedWriter(new FileWriter(outPath +
                             "ValidXGB.csv", this.config.appendToFile))){
            IntStream.range(0, validIndexes.length).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 1_000_000 == 0) {
                    timer.tocLoop("valid", count);
                }
                int validIndex = validIndexes[index];

                List<MLSparseVector> feats =
                        this.featExtractor.extractFeatures(validIndex);
                long[] engageAction = this.data.engageAction[validIndex];
                String featCSV = this.toCSV(feats);
                StringBuilder targetAll = new StringBuilder();

                for (int i = 0; i < ACTIONS.length; i++) {
                    int target = 0;
                    if (engageAction != null && engageAction[ACTIONS[i].index] > 0) {
                        target = 1;
                    }
                    targetAll.append(target);
                    targetAll.append(",");
                }
                String tweet = indexToTweet[RecSys20Helper.getTweetIndex(this.data.engage[validIndex])];
                int cId = RecSys20Helper.getCreatorIndex(this.data.engage[validIndex]);
                int uId = RecSys20Helper.getUserIndex(this.data.engage[validIndex]);

                String user = indexToUser[uId];
                String creator = indexToUser[cId];
                String csv =
                        targetAll + featCSV + tweet + ',' + uId + ',' + cId + ',' + user + "\n";
                try {
                    submitWriter.write(csv);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void submitLB(final String outPath) throws Exception {
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

        AtomicInteger counter = new AtomicInteger(0);

        try (BufferedWriter submitWriter =
                     new BufferedWriter(new FileWriter(outPath +
                             "Submit.csv", this.config.appendToFile))){
            IntStream.range(this.data.lbEngageIndex, this.data.testEngageIndex).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 1_000_000 == 0) {
                    timer.tocLoop("submit", count);
                }

                List<MLSparseVector> feats =
                        this.featExtractor.extractFeatures(index);
                String featCSV = this.toCSV(feats);
                String tweet = indexToTweet[RecSys20Helper.getTweetIndex(this.data.engage[index])];
                int cId = RecSys20Helper.getCreatorIndex(this.data.engage[index]);
                int uId = RecSys20Helper.getUserIndex(this.data.engage[index]);

                String user = indexToUser[uId];
                String creator = indexToUser[cId];
                String csv =
                        featCSV + tweet + ',' + uId + ',' + cId + ',' + user + "\n";
                try {
                    submitWriter.write(csv);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
    }

    public void submitTest(final String outPath) throws Exception{
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

        AtomicInteger counter = new AtomicInteger(0);

        try (BufferedWriter submitWriter =
                     new BufferedWriter(new FileWriter(outPath +
                             "Test.csv", this.config.appendToFile))){
            IntStream.range(this.data.testEngageIndex, this.data.testEngageIndexEnd).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 1_000_000 == 0) {
                    timer.tocLoop("Test", count);
                }

                List<MLSparseVector> feats =
                        this.featExtractor.extractFeatures(index);
                String featCSV = this.toCSV(feats);
                String tweet = indexToTweet[RecSys20Helper.getTweetIndex(this.data.engage[index])];
                int cId = RecSys20Helper.getCreatorIndex(this.data.engage[index]);
                int uId = RecSys20Helper.getUserIndex(this.data.engage[index]);

                String user = indexToUser[uId];
                String creator = indexToUser[cId];
                String csv =
                        featCSV + tweet + ',' + uId + ',' + cId + ',' + user + "\n";
                try {
                    submitWriter.write(csv);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
    }

    public static void main(final String[] args) {
        try {
            String path = "/data/recsys2020/Data/";
            if (args.length > 1) {
                path = args[1];
            }
            String xgbPath = path + "Models/DL/";
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

            //TRAIN libsvm with multiple date intervals

            if (args[0].startsWith("train") == true) {
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
                    config.chunkIndex = i;
                    config.nChunks= dates.length;

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
                timer.toc("starting VALID");
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
                model.validate(xgbPath);
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
                model.submitLB(xgbPath);
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

                model.submitTest(xgbPath);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
