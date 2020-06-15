package recsys2020;

import common.core.feature.MLFeatureTransform;
import common.core.feature.MLSparseFeature;
import common.core.linalg.MLSparseMatrixAOO;
import common.core.linalg.MLSparseMatrixFlat;
import common.core.linalg.MLSparseVector;
import common.core.utils.MLIOUtils;
import common.core.utils.MLTimer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import recsys2020.RecSys20Data.TweetFeature;


public class RecSys20DataParser {

    public static final int TWEET_INDEX = 2;
    public static final int CREATOR_USER_INDEX = 9;
    public static final int ENGAGING_USER_INDEX = 14;

    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20DataParser");
        timer.tic();
    }

    public RecSys20Data data;

    public RecSys20DataParser() {
        this.data = new RecSys20Data();
    }

    public RecSys20DataParser(final RecSys20Data dataP) {
        this.data = dataP;
    }

    public void getAllUsersAndTweets(final String[] files) throws Exception {

        //get all unique user tweet and user ids
        this.data.userToIndex = new HashMap();
        int userIndex = 0;

        this.data.tweetToIndex = new HashMap();
        int tweetIndex = 0;

        int counter = 0;
        for (String file : files) {
            try (BufferedReader reader =
                         new BufferedReader(new FileReader(file))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    counter++;
                    if (counter % 10_000_000 == 0) {
                        this.timer.tocLoop("getAllUsersAndTweets " + file,
                                counter);
                    }
                    String[] split = line.split("\u0001");

                    if (this.data.tweetToIndex.containsKey(split[TWEET_INDEX]) == false) {
                        this.data.tweetToIndex.put(split[TWEET_INDEX],
                                tweetIndex);
                        tweetIndex++;
                    }

                    if (this.data.userToIndex.containsKey(split[CREATOR_USER_INDEX]) == false) {
                        this.data.userToIndex.put(split[CREATOR_USER_INDEX],
                                userIndex);
                        userIndex++;
                    }

                    if (this.data.userToIndex.containsKey(split[ENGAGING_USER_INDEX]) == false) {
                        this.data.userToIndex.put(split[ENGAGING_USER_INDEX],
                                userIndex);
                        userIndex++;
                    }
                }
            }
        }

        this.timer.tocLoop("getAllUsersAndTweets ", counter);
        timer.toc("Unique users " + this.data.userToIndex.size());
        timer.toc("Unique tweets " + this.data.tweetToIndex.size());
    }

    public void initTweetFeatures() {
        this.data.tweetFeatures = new HashMap();
        for (TweetFeature featName : TweetFeature.values()) {

            if (featName.equals(TweetFeature.text_tokens) ||
                    featName.equals(TweetFeature.hashtags) ||
                    featName.equals(TweetFeature.present_media) ||
                    featName.equals(TweetFeature.present_links) ||
                    featName.equals(TweetFeature.present_domains)) {
                MLSparseFeature feature =
                        new MLSparseFeature(this.data.tweetToIndex.size(),
                                null,
                                null, MLSparseMatrixAOO.class);
                this.data.tweetFeatures.put(featName, feature);

            } else {
                MLSparseFeature feature =
                        new MLSparseFeature(this.data.tweetToIndex.size(),
                                null,
                                null, MLSparseMatrixFlat.class);
                this.data.tweetFeatures.put(featName, feature);

            }
        }
    }

    public void parse(final String file,
                      final int type) throws Exception {
        //TRAIN:type=0   LB:type=1   TEST:type=2
        //data load must be called in order

        AtomicInteger engageCounter = new AtomicInteger(0);
        switch (type) {
            case 0: {
                //TRAIN
                //tweet data
                this.data.tweetCreation =
                        new long[this.data.tweetToIndex.size()];
                this.initTweetFeatures();

                //user data
                this.data.userCreation =
                        new long[this.data.userToIndex.size()];

                //engage data
                this.data.engage = new int[190_000_000][];
                this.data.engageAction = new long[190_000_000][];
                break;
            }
            case 1: {
                //LB
                engageCounter.set(this.data.lbEngageIndex);
                break;
            }
            case 2: {
                //TEST
                engageCounter.set(this.data.testEngageIndex);
                break;
            }
        }

        AtomicInteger lineCounter = new AtomicInteger(0);
        try (BufferedReader reader =
                     new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int count = lineCounter.incrementAndGet();
                if (count % 5_000_000 == 0) {
                    timer.tocLoop("parse", count);
                }
                String[] split = line.split("\u0001", -1);

                int tweetIndex =
                        this.data.tweetToIndex.get(split[TWEET_INDEX]);
                int creatorIndex =
                        this.data.userToIndex.get(split[CREATOR_USER_INDEX]);
                int userIndex =
                        this.data.userToIndex.get(split[ENGAGING_USER_INDEX]);

                //////////////////// parse tweet data
                long tweetCreation = Long.parseLong(split[8]);
                synchronized (this.data.tweetCreation) {
                    if (this.data.tweetCreation[tweetIndex] != 0) {
                        if (this.data.tweetCreation[tweetIndex] != tweetCreation) {
                            System.out.println("creation doesn't match");
                        }
                    } else {
                        this.data.tweetCreation[tweetIndex] = tweetCreation;

                        for (Map.Entry<TweetFeature, MLSparseFeature> entry :
                                this.data.tweetFeatures.entrySet()) {
                            TweetFeature featureName = entry.getKey();
                            MLSparseFeature feature = entry.getValue();

                            String data = split[featureName.index].trim();
                            if (data.length() == 0) {
                                continue;
                            }

                            if (type == 0) {
                                feature.addRow(tweetIndex, data.split("\t"));
                                continue;
                            }

                            if (feature.infMode() == false) {
                                feature.addRow(tweetIndex, data.split("\t"));
                                continue;
                            }

                            String[] featSplit = data.split("\t");
                            feature.addRow(tweetIndex, featSplit);
                            MLSparseVector rowInf =
                                    feature.getFeatInf(featSplit);
                            if (rowInf.isEmpty() == false) {
                                feature.getFeatMatrixTransformed().setRow
                                        (rowInf, tweetIndex);
                            } else {
                                feature.getFeatMatrixTransformed().setRow
                                        (null, tweetIndex);
                            }
                        }
                    }
                }

                //////////////////// parse user data
                int creatorFollowers = Integer.parseInt(split[10]);
                int creatorFollowing = Integer.parseInt(split[11]);
                int creatorVerified = 0;
                if (Boolean.parseBoolean(split[12]) == true) {
                    creatorVerified = 1;
                }
                synchronized (this.data.userCreation) {
                    long userCreation = Long.parseLong(split[13]);
                    if (this.data.userCreation[creatorIndex] != 0) {
                        if (this.data.userCreation[creatorIndex] != userCreation) {
                            System.out.println("creation doesn't match");
                        }
                    } else {
                        this.data.userCreation[creatorIndex] = userCreation;
                    }
                }

                int useFollowers = Integer.parseInt(split[15]);
                int userFollowing = Integer.parseInt(split[16]);
                int userVerified = 0;
                if (Boolean.parseBoolean(split[17]) == true) {
                    userVerified = 1;
                }
                synchronized (this.data.userCreation) {
                    long userCreation = Long.parseLong(split[18]);
                    if (this.data.userCreation[userIndex] != 0) {
                        if (this.data.userCreation[userIndex] != userCreation) {
                            System.out.println("creation doesn't match");
                        }
                    } else {
                        this.data.userCreation[userIndex] = userCreation;
                    }
                }

                //////////////////// parse engage data
                int engageIndex = engageCounter.getAndIncrement();
                int follow = 0;
                if (Boolean.parseBoolean(split[19]) == true) {
                    follow = 1;
                }
                int[] engage = new int[]{
                        tweetIndex,
                        creatorIndex,
                        creatorFollowers,
                        creatorFollowing,
                        creatorVerified,
                        userIndex,
                        useFollowers,
                        userFollowing,
                        userVerified,
                        follow};
                this.data.engage[engageIndex] = engage;

                if (split.length > 20) {
                    //engagement timestamp
                    long[] engageAction = new long[4];
                    boolean hasAction = false;
                    for (int i = 0; i < 4; i++) {
                        if (split[20 + i].length() > 0) {
                            long timestamp = Long.parseLong(split[20 + i]);
                            engageAction[i] = timestamp;
                            hasAction = true;
                        }
                    }
                    if (hasAction == true) {
                        this.data.engageAction[engageIndex] = engageAction;
                    }
                }
            }
        }
        //set boundaries
        switch (type) {
            case 0: {
                //TRAIN
                this.data.lbEngageIndex = engageCounter.get();
                break;
            }
            case 1: {
                //LB
                this.data.testEngageIndex = engageCounter.get();
                break;
            }
            case 2: {
                //TEST
                this.data.testEngageIndexEnd = engageCounter.get();
                break;
            }
        }

    }


    public static void applyTransforms(final String path) throws Exception {
        RecSys20Data data = MLIOUtils.readObjectFromFile(path + "parsed.out",
                RecSys20Data.class);
        timer.toc("data loaded");

        //apply transforms
        for (Map.Entry<TweetFeature, MLSparseFeature> entry :
                data.tweetFeatures.entrySet()) {
            if (entry.getKey().equals(TweetFeature.text_tokens) == true) {
                entry.getValue().setFeatureTransforms(new MLFeatureTransform[]{new MLFeatureTransform.ColSelectorTransform(1_000_000)});
                entry.getValue().finalizeFeature(false);
            } else {
                entry.getValue().setFeatureTransforms(new MLFeatureTransform[]{new MLFeatureTransform.ColSelectorTransform(1_000_000)});
                entry.getValue().finalizeFeature(true);
            }
            timer.toc(entry.getKey() + " " + entry.getValue().getFeatMatrixTransformed().getNCols());
        }

        RecSys20DataParser parser = new RecSys20DataParser(data);
        parser.parse(path + "val.tsv", 1);
        parser.parse(path + "competition_test.tsv", 2);

        MLIOUtils.writeObjectToFile(data,
                path + "parsed_transformed_1M.out");
    }

    public static void main(final String[] args) {
        try {
            String path = "/data/recsys2020/Data/";
            if (args.length > 0) {
                path = args[0];
            }

            //parse data
            RecSys20DataParser parser = new RecSys20DataParser();
            parser.getAllUsersAndTweets(new String[]{
                    path + "training.tsv",
                    path + "val.tsv",
                    path + "competition_test.tsv"});
            parser.parse(path + "training.tsv", 0);
            MLIOUtils.writeObjectToFile(parser.data,
                    path + "parsed.out");

            //apply transform and save
            for (Map.Entry<TweetFeature, MLSparseFeature> entry :
                    parser.data.tweetFeatures.entrySet()) {
                if (entry.getKey().equals(TweetFeature.text_tokens) == true) {
                    entry.getValue().setFeatureTransforms(new MLFeatureTransform[]{new MLFeatureTransform.ColSelectorTransform(1_000_000)});
                    entry.getValue().finalizeFeature(false);
                } else {
                    entry.getValue().setFeatureTransforms(new MLFeatureTransform[]{new MLFeatureTransform.ColSelectorTransform(1_000_000)});
                    entry.getValue().finalizeFeature(true);
                }
                timer.toc(entry.getKey() + " " + entry.getValue().getFeatMatrixTransformed().getNCols());
            }

            parser.parse(path + "val.tsv", 1);
            parser.parse(path + "competition_test.tsv", 2);

            MLIOUtils.writeObjectToFile(parser.data,
                    path + "parsed_transformed_1M.out");

//            //use this to try different transforms
//            RecSys20DataParser parser = new RecSys20DataParser();
//            parser.applyTransforms(path);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}