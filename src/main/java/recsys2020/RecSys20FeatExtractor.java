package recsys2020;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;

import common.core.linalg.MLDenseVector;
import common.core.linalg.MLSparseVector;
import common.core.utils.MLTimer;
import recsys2020.RecSys20Data.EngageType;
import recsys2020.RecSys20Data.TweetFeature;
import recsys2020.RecSys20Model.RecSys20Config;

public class RecSys20FeatExtractor {

    public static class TargetRecord {

        public int[] targetEngage;
        public long[] targetAction;

        public int targetTweetIndex;
        public int targetUserIndex;
        public int targetCreatorIndex;

    }

    public static long MIN_DATE = 1580947211;
    public static int N_ENGAGE = EngageType.values().length;
    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20FeatExtractor");
        timer.tic();
    }

    public static TweetFeature[] TWEET_FEATURES_TO_USE = new TweetFeature[]{
            TweetFeature.text_tokens,
            TweetFeature.present_media,
            TweetFeature.tweet_type,
            TweetFeature.language
    };

    //data
    public RecSys20Split split;
    public RecSys20Data data;
    public RecSys20TextData textData;
    public RecSys20Config config;

    //cache: allows fast indexing into data.engage array
    public Set<Integer>[][] userToCreate;
    public Set<Integer>[][] userToEngage;

    public int[][] engageCache;
    public int[][] createCache;

    public int[][] engageLBCache;
    public int[][] createLBCache;

    public RecSys20NeighborCF cf;
//    public RecSys20Embedding embed;


    public RecSys20FeatExtractor(final RecSys20Data dataP,
                                 final RecSys20TextData textDataP,
                                 final RecSys20Split splitP,
                                 final RecSys20Config configP) throws Exception {
        this.data = dataP;
        this.textData = textDataP;
        this.split = splitP;
        this.config = configP;

        this.initCache();

        this.cf = new RecSys20NeighborCF(this.data, this.split, this.config);
        this.cf.initR(this.userToEngage);

//        this.embed = new RecSys20Embedding(this);

    }

    public void initCache() throws Exception {

        this.userToCreate =
                new Set[this.data.userToIndex.size()][N_ENGAGE];
        this.userToEngage =
                new Set[this.data.userToIndex.size()][N_ENGAGE];

        this.engageCache = new int[this.data.userToIndex.size()][2];
        this.createCache = new int[this.data.userToIndex.size()][2];

        this.engageLBCache = new int[this.data.userToIndex.size()][2];
        this.createLBCache = new int[this.data.userToIndex.size()][2];

        IntStream.range(0, this.data.testEngageIndex).parallel().forEach(index -> {
            if (index >= this.data.lbEngageIndex) {
                int[] engage = this.data.engage[index];
                int creatorIndex = RecSys20Helper.getCreatorIndex(engage);
                int userIndex = RecSys20Helper.getUserIndex(engage);

                synchronized (this.engageLBCache[userIndex]) {
                    this.engageLBCache[userIndex][0]++;
                }
                synchronized (this.createLBCache[creatorIndex]) {
                    this.createLBCache[creatorIndex][0]++;
                }
                return;
            }

            int[] engage = this.data.engage[index];
            int creatorIndex = RecSys20Helper.getCreatorIndex(engage);
            int userIndex = RecSys20Helper.getUserIndex(engage);
            long[] engageAction = this.data.engageAction[index];

            if (this.config.removeTrain == true && this.split.isTrain(index) == true) {
                return;
            }
            if (this.config.removeValid == true && this.split.isValid(index) == true) {
                return;
            }

            synchronized (this.engageCache[userIndex]) {
                this.engageCache[userIndex][0]++;
            }
            synchronized (this.createCache[creatorIndex]) {
                this.createCache[creatorIndex][0]++;
            }

            if (engageAction == null) {
                return;
            }

            synchronized (this.engageCache[userIndex]) {
                this.engageCache[userIndex][1]++;
            }

            synchronized (this.createCache[creatorIndex]) {
                this.createCache[creatorIndex][1]++;
            }

            for (int i = 0; i < N_ENGAGE; i++) {
                if (engageAction[i] > 0) {
                    synchronized (this.userToCreate[creatorIndex]) {
                        Set<Integer> set =
                                this.userToCreate[creatorIndex][i];
                        if (set == null) {
                            set = new TreeSet<>();
                            this.userToCreate[creatorIndex][i] = set;
                        }
                        set.add(index);
                    }

                    synchronized (this.userToEngage[userIndex]) {
                        Set<Integer> set =
                                this.userToEngage[userIndex][i];
                        if (set == null) {
                            set = new TreeSet<>();
                            this.userToEngage[userIndex][i] = set;
                        }
                        set.add(index);
                    }
                }
            }
        });

        timer.toc("initCache done");
    }

    public List<MLSparseVector> getTweetFeatures(final TargetRecord record) {

        // NOTE: all validation and test tweets are cold start so
        // engagement-based feature won't work
        List<MLSparseVector> feats = new LinkedList();

        for (TweetFeature feature : TWEET_FEATURES_TO_USE) {
            feats.add(this.data.tweetFeatures.get(feature).getRowTransformed(record.targetTweetIndex, true));
        }

//        Instant instant =
//                Instant.ofEpochSecond(this.data
//                .tweetCreation[targetTweetIndex]);
//        ZonedDateTime zoned = ZonedDateTime.ofInstant(instant, ZoneOffset
//        .UTC);
//        feats.add(new MLDenseVector(new float[]{
//                zoned.getDayOfWeek().getValue(),
//                zoned.getHour()
//        }).toSparse());

        return feats;
    }

    public List<MLSparseVector> getCreatorFeatures(final TargetRecord record) {

        List<MLSparseVector> feats = new LinkedList();

        int creatorFollowers =
                RecSys20Helper.getCreatorFollowers(record.targetEngage);
        int creatorFollowing =
                RecSys20Helper.getCreatorFollowing(record.targetEngage);
        float creatorFollowerRatio = (creatorFollowing > 0) ?
                creatorFollowers / ((float) creatorFollowing) : 0;
        int[] createCache = this.createCache[record.targetCreatorIndex];
        int[] engageCache = this.engageCache[record.targetCreatorIndex];
        feats.add(new MLDenseVector(new float[]{
                creatorFollowers,
                creatorFollowing,
                creatorFollowerRatio,
                RecSys20Helper.getCreatorVerified(record.targetEngage),
                (float) ((MIN_DATE - this.data.userCreation[record.targetCreatorIndex]) / 86400.0),

                createCache[0],
                createCache[1],
                createCache[0] > 0 ?
                        createCache[1] / ((float) createCache[0]) : 0,

                engageCache[0],
                engageCache[1],
                engageCache[0] > 0 ?
                        engageCache[1] / ((float) engageCache[0]) : 0
        }).toSparse());

        //features for tweets engaged with by creator
        Set<Integer>[] targetCreatorEngage =
                this.userToEngage[record.targetCreatorIndex];
        long[] engageDates = new long[N_ENGAGE];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetCreatorEngage[i] == null) {
                continue;
            }
            for (int index : targetCreatorEngage[i]) {
                long[] engageAction = this.data.engageAction[index];
                if (engageAction[i] > engageDates[i]) {
                    engageDates[i] = engageAction[i];
                }
            }
        }
        float[] engageDatesF = new float[N_ENGAGE];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (engageDates[i] > 0) {
                engageDatesF[i] =
                        (float) (engageDates[i] - MIN_DATE);
            }
        }
        feats.add(new MLDenseVector(engageDatesF).toSparse());

        //features for tweets created by creator
        Set<Integer>[] targetCreatorCreate =
                this.userToCreate[record.targetCreatorIndex];
        float[] createCounts = new float[N_ENGAGE];
        float[] createCountsNorm = new float[N_ENGAGE];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetCreatorCreate[i] == null) {
                continue;
            }
            createCounts[i] = targetCreatorCreate[i].size();
            createCountsNorm[i] =
                    targetCreatorCreate[i].size() / ((float) createCache[0]);
        }
        feats.add(new MLDenseVector(createCounts).toSparse());
        feats.add(new MLDenseVector(createCountsNorm).toSparse());

        return feats;
    }

    public List<MLSparseVector> getUserFeatures(final TargetRecord record) {

        List<MLSparseVector> feats = new LinkedList();

        int userFollowers =
                RecSys20Helper.getUserFollowers(record.targetEngage);
        int userFollowing =
                RecSys20Helper.getUserFollowing(record.targetEngage);
        float userFollowerRatio = (userFollowing > 0) ?
                userFollowers / ((float) userFollowing) : 0;
        int creatorFollowers =
                RecSys20Helper.getCreatorFollowers(record.targetEngage);
        int[] createCache = this.createCache[record.targetUserIndex];
        int[] engageCache = this.engageCache[record.targetUserIndex];
        feats.add(new MLDenseVector(new float[]{
                userFollowers,
                userFollowing,
                userFollowerRatio,
                creatorFollowers - userFollowers,
                RecSys20Helper.getUserVerified(record.targetEngage),
                (float) ((MIN_DATE - this.data.userCreation[record.targetUserIndex]) / 86400.0),

                createCache[0],
                createCache[1],
                createCache[0] > 0 ?
                        createCache[1] / ((float) createCache[0]) : 0,

                engageCache[0],
                engageCache[1],
                engageCache[0] > 0 ?
                        engageCache[1] / ((float) engageCache[0]) : 0
        }).toSparse());

        //features for tweets created by target user
        Set<Integer>[] targetUserCreate =
                this.userToCreate[record.targetUserIndex];
        float[] createCounts = new float[N_ENGAGE];
        float[] createCountsNorm = new float[N_ENGAGE];
        long[] createDates = new long[N_ENGAGE];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetUserCreate[i] == null) {
                continue;
            }
            createCounts[i] = targetUserCreate[i].size();
            createCountsNorm[i] = createCounts[i] / ((float) createCache[0]);
            for (int index : targetUserCreate[i]) {
                long[] engageAction = this.data.engageAction[index];
                if (engageAction[i] > createDates[i]) {
                    createDates[i] = engageAction[i];
                }
            }
        }
        float[] createDatesF = new float[N_ENGAGE];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (createDates[i] > 0) {
                createDatesF[i] =
                        (float) ((createDates[i] - MIN_DATE) / 3600.0);
            }
        }
        feats.add(new MLDenseVector(createCounts).toSparse());
        feats.add(new MLDenseVector(createCountsNorm).toSparse());
        feats.add(new MLDenseVector(createDatesF).toSparse());

        return feats;
    }

    public List<MLSparseVector> getUserCreatorFeatures(final TargetRecord record) {
        List<MLSparseVector> feats = new LinkedList();

        feats.add(new MLDenseVector(new float[]{
                RecSys20Helper.getUserCreatorFollow(record.targetEngage)
        }).toSparse());

        Set<Integer>[] targetUserEngage =
                this.userToEngage[record.targetUserIndex];
        int[] targetUserEngageCache = this.engageCache[record.targetUserIndex];

        float[] engageCounts = new float[N_ENGAGE];
        float[] engageCountsNorm = new float[N_ENGAGE];
        float[] engageCountsCreator = new float[N_ENGAGE];
        long[] engageDates = new long[N_ENGAGE];
        Set<Integer> userEngageLanguages = new TreeSet<>();
        for (int i = 0; i < N_ENGAGE; i++) {
            float[] engageContent = new float[7];
            if (targetUserEngage[i] == null) {
                feats.add(new MLDenseVector(engageContent).toSparse());
                continue;
            }
            //total engagement counts
            engageCounts[i] = targetUserEngage[i].size();
            engageCountsNorm[i] =
                    engageCounts[i] / ((float) targetUserEngageCache[0]);
            for (int engageIndex : targetUserEngage[i]) {
                int[] engage = this.data.engage[engageIndex];
                long[] action = this.data.engageAction[engageIndex];

                if (record.targetCreatorIndex == RecSys20Helper.getCreatorIndex(engage)) {
                    //engagement count with tweet creator
                    engageCountsCreator[i]++;
                }

                long engageDate = action[i];
                if (engageDates[i] < engageDate) {
                    //date of last engagement
                    engageDates[i] = engageDate;
                }

                int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                MLSparseVector tweetTags =
                        this.data.tweetFeatures.get(TweetFeature.hashtags).getRow(tweetIndex, false);
                MLSparseVector tweetMedia =
                        this.data.tweetFeatures.get(TweetFeature.present_media).getRow(tweetIndex, false);
                MLSparseVector tweetLinks =
                        this.data.tweetFeatures.get(TweetFeature.present_links).getRow(tweetIndex, false);
                MLSparseVector tweetDomains =
                        this.data.tweetFeatures.get(TweetFeature.present_domains).getRow(tweetIndex, false);
                MLSparseVector tweetTextTokens =
                        this.data.tweetFeatures.get(TweetFeature.text_tokens).getRow(tweetIndex, false);

                engageContent[0] += (tweetTags == null) ? 0 :
                        tweetTags.getIndexes().length;
                engageContent[1] += (tweetMedia == null) ? 0 :
                        tweetMedia.getIndexes().length;
                engageContent[2] += (tweetLinks == null) ? 0 :
                        tweetLinks.getIndexes().length;
                engageContent[3] += (tweetDomains == null) ? 0 :
                        tweetDomains.getIndexes().length;
                engageContent[4] += (tweetTextTokens == null) ? 0 :
                        tweetTextTokens.getIndexes().length;
                if (RecSys20Helper.getCreatorVerified(engage) > 0) {
                    engageContent[5]++;
                }
                if (RecSys20Helper.getUserCreatorFollow(engage) > 0) {
                    engageContent[6]++;
                }

                MLSparseVector tweetLanguage =
                        this.data.tweetFeatures.get(TweetFeature.language).getRow(tweetIndex, false);
                int languageIndex = (tweetLanguage != null) ?
                        tweetLanguage.getIndexes()[0] : -1;
                if (languageIndex >= 0) {
                    userEngageLanguages.add(languageIndex);
                }
            }

            if (engageCounts[i] > 1) {
                for (int j = 0; j < engageContent.length; j++) {
                    engageContent[j] = engageContent[j] / engageCounts[i];
                }
            }
            feats.add(new MLDenseVector(engageContent).toSparse());
        }
        float[] engageDatesF = new float[engageDates.length];
        for (int i = 0; i < engageDates.length; i++) {
            if (engageDates[i] > 0) {
                //this is necessary to avoid overflow
                engageDatesF[i] =
                        (float) ((engageDates[i] - MIN_DATE) / 3600.0);
            }
        }
        feats.add(new MLDenseVector(engageCounts).toSparse());
        feats.add(new MLDenseVector(engageCountsCreator).toSparse());
        feats.add(new MLDenseVector(engageDatesF).toSparse());
        feats.add(new MLDenseVector(engageCountsNorm).toSparse());
        feats.add(new MLDenseVector(new float[]{
                userEngageLanguages.size(),
        }).toSparse());

        return feats;
    }

    public double[] getTweetSimilarity(final int tweetIndex1,
                                       final int creatorIndex1,
                                       final int tweetIndex2,
                                       final int creatorIndex2) {
        return new double[]{
                //same language
                RecSys20Helper.indexSame(
                        this.data.tweetFeatures.get(TweetFeature.language).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.language).getRow(tweetIndex2, false)),

                //same type
                RecSys20Helper.indexSame(
                        this.data.tweetFeatures.get(TweetFeature.tweet_type).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.tweet_type).getRow(tweetIndex2, false)),

                //text token intersect
                RecSys20Helper.indexIntersect(
                        this.data.tweetFeatures.get(TweetFeature.text_tokens).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.text_tokens).getRow(tweetIndex2, false),
                        false),

                //media intersect
                RecSys20Helper.indexIntersect(
                        this.data.tweetFeatures.get(TweetFeature.present_media).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.present_media).getRow(tweetIndex2, false),
                        false),

                //domains intersect
                RecSys20Helper.indexIntersect(
                        this.data.tweetFeatures.get(TweetFeature.present_domains).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.present_domains).getRow(tweetIndex2, false),
                        false),

                //links intersect
                RecSys20Helper.indexIntersect(
                        this.data.tweetFeatures.get(TweetFeature.present_links).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.present_links).getRow(tweetIndex2, false),
                        false),

                //tags intersect
                RecSys20Helper.indexIntersect(
                        this.data.tweetFeatures.get(TweetFeature.hashtags).getRow(tweetIndex1, false),
                        this.data.tweetFeatures.get(TweetFeature.hashtags).getRow(tweetIndex2, false),
                        false),

                //creator similarity, improves around 0.001 but slow...
                RecSys20Helper.multiply(
                        this.cf.userCreatorT.getRow(creatorIndex1, false),
                        this.cf.userCreatorT.getRow(creatorIndex2, false)),

                RecSys20Helper.multiply(
                        this.cf.userCreator.getRow(creatorIndex1, false),
                        this.cf.userCreator.getRow(creatorIndex2, false))

                //tweet embedding similarity
//                this.embed.tweetTweetScore(tweetIndex1, tweetIndex2, false)
        };
    }

    public List<MLSparseVector> getUserTweetFeatures(final TargetRecord record) {
        List<MLSparseVector> feats = new LinkedList();

        Set<Integer>[] targetUserEngage =
                this.userToEngage[record.targetUserIndex];
        double[][] userTweetSimilarity = new double[N_ENGAGE][9];
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetUserEngage[i] == null) {
                continue;
            }
            for (int engageIndex : targetUserEngage[i]) {
                int[] engage = this.data.engage[engageIndex];
                double[] similarity = this.getTweetSimilarity(
                        record.targetTweetIndex,
                        record.targetCreatorIndex,
                        RecSys20Helper.getTweetIndex(engage),
                        RecSys20Helper.getCreatorIndex(engage));

                for (int j = 0; j < similarity.length; j++) {
                    userTweetSimilarity[i][j] += similarity[j];
                }
            }
        }

        for (int i = 0; i < N_ENGAGE; i++) {
            float[] sim = new float[userTweetSimilarity[i].length];
            if (targetUserEngage[i] == null) {
                feats.add(new MLDenseVector(sim).toSparse());
                continue;
            }

            int norm = targetUserEngage[i].size();
            for (int j = 0; j < userTweetSimilarity[i].length; j++) {
                if (norm > 1) {
                    sim[j] = (float) (userTweetSimilarity[i][j] / norm);
                } else {
                    sim[j] = (float) userTweetSimilarity[i][j];
                }
            }
            feats.add(new MLDenseVector(sim).toSparse());
        }

        return feats;
    }

    public List<MLSparseVector> getLBDataFeatures(final TargetRecord record) {
        List<MLSparseVector> feats = new LinkedList();

        feats.add(new MLDenseVector(new float[]{
                this.engageLBCache[record.targetUserIndex][0],
                this.createLBCache[record.targetUserIndex][0],

                this.engageLBCache[record.targetCreatorIndex][0],
                this.createLBCache[record.targetCreatorIndex][0]
        }).toSparse());

        return feats;
    }

    public List<MLSparseVector> getRawTextCountFeatures(final TargetRecord record) {
        List<MLSparseVector> feats = new LinkedList();

        //numToks,numFirstUpper,numAllCaps,
        //numPunctuation, maxTokLength,avgTokLength
        int NUM_TEXT_COUNTS = 6;

        //Target Tweet Features
        float[] tokenCounts = new float[NUM_TEXT_COUNTS];
        if (this.textData.tweetToTokCounts[record.targetTweetIndex] != null) {
            tokenCounts =
                    this.textData.tweetToTokCounts[record.targetTweetIndex];
        }
        feats.add(new MLDenseVector(tokenCounts).toSparse());

        //Features for tweets created by creator
        Set<Integer>[] targetCreatorCreate =
                this.userToCreate[record.targetCreatorIndex];
        float[] targetCreatorCreateTextCounts = new float[NUM_TEXT_COUNTS];
        int nTotal = 0;
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetCreatorCreate[i] == null) {
                continue;
            }
            nTotal += targetCreatorCreate[i].size();
            for (int index : targetCreatorCreate[i]) {
                int[] engage = this.data.engage[index];
                int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                float[] tweetTokCounts =
                        this.textData.tweetToTokCounts[tweetIndex];
                for (int j = 0; j < tweetTokCounts.length; j++) {
                    targetCreatorCreateTextCounts[j] += tweetTokCounts[j];
                }
            }
        }
        if (nTotal > 1) {
            for (int i = 0; i < targetCreatorCreateTextCounts.length; i++) {
                targetCreatorCreateTextCounts[i] =
                        targetCreatorCreateTextCounts[i] / nTotal;
            }
        }
        feats.add(new MLDenseVector(targetCreatorCreateTextCounts).toSparse());

        //Features for tweets created by target user
        Set<Integer>[] targetUserCreate =
                this.userToCreate[record.targetUserIndex];
        float[] targetUserCreateTextCounts = new float[NUM_TEXT_COUNTS];
        nTotal = 0;
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetUserCreate[i] == null) {
                continue;
            }
            nTotal += targetUserCreate[i].size();
            for (int index : targetUserCreate[i]) {
                int[] engage = this.data.engage[index];
                int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                float[] tweetTokCounts =
                        this.textData.tweetToTokCounts[tweetIndex];
                for (int j = 0; j < tweetTokCounts.length; j++) {
                    targetUserCreateTextCounts[j] += tweetTokCounts[j];
                }
            }
        }
        if (nTotal > 1) {
            for (int i = 0; i < targetUserCreateTextCounts.length; i++) {
                targetUserCreateTextCounts[i] =
                        targetUserCreateTextCounts[i] / nTotal;
            }
        }
        feats.add(new MLDenseVector(targetUserCreateTextCounts).toSparse());

        // Features for Tweets Engaged with Target User
        Set<Integer>[] targetUserEngage =
                this.userToEngage[record.targetUserIndex];
        float[] targetUserEngageTextCounts = new float[NUM_TEXT_COUNTS];
        nTotal = 0;
        for (int i = 0; i < N_ENGAGE; i++) {
            if (targetUserEngage[i] == null) {
                continue;
            }
            nTotal += targetUserEngage[i].size();
            for (int index : targetUserEngage[i]) {
                int[] engage = this.data.engage[index];
                int tweetIndex = RecSys20Helper.getTweetIndex(engage);
                float[] tweetTokCounts =
                        this.textData.tweetToTokCounts[tweetIndex];
                for (int j = 0; j < tweetTokCounts.length; j++) {
                    targetUserEngageTextCounts[j] += tweetTokCounts[j];
                }
            }
        }
        if (nTotal > 1) {
            for (int i = 0; i < targetUserEngageTextCounts.length; i++) {
                targetUserEngageTextCounts[i] =
                        targetUserEngageTextCounts[i] / nTotal;
            }
        }
        feats.add(new MLDenseVector(targetUserEngageTextCounts).toSparse());


        return feats;
    }

    public List<MLSparseVector> extractFeatures(final int targetEngageIndex) {

        TargetRecord record = new TargetRecord();
        record.targetEngage = this.data.engage[targetEngageIndex];
        record.targetTweetIndex =
                RecSys20Helper.getTweetIndex(record.targetEngage);
        record.targetUserIndex =
                RecSys20Helper.getUserIndex(record.targetEngage);
        record.targetCreatorIndex =
                RecSys20Helper.getCreatorIndex(record.targetEngage);

        List<MLSparseVector> feats = new LinkedList<>();

        //tweet features
        feats.addAll(this.getTweetFeatures(record));

        //creator features
        feats.addAll(this.getCreatorFeatures(record));

        //user features
        feats.addAll(this.getUserFeatures(record));

        //user-creator features
        feats.addAll(this.getUserCreatorFeatures(record));

        //user-tweet features
        feats.addAll(this.getUserTweetFeatures(record));

        //collaborative filtering features
        feats.add(this.cf.getItemItem(record));

//        //LB data features
//        feats.addAll(this.getLBDataFeatures(record));
//
//        //raw text features
//        feats.addAll(this.getRawTextCountFeatures(record));

        return feats;
    }

}

//1M col selection
//RecSys20DataParser: present_media 2 elapsed [11 min 49 sec]
//RecSys20DataParser: present_links 0 elapsed [11 min 54 sec]
//RecSys20DataParser: language 11 elapsed [12 min 12 sec]
//RecSys20DataParser: present_domains 0 elapsed [12 min 17 sec]
//RecSys20DataParser: hashtags 0 elapsed [12 min 28 sec]
//RecSys20DataParser: tweet_type 3 elapsed [12 min 39 sec]
//RecSys20DataParser: text_tokens 297 elapsed [23 min 25 sec]

//-user engagement counts by day of week
//-counts in last 12h, 24h etc.
//-counts by following vs not following