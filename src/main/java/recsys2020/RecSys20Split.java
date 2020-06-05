package recsys2020;

import common.core.utils.MLRandomUtils;
import common.core.utils.MLTimer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import org.apache.hadoop.util.hash.Hash;
import recsys2020.RecSys20Model.RecSys20Config;

public class RecSys20Split {

    //default valid date
    public static final long MAX_DATE = 1581551999;
    public static final long VALID_DATE_CUTOFF_4H = MAX_DATE - 4 * 60 * 60;
    public static MLTimer timer;

    public static int CURRENT_NEG_INDEX = -1;
    public static int[] NEG_INDEXES;
    public static int CUR_SEED = 0;
    public static Set<Integer> VALID_NEG;

    static {
        timer = new MLTimer("RecSys20Split");
        timer.tic();
    }

    public Set<Integer> trainIndexSet;
    public Set<Integer> validIndexSet;
    public RecSys20Config config;
    public long trainDate;
    public long validDate;

    public RecSys20Split(final RecSys20Config configP,
                         final long trainDateP,
                         final long validDateP) {
        this.config = configP;
        this.trainDate = trainDateP;
        this.validDate = validDateP;
    }

    public void split(final RecSys20Data data) {
        this.trainIndexSet = new HashSet();
        this.validIndexSet = new HashSet();
        AtomicInteger posCount = new AtomicInteger(0);
        AtomicInteger negCount = new AtomicInteger(0);
        final int[] negIndexes = new int[data.lbEngageIndex];
        IntStream.range(0, data.lbEngageIndex).parallel().forEach(index -> {
            long[] engageAction = data.engageAction[index];
            if (engageAction == null) {
                negIndexes[negCount.getAndIncrement()] = index;
                return;
            }
            posCount.incrementAndGet();

            boolean isTrain = false;
            for (int i = 0; i < engageAction.length; i++) {
//               if (engageAction[i] > VALID_DATE_CUTOFF) {
                if (engageAction[i] > this.validDate) {
                    synchronized (this.validIndexSet) {
                        this.validIndexSet.add(index);
                    }
                    return;
                }
//              if ((engageAction[i] > this.trainDate)
//              && (engageAction[i] <= this.validDate) ){
                if (engageAction[i] > this.trainDate) {
                    isTrain = true;
                }
            }
            if (isTrain == true) {
                synchronized (this.trainIndexSet) {
                    this.trainIndexSet.add(index);
                }
            }
        });

        int[] negIndexesShuffled = Arrays.copyOfRange(negIndexes, 0,
                negCount.get());
        Arrays.sort(negIndexesShuffled);
        MLRandomUtils.shuffle(negIndexesShuffled, new Random(0));

        timer.toc("total " + data.lbEngageIndex);
        timer.toc("totalPos " + posCount.get());
        timer.toc("totalNeg " + negCount.get());

        //get negatives for training and validation, try to preserve
        //training data ratio for logL
//        double ratio = ((double) negCount.get()) / posCount.get();
//        int nNegTrain = (int) (ratio * this.trainIndexSet.size());
//        int nNegValid = (int) (ratio * this.validIndexSet.size());

        //50/50 ratio seems to work slightly better...
        int nNegTrain = this.trainIndexSet.size();
        int nNegValid = this.validIndexSet.size();

        timer.toc("trainPos " + trainIndexSet.size());
        timer.toc("trainNeg " + nNegTrain);
        timer.toc("validPos " + validIndexSet.size());
        timer.toc("validNeg " + nNegValid);

        for (int i = 0; i < nNegValid + nNegTrain; i++) {
            if (i < nNegValid) {
                this.validIndexSet.add(negIndexesShuffled[i]);
            } else {
                this.trainIndexSet.add(negIndexesShuffled[i]);
            }
        }

        timer.toc("split done");
    }

    public void splitByCreateDate(final RecSys20Data data) {
        this.trainIndexSet = new HashSet();
        Set<Integer> trainNegSet = new HashSet();
        this.validIndexSet = new HashSet();
        Set<Integer> validNegSet = new HashSet();
        IntStream.range(0, data.lbEngageIndex).parallel().forEach(index -> {
            long[] engageAction = data.engageAction[index];
            if (engageAction == null) {
                int tweetIndex =
                        RecSys20Helper.getTweetIndex(data.engage[index]);
                long creationDate = data.tweetCreation[tweetIndex];
                if (creationDate > VALID_DATE_CUTOFF_4H) {
                    validNegSet.add(index);
                    return;
                }
                if ((creationDate > this.trainDate) && (creationDate < this.validDate)) {
                    trainNegSet.add(index);
                    return;
                }
                return;
            }

            boolean isTrain = false;
            for (int i = 0; i < engageAction.length; i++) {
                if (engageAction[i] > VALID_DATE_CUTOFF_4H) {
                    synchronized (this.validIndexSet) {
                        this.validIndexSet.add(index);
                    }
                    return;
                }

                if ((engageAction[i] > this.trainDate)
                        && (engageAction[i] <= this.validDate)) {
                    isTrain = true;
                }
            }
            if (isTrain == true) {
                synchronized (this.trainIndexSet) {
                    this.trainIndexSet.add(index);
                }
            }
        });

        timer.toc("trainPos " + trainIndexSet.size());
        timer.toc("trainNeg " + trainNegSet.size());
        timer.toc("validPos " + validIndexSet.size());
        timer.toc("validNeg " + validNegSet.size());

        trainIndexSet.addAll(trainNegSet);
        validIndexSet.addAll(validNegSet);
    }

    public void splitWithCache(final RecSys20Data data) {
        this.trainIndexSet = new HashSet();
        this.validIndexSet = new HashSet();
        AtomicInteger posCount = new AtomicInteger(0);
        AtomicInteger negCount = new AtomicInteger(0);
        final int[] negIndexes = new int[data.lbEngageIndex];
        IntStream.range(0, data.lbEngageIndex).parallel().forEach(index -> {
            long[] engageAction = data.engageAction[index];
            if (engageAction == null) {
                negIndexes[negCount.getAndIncrement()] = index;
                return;
            }
            posCount.incrementAndGet();

            boolean isTrain = false;
            for (int i = 0; i < engageAction.length; i++) {
                if (engageAction[i] > VALID_DATE_CUTOFF_4H) {
                    synchronized (this.validIndexSet) {
                        this.validIndexSet.add(index);
                    }
                    return;
                }

                if ((engageAction[i] > this.trainDate)
                        && (engageAction[i] <= this.validDate)) {
                    isTrain = true;
                }
            }
            if (isTrain == true) {
                synchronized (this.trainIndexSet) {
                    this.trainIndexSet.add(index);
                }
            }
        });

        int[] negIndexesShuffled = Arrays.copyOfRange(negIndexes, 0,
                negCount.get());
        Arrays.sort(negIndexesShuffled);
        MLRandomUtils.shuffle(negIndexesShuffled, new Random(0));

        timer.toc("total " + data.lbEngageIndex);
        timer.toc("totalPos " + posCount.get());
        timer.toc("totalNeg " + negCount.get());

        //50/50 ratio seems to work slightly better...
//        int nNegTrain = this.trainIndexSet.size();
        int nNegValid = this.validIndexSet.size();

        //to use all 0's
        int nNegTrain =
                (int) Math.floor((negIndexesShuffled.length - nNegValid) /
                        ((double) this.config.nChunks));

        timer.toc("trainPos " + trainIndexSet.size());
        timer.toc("trainNeg " + nNegTrain);
        timer.toc("validPos " + validIndexSet.size());
        timer.toc("validNeg " + nNegValid);

        for (int i = 0; i < nNegValid; i++) {
            //use same 0's for valid
            this.validIndexSet.add(negIndexesShuffled[i]);
        }
        if (CURRENT_NEG_INDEX < 0) {
            //first chunk
            CURRENT_NEG_INDEX = nNegValid;
        }

        for (int i = 0; i < nNegTrain; i++) {
            //take different 0's for each chunk
            this.trainIndexSet.add(negIndexesShuffled[CURRENT_NEG_INDEX]);
            CURRENT_NEG_INDEX++;
        }

        timer.toc("split done CURRENT_NEG_INDEX=" + CURRENT_NEG_INDEX);
    }

    public boolean isValid(final int index) {
        return this.validIndexSet.contains(index);
    }

    public boolean isTrain(final int index) {
        return this.trainIndexSet.contains(index);
    }

    public boolean isTrainOrValid(final int index) {
        return this.trainIndexSet.contains(index)
                || this.validIndexSet.contains(index);
    }

    public long sampleTrainDate(int index) {

        //sample date randomly in [trainDate, validDate)
        SplittableRandom random = new SplittableRandom(index);
        long length = this.validDate - this.trainDate;
        return this.trainDate + (long) (random.nextDouble() * length);
    }

}