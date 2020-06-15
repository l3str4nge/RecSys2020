package recsys2020;

import common.core.utils.MLRandomUtils;
import common.core.utils.MLTimer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import recsys2020.RecSys20Model.RecSys20Config;

public class RecSys20Split {

    //default valid date
    public static final long MAX_DATE = 1581551999;
    public static final long VALID_DATE_CUTOFF_4H = MAX_DATE - 4 * 60 * 60;
    public static MLTimer timer;

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
                //NEGATIVES
                int tweetIndex =
                        RecSys20Helper.getTweetIndex(data.engage[index]);
                long creationDate = data.tweetCreation[tweetIndex];
                if (creationDate > VALID_DATE_CUTOFF_4H) {
                    synchronized (validNegSet) {
                        validNegSet.add(index);
                    }
                    return;
                }
                if ((creationDate > this.trainDate) && (creationDate < this.validDate)) {
                    synchronized (trainNegSet) {
                        trainNegSet.add(index);
                    }
                    return;
                }
                return;
            }

            //POSITIVES
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
        this.trainIndexSet.addAll(trainNegSet);
        this.validIndexSet.addAll(validNegSet);
    }

    public void splitByCreateDateBlend(final RecSys20Data data) {
        this.trainIndexSet = new HashSet();
        Set<Integer> trainNegSet = new HashSet();
        this.validIndexSet = new HashSet();
        Set<Integer> validNegSet = new HashSet();
        IntStream.range(0, data.lbEngageIndex).parallel().forEach(index -> {
            long[] engageAction = data.engageAction[index];
            if (engageAction == null) {
                //NEGATIVES
                int tweetIndex =
                        RecSys20Helper.getTweetIndex(data.engage[index]);
                long creationDate = data.tweetCreation[tweetIndex];
                if (creationDate > this.validDate) {
                    synchronized (validNegSet) {
                        validNegSet.add(index);
                    }
                    return;
                }
                if (creationDate > this.trainDate) {
                    synchronized (trainNegSet) {
                        trainNegSet.add(index);
                    }
                    return;
                }
                return;
            }

            //POSITIVES
            long minDate = -1;
            for (int i = 0; i < engageAction.length; i++) {
                if (engageAction[i] > 0) {
                    if (minDate < 0) {
                        minDate = engageAction[i];
                    } else {
                        minDate = Math.min(minDate, engageAction[i]);
                    }
                }
            }
            if (minDate > this.validDate) {
                synchronized (this.validIndexSet) {
                    this.validIndexSet.add(index);
                }

            } else if (minDate > this.trainDate) {
                synchronized (this.trainIndexSet) {
                    this.trainIndexSet.add(index);
                }
            }
        });
        timer.toc("trainPos " + trainIndexSet.size());
        timer.toc("trainNeg " + trainNegSet.size());
        timer.toc("validPos " + validIndexSet.size());
        timer.toc("validNeg " + validNegSet.size());
        this.trainIndexSet.addAll(trainNegSet);
        this.validIndexSet.addAll(validNegSet);
    }

    public boolean isValid(final int index) {
        return this.validIndexSet.contains(index);
    }

    public boolean isTrain(final int index) {
        return this.trainIndexSet.contains(index);
    }

}