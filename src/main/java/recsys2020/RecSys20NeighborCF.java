package recsys2020;

import common.core.linalg.FloatElement;
import common.core.linalg.MLSparseMatrix;
import common.core.linalg.MLSparseMatrixAOO;
import common.core.linalg.MLSparseVector;
import common.core.utils.MLIOUtils;
import common.core.utils.MLTimer;
import recsys2020.RecSys20Data.EngageType;
import recsys2020.RecSys20Model.RecSys20Config;
import recsys2020.RecSys20FeatExtractor.TargetRecord;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys20NeighborCF {

    public static int N_ENGAGE = EngageType.values().length;
    public static MLTimer timer;

    static {
        timer = new MLTimer("RecSys20NeighborCF");
        timer.tic();
    }

    //data
    public RecSys20Split split;
    public RecSys20Data data;
    public RecSys20Config config;

    public MLSparseMatrix userCreator;
    public MLSparseMatrix userCreatorT;

    public RecSys20NeighborCF(final RecSys20Data dataP,
                              final RecSys20Split splitP,
                              final RecSys20Config configP) {
        this.data = dataP;
        this.split = splitP;
        this.config = configP;
    }

    public Set<Integer>[][] initCache() {
        Set<Integer>[][] userToEngage =
                new Set[this.data.userToIndex.size()][N_ENGAGE + 1];

        IntStream.range(0, this.data.lbEngageIndex).parallel().forEach(index -> {
            if (this.config.removeTrain == true && this.split.isTrain(index) == true) {
                return;
            }
            if (this.config.removeValid == true && this.split.isValid(index) == true) {
                return;
            }

            int[] engage = this.data.engage[index];
            long[] engageAction = this.data.engageAction[index];
            int userIndex = RecSys20Helper.getUserIndex(engage);
            if (engageAction == null) {
                synchronized (userToEngage[userIndex]) {
                    Set<Integer> set = userToEngage[userIndex][N_ENGAGE];
                    if (set == null) {
                        set = new HashSet();
                        userToEngage[userIndex][N_ENGAGE] = set;
                    }
                    set.add(index);
                }
                return;
            }

            for (int i = 0; i < engageAction.length; i++) {
                if (engageAction[i] > 0) {
                    synchronized (userToEngage[userIndex]) {
                        Set<Integer> set = userToEngage[userIndex][i];
                        if (set == null) {
                            set = new HashSet();
                            userToEngage[userIndex][i] = set;
                        }
                        set.add(index);
                    }
                }
            }
        });
        timer.toc("initCache done");

        return userToEngage;
    }

    public void initR(final Set<Integer>[][] userToEngage) {

        final int N_USERS = this.data.userToIndex.size();
        final int N_TWEETS = this.data.tweetToIndex.size();

        MLSparseVector[] userCreatorRows = new MLSparseVector[N_USERS];

        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(0, N_USERS).parallel().forEach(userIndex -> {
            int curCount = counter.incrementAndGet();
            if (curCount % 10_000_000 == 0) {
                timer.tocLoop("initR", curCount);
            }
            Set<Integer>[] userEngages = userToEngage[userIndex];
            if (userEngages == null) {
                return;
            }

            Map<Integer, FloatElement> elementMap = new TreeMap<>();
            for (int i = 0; i < N_ENGAGE; i++) {
                Set<Integer> userEngage = userEngages[i];
                if (userEngage == null) {
                    continue;
                }
                for (int engageIndex : userEngages[i]) {
                    int[] engage = this.data.engage[engageIndex];
                    int creatorIndex = RecSys20Helper.getCreatorIndex(engage);

                    FloatElement element = elementMap.get(creatorIndex);
                    if (element == null) {
                        element = new FloatElement(creatorIndex, 1);
                        elementMap.put(creatorIndex, element);
                    } else {
                        element.setValue(element.getValue() + 1);
                    }
                }
            }

            int[] indexes = new int[elementMap.size()];
            float[] values = new float[elementMap.size()];
            int cur = 0;
            for (Map.Entry<Integer, FloatElement> entry :
                    elementMap.entrySet()) {
                indexes[cur] = entry.getValue().getIndex();
                values[cur] = entry.getValue().getValue();
                cur++;
            }
            userCreatorRows[userIndex] = new MLSparseVector(indexes, values,
                    null, N_USERS);
        });
        this.userCreator = new MLSparseMatrixAOO(userCreatorRows, N_USERS);
        this.userCreatorT = this.userCreator.transpose();
        this.userCreatorT.applyColNorm(this.userCreatorT.getColNorm(2));
        this.userCreatorT.applyRowNorm(this.userCreatorT.getRowNorm(2));
        timer.toc("userCreator" +
                " nRows:" + this.userCreator.getNRows() +
                " nCols:" + this.userCreator.getNCols() +
                " nnz:" + this.userCreator.getNNZ());
    }

    public MLSparseVector getItemItem(final TargetRecord record) {
        MLSparseVector userVector =
                this.userCreator.getRow(record.targetUserIndex,
                        false);
        MLSparseVector creatorVector =
                this.userCreatorT.getRow(record.targetCreatorIndex, false);
        if (userVector == null || creatorVector == null) {
            return new MLSparseVector(null, null, null, 1);
        }

        //item-item similarity
        int[] creatorIndexes = creatorVector.getIndexes();
        float[] creatorValues = creatorVector.getValues();
        double score = 0;
        for (int userCreatorIndex : userVector.getIndexes()) {
            MLSparseVector userCreatorVector =
                    this.userCreatorT.getRow(userCreatorIndex,
                            false);
            if (userCreatorVector == null) {
                continue;
            }

            int[] indexes = userCreatorVector.getIndexes();
            float[] values = userCreatorVector.getValues();
            for (int i = 0; i < indexes.length; i++) {
                int index = Arrays.binarySearch(creatorIndexes, indexes[i]);
                if (index >= 0) {
                    score += ((double) values[i]) * ((double) creatorValues[index]);
                }
            }
        }
        if (score == 0) {
            return new MLSparseVector(null, null, null, 1);
        } else {
            return new MLSparseVector(
                    new int[]{0},
                    new float[]{(float) score},
                    null,
                    1);
        }
    }


    public MLSparseVector getUserUser(final int targetUserIndex,
                                      final int targetCreatorIndex) {
        MLSparseVector userVector = this.userCreator.getRow(targetUserIndex,
                false);
        MLSparseVector creatorVector =
                this.userCreatorT.getRow(targetCreatorIndex, false);
        if (userVector == null || creatorVector == null) {
            return new MLSparseVector(null, null, null, 1);
        }

        //user-user similarity
        int[] userIndexes = userVector.getIndexes();
        float[] userValues = userVector.getValues();

        float score = 0;
        for (int creatorUserIndex : creatorVector.getIndexes()) {
            MLSparseVector creatorUserVector =
                    this.userCreator.getRow(creatorUserIndex,
                            false);
            if (creatorUserVector == null) {
                continue;
            }

            int[] indexes = creatorUserVector.getIndexes();
            float[] values = creatorUserVector.getValues();
            for (int i = 0; i < indexes.length; i++) {
                int index = Arrays.binarySearch(userIndexes, indexes[i]);
                if (index >= 0) {
                    score += values[i] * userValues[index];
                }
            }
        }

        if (score == 0) {
            return new MLSparseVector(null, null, null, 1);
        } else {
            return new MLSparseVector(new int[]{0}, new float[]{score}, null,
                    1);
        }
    }

    public void validateItemItem() {
        Integer[] validIndexes = new Integer[this.split.validIndexSet.size()];
        this.split.validIndexSet.toArray(validIndexes);
        Arrays.sort(validIndexes);

        FloatElement[] preds =
                new FloatElement[validIndexes.length];
        AtomicInteger counter = new AtomicInteger(0);
        IntStream.range(0, validIndexes.length).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 1_000_000 == 0) {
                timer.tocLoop("validateItemItem", count);
            }
            int validIndex = validIndexes[index];
            int[] engage = this.data.engage[validIndex];

            int userIndex = RecSys20Helper.getUserIndex(engage);
            int creatorIndex = RecSys20Helper.getCreatorIndex(engage);

            MLSparseVector userVector = this.userCreator.getRow(userIndex,
                    false);
            MLSparseVector creatorVector =
                    this.userCreatorT.getRow(creatorIndex,
                            false);
            if (userVector == null || creatorVector == null) {
                preds[index] = new FloatElement(validIndex, 0);
                return;
            }

            Map<Integer, Float> map = new HashMap<>();
            for (int i = 0; i < creatorVector.getIndexes().length; i++) {
                map.put(creatorVector.getIndexes()[i],
                        creatorVector.getValues()[i]);
            }
            float score = 0;
            for (int userCreatorIndex : userVector.getIndexes()) {
                MLSparseVector userCreatorVector =
                        this.userCreatorT.getRow(userCreatorIndex,
                                false);
                if (userCreatorVector == null) {
                    continue;
                }

                int[] indexes = userCreatorVector.getIndexes();
                float[] values = userCreatorVector.getValues();
                for (int i = 0; i < indexes.length; i++) {
                    Float value = map.get(indexes[i]);
                    if (value != null) {
                        score += values[i] * value;
                    }
                }
            }
            preds[index] = new FloatElement(validIndex, score);
        });

        //evaluate
        RecSys20Eval eval = new RecSys20Eval();
        for (EngageType engage : EngageType.values()) {
            eval.evaluate(engage, preds, this.data);
        }
    }

}
