package recsys2020;

import common.core.linalg.MLSparseMatrix;
import common.core.linalg.MLSparseVector;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public class RecSys20Helper {

    public static int getTweetIndex(final int[] engage) {
        return engage[0];
    }

    public static int getCreatorIndex(final int[] engage) {
        return engage[1];
    }

    public static int getCreatorFollowers(final int[] engage) {
        return engage[2];
    }

    public static int getCreatorFollowing(final int[] engage) {
        return engage[3];
    }

    public static int getCreatorVerified(final int[] engage) {
        return engage[4];
    }

    public static int getUserIndex(final int[] engage) {
        return engage[5];
    }

    public static int getUserFollowers(final int[] engage) {
        return engage[6];
    }

    public static int getUserFollowing(final int[] engage) {
        return engage[7];
    }

    public static int getUserVerified(final int[] engage) {
        return engage[8];
    }

    public static int getUserCreatorFollow(final int[] engage) {
        return engage[9];
    }

    public static double indexSame(final MLSparseVector row1,
                                   final MLSparseVector row2) {

        if (row1 == null || row2 == null) {
            return 0f;
        }

        int[] indexes1 = row1.getIndexes();
        int[] indexes2 = row2.getIndexes();

        if (indexes1.length > 1 || indexes2.length > 1) {
            throw new IllegalStateException("index.length > 1");
        }

        if (indexes1[0] == indexes2[0]) {
            return 1;
        }

        return 0;
    }

    public static double indexIntersect(final MLSparseVector row1,
                                        final MLSparseVector row2,
                                        final boolean normalize) {

        if (row1 == null || row2 == null) {
            return 0;
        }

        int[] indexes1 = row1.getIndexes();
        int[] indexes2 = row2.getIndexes();

        double intersect = 0;
        for (int index : indexes1) {
            if (Arrays.binarySearch(indexes2, index) >= 0) {
                intersect++;
            }
        }
        if (normalize == true) {
            intersect = intersect / (indexes1.length * indexes2.length);
        }

        return intersect;
    }

    public static double indexIntersect(final MLSparseVector row1,
                                        final MLSparseVector row2,
                                        final int[] skip) {

        if (row1 == null || row2 == null) {
            return 0f;
        }

        int[] indexes1 = row1.getIndexes();
        int[] indexes2 = row2.getIndexes();

        double intersect = 0;
        for (int index : indexes1) {
            if (Arrays.binarySearch(skip, index) >= 0) {
                continue;
            }

            if (Arrays.binarySearch(indexes2, index) >= 0) {
                intersect++;
            }
        }
        return intersect;
    }

    public static double multiply(final MLSparseVector row1,
                                  final MLSparseVector row2) {
        if (row1 == null || row2 == null) {
            return 0;
        }

        int[] indexes1 = row1.getIndexes();
        float[] values1 = row1.getValues();

        int[] indexes2 = row2.getIndexes();
        float[] values2 = row2.getValues();

        double product = 0;
        for (int i = 0; i < indexes1.length; i++) {
            int index2 = Arrays.binarySearch(indexes2, indexes1[i]);
            if (index2 < 0) {
                continue;
            }
            product += ((double) values1[i]) * ((double) values2[index2]);

        }
        return product;
    }


    public static boolean isBefore(final long date,
                                   final long[] engageAction) {
        if (date < 0) {
            return true;
        }

        for (int i = 0; i < engageAction.length; i++) {
            if (engageAction[i] >= date) {
                return false;
            }
        }
        return true;
    }

    public static long[][] getTrainValidDates(final long interval,
                                              final int nIntervals,
                                              final long validCutOff) {
        long[][] dates = new long[nIntervals][];

        for (int i = 0; i < nIntervals; i++) {
            if (i == 0) {
                dates[i] = new long[]{
                        validCutOff - interval,
                        validCutOff
                };
            } else {
                dates[i] = new long[]{
                        dates[i - 1][0] - interval,
                        dates[i - 1][0]
                };
            }
        }

        return dates;
    }

    public static String toLIBSVM(final List<MLSparseVector> feats,
                                  final int[] indexesToRemove) {
        int offset = 0;
        StringBuilder builder = new StringBuilder();
        int index = -1;
        for (MLSparseVector feat : feats) {
            index++;
            if (indexesToRemove != null &&
                    Arrays.binarySearch(indexesToRemove, index) >= 0) {
                continue;
            }

            if (feat.isEmpty() == true) {
                offset += feat.getLength();
                continue;
            }

            int[] indexes = feat.getIndexes();
            float[] values = feat.getValues();
            for (int i = 0; i < indexes.length; i++) {
                float val = values[i];
                if (val == Math.round(val)) {
                    builder.append(
                            " " + (offset + indexes[i]) + ":" + ((int) val));
                } else {
                    builder.append(" " + (offset + indexes[i]) + ":"
                            + String.format("%.5f", val));
                }
            }
            offset += feat.getLength();
        }
        return builder.toString();
    }

    public static MLSparseVector concat(final List<MLSparseVector> feats,
                                        final int[] indexesToRemove) {
        int length = 0;
        int nnz = 0;
        int index = -1;
        for (MLSparseVector vector : feats) {
            index++;
            if (indexesToRemove != null &&
                    Arrays.binarySearch(indexesToRemove, index) >= 0) {
                continue;
            }

            if (vector.isEmpty() == false) {
                nnz += vector.getIndexes().length;
            }
            length += vector.getLength();
        }
        if (nnz == 0) {
            return new MLSparseVector(null, null, null, length);
        }

        int[] indexes = new int[nnz];
        float[] values = new float[nnz];
        int cur = 0;
        int offset = 0;
        index = -1;
        for (MLSparseVector vector : feats) {
            index++;
            if (indexesToRemove != null &&
                    Arrays.binarySearch(indexesToRemove, index) >= 0) {
                continue;
            }

            int[] vecInds = vector.getIndexes();
            if (vecInds != null) {
                float[] vecVals = vector.getValues();
                for (int j = 0; j < vecInds.length; j++) {
                    indexes[cur] = offset + vecInds[j];
                    values[cur] = vecVals[j];
                    cur++;
                }
            }
            offset += vector.getLength();
        }
        return new MLSparseVector(indexes, values, null, length);
    }


    public static DMatrix toDMatrix(final List<MLSparseVector>[] features,
                                    final int[] indexesToRemove,
                                    final int[][] targets,
                                    final int targetIndex) throws XGBoostError {

        Iterator<LabeledPoint> it = new Iterator<LabeledPoint>() {
            int cur = 0;

            @Override
            public boolean hasNext() {
                return this.cur < features.length;
            }

            @Override
            public LabeledPoint next() {
                int index = this.cur;
                this.cur++;

                MLSparseVector row = concat(features[index], indexesToRemove);
                if (targets != null) {
                    if (row.isEmpty() == true) {
                        return new LabeledPoint(
                                targets[index][targetIndex],
                                new int[0],
                                new float[0]);
                    } else {
                        return new LabeledPoint(
                                targets[index][targetIndex],
                                row.getIndexes(),
                                row.getValues());
                    }
                } else {
                    if (row.isEmpty() == true) {
                        return new LabeledPoint(
                                0,
                                new int[0],
                                new float[0]);
                    } else {
                        return new LabeledPoint(
                                0,
                                row.getIndexes(),
                                row.getValues());
                    }
                }
            }
        };

        return new DMatrix(it, null);
    }

    public static DMatrix toDMatrix(final MLSparseMatrix features,
                                    final int[][] targets,
                                    final int targetIndex) throws XGBoostError {

        Iterator<LabeledPoint> it = new Iterator<LabeledPoint>() {
            int cur = 0;

            @Override
            public boolean hasNext() {
                return this.cur < features.getNRows();
            }

            @Override
            public LabeledPoint next() {
                int index = this.cur;
                this.cur++;

                MLSparseVector row = features.getRow(index, false);
                if (targets != null) {
                    if (row != null) {
                        return new LabeledPoint(
                                targets[index][targetIndex],
                                new int[0],
                                new float[0]);
                    } else {
                        return new LabeledPoint(
                                targets[index][targetIndex],
                                row.getIndexes(),
                                row.getValues());
                    }
                } else {
                    if (row != null) {
                        return new LabeledPoint(
                                0,
                                new int[0],
                                new float[0]);
                    } else {
                        return new LabeledPoint(
                                0,
                                row.getIndexes(),
                                row.getValues());
                    }
                }
            }
        };

        return new DMatrix(it, null);
    }


    public static int[] argsort(final int[][] a,
                                final int targetIndex) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return Integer.compare(a[i2][targetIndex], a[i1][targetIndex]);
            }
        });

        int[] ranking = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            if (i > 0 && a[indexes[i]][targetIndex] == a[indexes[i - 1]][targetIndex]) {
                //equal counts should have the same rank
                ranking[indexes[i]] = ranking[indexes[i - 1]];
            } else {
                ranking[indexes[i]] = i + 1;
            }
        }
        return ranking;
    }

    public static float sum(final float[] arr) {
        float sum = 0;
        for (float x : arr) {
            sum += x;
        }
        return sum;
    }

}
