package common.core.feature;

import java.io.Serializable;
import java.util.Arrays;

public class MLColumnRawStore implements Serializable {

    public enum RawStoreType {
        STRING,
        INT,
        LONG,
        FLOAT,
        DOUBLE
    }

    private static final long serialVersionUID = -3185293451205783897L;

    private RawStoreType storeType;
    private int nRows;

    private String[] dataString;
    private int[] dataInt;
    private long[] dataLong;
    private float[] dataFloat;
    private double[] dataDouble;

    private transient String[] dataStringCache;
    private transient int[] dataIntCache;
    private transient long[] dataLongCache;
    private transient float[] dataFloatCache;
    private transient double[] dataDoubleCache;

    public MLColumnRawStore(final RawStoreType storeTypeP, final int nRows) {
        this.storeType = storeTypeP;
        this.nRows = nRows;
        this.prepareForData();
    }

    public void addToStore(final int rowIndex, final double value) {
        if (this.storeType.equals(RawStoreType.DOUBLE) == false) {
            throw new IllegalArgumentException("store must be of type "
                    + RawStoreType.DOUBLE + ", current type " + this.storeType);
        }

        synchronized (this) {
            this.dataDouble[rowIndex] = value;
        }
    }

    public void addToStore(final int rowIndex, final float value) {
        if (this.storeType.equals(RawStoreType.FLOAT) == false) {
            throw new IllegalArgumentException("store must be of type "
                    + RawStoreType.FLOAT + ", current type " + this.storeType);
        }

        synchronized (this) {
            this.dataFloat[rowIndex] = value;
        }
    }

    public void addToStore(final int rowIndex, final int value) {
        if (this.storeType.equals(RawStoreType.INT) == false) {
            throw new IllegalArgumentException("store must be of type "
                    + RawStoreType.INT + ", current type " + this.storeType);
        }

        synchronized (this) {
            this.dataInt[rowIndex] = value;
        }
    }

    public void addToStore(final int rowIndex, final long value) {
        if (this.storeType.equals(RawStoreType.LONG) == false) {
            throw new IllegalArgumentException("store must be of type "
                    + RawStoreType.LONG + ", current type " + this.storeType);
        }

        synchronized (this) {
            this.dataLong[rowIndex] = value;
        }
    }

    public void addToStore(final int rowIndex, final String value) {
        switch (this.storeType) {
            case STRING: {
                synchronized (this) {
                    this.dataString[rowIndex] = value;
                }
                break;
            }

            case INT: {
                synchronized (this) {
                    this.dataInt[rowIndex] = Integer.parseInt(value);
                }
                break;
            }

            case LONG: {
                synchronized (this) {
                    this.dataLong[rowIndex] = Long.parseLong(value);
                }
                break;
            }

            case FLOAT: {
                synchronized (this) {
                    this.dataFloat[rowIndex] = Float.parseFloat(value);
                }
                break;
            }

            case DOUBLE: {
                synchronized (this) {
                    this.dataDouble[rowIndex] = Double.parseDouble(value);
                }
                break;
            }
        }
    }

    public int getCapacity() {
        return this.nRows;
    }

    public double getDoubleValue(final int rowIndex) {
        if (this.storeType.equals(RawStoreType.DOUBLE) == false) {
            throw new IllegalArgumentException("store must be of type"
                    + RawStoreType.DOUBLE + ", current type " + this.storeType);
        }

        return this.dataDouble[rowIndex];
    }

    public float getFloatValue(final int rowIndex) {
        if (this.storeType.equals(RawStoreType.FLOAT) == false) {
            throw new IllegalArgumentException("store must be of type"
                    + RawStoreType.FLOAT + ", current type " + this.storeType);
        }

        return this.dataFloat[rowIndex];
    }

    public int getIntValue(final int rowIndex) {
        if (this.storeType.equals(RawStoreType.INT) == false) {
            throw new IllegalArgumentException("store must be of type"
                    + RawStoreType.INT + ", current type " + this.storeType);
        }

        return this.dataInt[rowIndex];
    }

    public long getLongValue(final int rowIndex) {
        if (this.storeType.equals(RawStoreType.LONG) == false) {
            throw new IllegalArgumentException("store must be of type"
                    + RawStoreType.LONG + ", current type " + this.storeType);
        }

        return this.dataLong[rowIndex];
    }

    public RawStoreType getStoreType() {
        return this.storeType;
    }

    public String getStringValue(final int rowIndex) {
        if (this.storeType.equals(RawStoreType.STRING) == false) {
            throw new IllegalArgumentException("store must be of type"
                    + RawStoreType.STRING + ", current type " + this.storeType);
        }

        return this.dataString[rowIndex];
    }

    public Object getValue(final int rowIndex) {
        switch (this.storeType) {
            case STRING: {
                return this.dataString[rowIndex];
            }

            case INT: {
                return this.dataInt[rowIndex];
            }

            case LONG: {
                return this.dataLong[rowIndex];
            }

            case FLOAT: {
                return this.dataFloat[rowIndex];
            }

            case DOUBLE: {
                return this.dataDouble[rowIndex];
            }

            default: {
                return null;
            }
        }
    }

    /**
     * Mutates the store *in-place* to remove all rows outside of [fromIndex,
     * toIndex).
     *
     * @param fromIndex Start (inclusive) of rows to keep.
     * @param toIndex   End (exclusive) or rows to keep.
     */
    public void sliceRows(final int fromIndex, final int toIndex) {

        this.nRows = toIndex - fromIndex;
        switch (this.storeType) {
            case STRING: {
                this.dataString = Arrays.copyOfRange(this.dataString,
                        fromIndex, toIndex);
                break;
            }

            case INT: {
                this.dataInt = Arrays.copyOfRange(this.dataInt, fromIndex,
                        toIndex);
                break;
            }

            case LONG: {
                this.dataLong = Arrays.copyOfRange(this.dataLong, fromIndex,
                        toIndex);
                break;
            }

            case FLOAT: {
                this.dataFloat = Arrays.copyOfRange(this.dataFloat, fromIndex
                        , toIndex);
                break;
            }

            case DOUBLE: {
                this.dataDouble = Arrays.copyOfRange(this.dataDouble,
                        fromIndex, toIndex);
                break;
            }
        }
    }

    public int getNumRows() {
        return nRows;
    }

    public synchronized void prepareToSerialize(final boolean withData) {
        if (!withData) {
            switch (this.storeType) {
                case STRING: {
                    this.dataStringCache = this.dataString;
                    this.dataString = null;
                    break;
                }
                case INT: {
                    this.dataIntCache = this.dataInt;
                    this.dataInt = null;
                    break;
                }
                case LONG: {
                    this.dataLongCache = this.dataLong;
                    this.dataLong = null;
                    break;
                }
                case FLOAT: {
                    this.dataFloatCache = this.dataFloat;
                    this.dataFloat = null;
                    break;
                }
                case DOUBLE: {
                    this.dataDoubleCache = this.dataDouble;
                    this.dataDouble = null;
                    break;
                }
            }
        }
    }

    public synchronized void finishSerialize() {
        switch (this.storeType) {
            case STRING: {
                if (this.dataString == null) {
                    this.dataString = this.dataStringCache;
                }
                break;
            }
            case INT: {
                if (this.dataInt == null) {
                    this.dataInt = this.dataIntCache;
                }
                break;
            }
            case LONG: {
                if (this.dataLong == null) {
                    this.dataLong = this.dataLongCache;
                }
                break;
            }
            case FLOAT: {
                if (this.dataFloat == null) {
                    this.dataFloat = this.dataFloatCache;
                }
                break;
            }
            case DOUBLE: {
                if (this.dataDouble == null) {
                    this.dataDouble = this.dataDoubleCache;
                }
                break;
            }
        }
    }

    public synchronized void prepareForData() {
        this.prepareForData(this.nRows);
    }

    public synchronized void prepareForData(final int nRowsP) {
        this.nRows = nRowsP;
        this.dataString = (this.storeType == RawStoreType.STRING) ?
                new String[this.nRows] : null;
        this.dataInt = (this.storeType == RawStoreType.INT) ?
                new int[this.nRows] : null;
        this.dataLong = (this.storeType == RawStoreType.LONG) ?
                new long[this.nRows] : null;
        this.dataFloat = (this.storeType == RawStoreType.FLOAT) ?
                new float[this.nRows] : null;
        this.dataDouble = (this.storeType == RawStoreType.DOUBLE) ?
                new double[this.nRows] : null;
    }
}
