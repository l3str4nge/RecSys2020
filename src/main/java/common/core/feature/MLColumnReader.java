package common.core.feature;

import java.io.Serializable;

public abstract class MLColumnReader implements Serializable {

    private static final long serialVersionUID = 950896977654034832L;

    protected String columnName;
    protected int columnIndex;
    protected MLSparseFeature feature;
    protected MLColumnRawStore rawStore;

    public MLColumnReader(final String columnNameP,
                          final MLColumnRawStore rawStoreP) {

        this.columnName = columnNameP;
        this.rawStore = rawStoreP;
        this.columnIndex = -1;
    }

    public MLColumnReader(final String columnNameP,
                          final MLColumnRawStore rawStoreP,
                          final int columnIndexP) {
        this.columnName = columnNameP;
        this.rawStore = rawStoreP;
        this.columnIndex = columnIndexP;
    }

    public MLColumnReader(final String columnNameP,
                          final MLSparseFeature featureP) {
        this.columnName = columnNameP;
        this.feature = featureP;
        this.columnIndex = -1;
    }

    public MLColumnReader(final String columnNameP,
                          final MLSparseFeature featureP,
                          final int columnIndexP) {
        this.columnName = columnNameP;
        this.feature = featureP;
        this.columnIndex = columnIndexP;
    }

    public abstract void addRow(final String[] row, final int rowIndex);

    public abstract void addRow(final String value, final int rowIndex);

    public abstract void addRow(final long value, final int rowIndex);

    public abstract void addRow(final int value, final int rowIndex);

    public abstract void addRow(final float value, final int rowIndex);

    public abstract void addRow(final double value, final int rowIndex);

    public int getColumnIndex() {
        return this.columnIndex;
    }

    public String getColumnName() {
        return this.columnName;
    }

    public MLSparseFeature getFeature() {
        return this.feature;
    }

    public MLColumnRawStore getRawStore() {
        return this.rawStore;
    }

    public abstract void setColumnIndex(final String[] header);
}
