package common.core.feature;

public class MLColumnReaderNum extends MLColumnReader {

    private static final long serialVersionUID = 235850967662026689L;

    public MLColumnReaderNum(final String columnNameP,
                             final MLSparseFeature featureP) {
        super(columnNameP, featureP);
    }

    public MLColumnReaderNum(final String columnNameP,
                             final MLSparseFeature featureP,
                             final int columnIndexP) {
        super(columnNameP, featureP, columnIndexP);
    }

    @Override
    public void addRow(double value, int rowIndex) {
        this.feature.addRow(rowIndex, (float) value);
    }

    @Override
    public void addRow(float value, int rowIndex) {
        this.feature.addRow(rowIndex, value);
    }

    @Override
    public void addRow(int value, int rowIndex) {
        this.feature.addRow(rowIndex, (float) value);
    }

    @Override
    public void addRow(long value, int rowIndex) {
        this.feature.addRow(rowIndex, (float) value);
    }

    @Override
    public void addRow(String value, int rowIndex) {
        this.feature.addRow(rowIndex, Float.parseFloat(value));
    }

    @Override
    public void addRow(final String[] rowSplit, final int rowIndex) {

        String rowValue = rowSplit[this.columnIndex].trim();
        if (rowValue.length() == 0) {
            return;
        }
        this.feature.addRow(rowIndex, Float.parseFloat(rowValue));
    }

    @Override
    public void setColumnIndex(final String[] header) {

        // default implementation tries to match columnName to header
        for (int i = 0; i < header.length; i++) {
            if (this.columnName.equals(header[i]) == true) {
                this.columnIndex = i;
                break;
            }
        }

        if (this.columnIndex < 0) {
            throw new IllegalArgumentException(
                    "column " + this.columnName + " not found in headder");
        }
    }

}
