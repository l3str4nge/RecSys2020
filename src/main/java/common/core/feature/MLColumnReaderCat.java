package common.core.feature;

public class MLColumnReaderCat extends MLColumnReader {

    private static final long serialVersionUID = 6842439928297557467L;

    public MLColumnReaderCat(final String columnNameP,
                             final MLSparseFeature featureP) {
        super(columnNameP, featureP);
    }

    public MLColumnReaderCat(final String columnNameP,
                             final MLSparseFeature featureP,
                             final int columnIndexP) {
        super(columnNameP, featureP, columnIndexP);
    }

    @Override
    public void addRow(double value, int rowIndex) {
        this.feature.addRow(rowIndex, value + "");

    }

    @Override
    public void addRow(float value, int rowIndex) {
        this.feature.addRow(rowIndex, value + "");

    }

    @Override
    public void addRow(int value, int rowIndex) {
        this.feature.addRow(rowIndex, value + "");

    }

    @Override
    public void addRow(long value, int rowIndex) {
        this.feature.addRow(rowIndex, value + "");

    }

    @Override
    public void addRow(String value, int rowIndex) {
        this.feature.addRow(rowIndex, value);

    }

    @Override
    public void addRow(final String[] rowSplit, final int rowIndex) {

        String rowValue = rowSplit[this.columnIndex].trim();
        if (rowValue.length() == 0) {
            return;
        }
        this.feature.addRow(rowIndex, rowValue);
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
