package common.core.feature;

public class MLColumnReaderRaw extends MLColumnReader {

    private static final long serialVersionUID = 8920569892370746289L;

    public MLColumnReaderRaw(final String columnNameP,
                             final MLColumnRawStore rawStoreP) {
        super(columnNameP, rawStoreP);
    }

    public MLColumnReaderRaw(final String columnNameP,
                             final MLColumnRawStore rawStoreP,
                             final int columnIndexP) {
        super(columnNameP, rawStoreP, columnIndexP);
    }

    @Override
    public void addRow(double value, int rowIndex) {
        this.rawStore.addToStore(rowIndex, value);
    }

    @Override
    public void addRow(float value, int rowIndex) {
        this.rawStore.addToStore(rowIndex, value);
    }

    @Override
    public void addRow(int value, int rowIndex) {
        this.rawStore.addToStore(rowIndex, value);
    }

    @Override
    public void addRow(long value, int rowIndex) {
        this.rawStore.addToStore(rowIndex, value);
    }

    @Override
    public void addRow(String value, int rowIndex) {
        this.rawStore.addToStore(rowIndex, value);
    }

    @Override
    public void addRow(final String[] rowSplit, final int rowIndex) {

        String rowValue = rowSplit[this.columnIndex].trim();
        if (rowValue.length() == 0) {
            return;
        }
        this.rawStore.addToStore(rowIndex, rowValue);
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
