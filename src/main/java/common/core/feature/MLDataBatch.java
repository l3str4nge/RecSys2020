package common.core.feature;

public class MLDataBatch {
    public int size;
    public float[][] data;
    public int[][] dataDims;
    public String[] dataLabels;

    public float[][] targets;
    public int[][] targetDims;
    public String[] targetLabels;

    /**
     * Initialize data and targets arrays using the provided
     * dataDims and targetDims. If targetDims is not provided
     * then targets array is not allocated.
     */
    public void initBatch() {
        this.data = new float[this.dataDims.length][];
        for (int i = 0; i < this.dataDims.length; i++) {
            int total = 1;
            for (int j = 0; j < this.dataDims[i].length; j++) {
                //calculate total length of each dimension
                total = total * this.dataDims[i][j];
            }
            this.data[i] = new float[total];
        }

        if (this.targetDims == null) {
            return;
        }
        this.targets = new float[this.targetDims.length][];
        for (int i = 0; i < this.targetDims.length; i++) {
            int total = 1;
            for (int j = 0; j < this.targetDims[i].length; j++) {
                //calculate total length of each dimension
                total = total * this.targetDims[i][j];
            }
            this.targets[i] = new float[total];
        }
    }
}
