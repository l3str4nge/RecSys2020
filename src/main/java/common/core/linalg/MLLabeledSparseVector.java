package common.core.linalg;

import java.io.Serializable;

public class MLLabeledSparseVector extends MLSparseVector
		implements Serializable {

	private static final long serialVersionUID = -93379512042314535L;
	private float label;

	public MLLabeledSparseVector(int[] indexesP, float[] valuesP, long[] datesP,
			int lengthP, float labelP) {
		super(indexesP, valuesP, datesP, lengthP);
		this.label = labelP;
	}

	public float getLabel() {
		return this.label;
	}

	public String toLIBSVMString() {
		if (label == Math.round(label)) {
			return String.format("%d", (int) this.label) + super.toLIBSVMString(0);
		} else {
			return String.format("%.5f", this.label) + super.toLIBSVMString(0);
		}
	}
}
