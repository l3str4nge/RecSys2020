package common.core.linalg;

import java.io.Serializable;

public class MutableInt implements Serializable {

	private static final long serialVersionUID = -3398549876974709321L;
	public int value;

	protected MutableInt() {
	}

	public MutableInt(final int valueP) {
		this.value = valueP;
	}

	public void increment() {
		this.value++;
	}
}