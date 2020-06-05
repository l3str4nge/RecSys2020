package common.core.utils;

import java.util.concurrent.TimeUnit;

import com.google.common.base.Stopwatch;

/**
 * inhouse timer using guava Stopwatch backend
 * to use timer, create object then call tic to start
 * then get printouts with toc or tocLoop
 * just before entering a loop, you can (optionally) get more accurate estimate by calling resetLoop
 * <p>
 * the MLTimer class is designed to be created one per project and re-used, although this is not required.
 */
public class MLTimer {

	private String name;
	private long loopSize;
	private Stopwatch timer;
	private transient Double loopStart = null;
	private transient Integer countStart = null;

	/**
	 * create simple timer with only name, unstarted; call tic to start
	 *
	 * @param nameP
	 */
	public MLTimer(final String nameP) {
		this.name = nameP;
		this.loopSize = 0;
		this.timer = Stopwatch.createUnstarted();
	}

	/**
	 * create timer that supports a loop of size loopSizeP; providing this allows timer to estimate time remaining.
	 *
	 * @param nameP
	 * @param loopSizeP
	 */
	public MLTimer(final String nameP, final int loopSizeP) {
		this.name = nameP;
		this.loopSize = loopSizeP;
		this.timer = Stopwatch.createUnstarted();
	}

	/**
	 * reset timer and start it
	 */
	public synchronized void tic() {
		this.timer.reset().start();
	}

	/**
	 * print the current elapsed time since last tic
	 */
	public synchronized void toc() {
		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		System.out.printf("%s: elapsed [%s]\n", this.name,
				formatSeconds((float) elapsedTime));
	}

	/**
	 * print the current elapsed time since last time, with custom message prefix
	 *
	 * @param message prefix of the printout
	 */
	public synchronized void toc(final String message) {
		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		System.out.printf("%s: %s elapsed [%s]\n", this.name, message,
				formatSeconds((float) elapsedTime));
	}

	/**
	 * print the current timer output since last tic, assuming loop (compute it/s)
	 *
	 * @param curLoop
	 */
	public synchronized void tocLoop(final int curLoop) {
		tocLoop("", curLoop);
	}

	/**
	 * set loop size for remaining time estimate
	 * you should call this right before entering a loop to obtain accurate speed estimate
	 * <p>
	 * this also resets internals for loop speed estimate for sensible speed calculation if you use the timer across multiple loops.
	 *
	 * @param loopSize
	 */
	public synchronized void resetLoop(final int loopSize) {
		this.loopSize = loopSize;
		this.countStart = 0;
		this.loopStart = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
	}

	/**
	 * print the current timer output since last tic, assuming loop (computer it/s)
	 * allows prefix message
	 * <p>
	 * first loop will have speed 0, subsequent loop uses first call of tocLoop to estimate time.
	 * if you don't know the size, you may pass 0 to skip remaining time calculation
	 * @param curLoop
	 */
	public synchronized void tocLoop(String message, final int curLoop) {
		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		String speedStr = "N/A";
		double speed = 0;
		if (this.loopStart == null) {
			this.loopStart = elapsedTime;
			this.countStart = curLoop;
		} else {
			//do some accurate speed estimate
			if (elapsedTime != this.loopStart) {
				speed = (curLoop - this.countStart) / (elapsedTime - this.loopStart);
			}
			speedStr = String.format("%.0f/s", speed);
		}

		if (this.loopSize > 0 && speed > 0) {
			double remainTime = (this.loopSize - curLoop) / speed;

			System.out.printf(
					"%s: %s[%2.2f%%] elapsed [%s] cur_spd [%s] remain [%s]\n",
					this.name, message, (curLoop * 100f) / this.loopSize,
					formatSeconds(elapsedTime), speedStr,
					formatSeconds(remainTime));
		} else {
			System.out.printf("%s: %s [%d] elapsed [%s] cur_spd [%s]\n",
					this.name, message, curLoop, formatSeconds(elapsedTime),
					speedStr);
		}
	}

	private static String formatSeconds(double secondsF) {
		if (secondsF < 0) {
			return Double.toString(secondsF);
		}
		TimeUnit base = TimeUnit.SECONDS;
		int s = (int) Math.floor(secondsF);
		// float remainder = (float) (secondsF - s);

		long days = base.toDays(s);
		s -= TimeUnit.DAYS.toSeconds(days);
		long hours = base.toHours(s);
		s -= TimeUnit.HOURS.toSeconds(hours);
		long minutes = base.toMinutes(s);
		s -= TimeUnit.MINUTES.toSeconds(minutes);
		long secondsL = base.toSeconds(s);

		StringBuilder sb = new StringBuilder();
		if (days > 0) {
			sb.append(days);
			sb.append(" days ");
		}
		if (hours > 0 || days > 0) {
			sb.append(hours);
			sb.append(" hr ");
		}
		if (hours > 0 || days > 0 || minutes > 0) {
			sb.append(minutes);
			sb.append(" min ");
		}
		sb.append(secondsL + " sec");

		return sb.toString();
	}

	/**
	 * timer should be checked qualitatively
	 *
	 * @param args
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws InterruptedException {
		MLTimer timer = new MLTimer("test");
		timer.tic();
		Thread.sleep(2000);
		timer.toc("sleep");
//		timer.resetLoop(5);
		for (int i = 0; i < 5; i++) {
			Thread.sleep(100);
			timer.tocLoop(i + 1);
		}
		timer.resetLoop(5);
		for (int i = 0; i < 5; i++) {
			Thread.sleep(100);
			timer.tocLoop("another loop", i + 1);
		}
	}

}
