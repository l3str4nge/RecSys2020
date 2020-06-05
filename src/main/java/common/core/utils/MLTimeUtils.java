package common.core.utils;

import java.time.LocalDate;
import java.time.ZonedDateTime;
import java.time.temporal.TemporalAdjusters;

public class MLTimeUtils {

	/**
	 * Applies an offset and then moves the date to the last date of the month.
	 * e.g. (2017-02-28, 1) ---> 2017-03-31 (2017-03-31, 1) ---> 2017-04-30
	 *
	 * @param date
	 *            Date from which the offset will be applied.
	 * @param monthsForward
	 *            Number of months to offset before rounding to last day of
	 *            month.
	 * @return Offset date, moved to the last day of the month.
	 */
	public static LocalDate getLastDayOfSubsequentMonth(LocalDate date,
			int monthsForward) {
		return date.plusMonths(monthsForward)
				.with(TemporalAdjusters.lastDayOfMonth());
	}

	/**
	 * Gets the ZonedDateTime corresponding to the start of day on the last day
	 * of the month n-months forward.
	 *
	 * e.g. If dateTime is 2017-02-28 00:00 and monthsForward is 1, get
	 * 2017-01-31 00:00 returned. This function should be invariant to the
	 * day-of-month and the time of the input.
	 *
	 * @param dateTime
	 *            Date from which the months are offset.
	 * @param monthsForward
	 *            Number of months forward to offset (may be negative).
	 * @return Offset date, moved to the start-of-day on the last day of the
	 *         month.
	 */
	public static ZonedDateTime getStartOfLastDayOfOffsetMonth(
			ZonedDateTime dateTime, int monthsForward) {
		return dateTime.plusMonths(monthsForward)
				.with(TemporalAdjusters.lastDayOfMonth());
	}

}
