"""Pure date-math for expanding a recurrence rule into concrete occurrences.

No I/O, no persistence: this module only turns "toda segunda por 8 semanas"
style rules into a sorted list of ``datetime`` occurrences, which the caller
(``schedule_store``) then persists as concrete rows. Exceptions and extra
dates are applied here, before anything is written to disk, so what gets
saved is always the final, reviewed list.
"""

from datetime import date, datetime, time, timedelta

INTERVAL_TYPES = ("once", "daily", "weekly", "monthly")
# Hard ceiling on how many occurrences a single rule can expand to. Protects
# against a runaway loop from a bad end_date/occurrence_count combination and
# against a user accidentally scheduling years of daily videos at once.
MAX_OCCURRENCES = 366


def _add_months(base: date, months: int) -> date:
    """Advance ``base`` by ``months``, clamping to the target month's last day."""
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = base.day
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1


def expand_occurrences(
    start_date: date,
    time_of_day: time,
    interval_type: str,
    interval_step: int = 1,
    end_date: date | None = None,
    occurrence_count: int | None = None,
    exceptions: set[date] | None = None,
    extra_dates: list[datetime] | None = None,
) -> list[datetime]:
    """Expand a recurrence rule into a sorted list of concrete occurrences.

    ``interval_type="once"`` ignores ``end_date``/``occurrence_count`` and
    always yields exactly one occurrence at ``start_date``+``time_of_day``.
    Recurring types require exactly one of ``end_date``/``occurrence_count``
    to know when to stop.
    """
    if interval_type not in INTERVAL_TYPES:
        raise ValueError(f"unsupported interval_type: {interval_type!r}")
    if interval_step < 1:
        raise ValueError("interval_step must be a positive integer")

    if interval_type == "once":
        occurrences = [datetime.combine(start_date, time_of_day)]
    else:
        if occurrence_count is None and end_date is None:
            raise ValueError(
                "recurring rules require either end_date or occurrence_count"
            )
        if occurrence_count is not None and occurrence_count > MAX_OCCURRENCES:
            raise ValueError(
                f"occurrence_count exceeds the maximum of {MAX_OCCURRENCES}"
            )

        occurrences = []
        index = 0
        while True:
            # Monthly always advances from the original start_date's day of
            # month, not from the previous (possibly clamped) occurrence:
            # Jan 31 -> Feb 28 -> Mar 31, never Jan 31 -> Feb 28 -> Mar 28.
            if interval_type == "daily":
                current_date = start_date + timedelta(days=index * interval_step)
            elif interval_type == "weekly":
                current_date = start_date + timedelta(weeks=index * interval_step)
            else:  # monthly
                current_date = _add_months(start_date, index * interval_step)

            if end_date is not None and current_date > end_date:
                break
            if occurrence_count is not None and len(occurrences) >= occurrence_count:
                break
            if len(occurrences) >= MAX_OCCURRENCES:
                raise ValueError(
                    f"end_date produces more than {MAX_OCCURRENCES} occurrences"
                )

            occurrences.append(datetime.combine(current_date, time_of_day))
            index += 1

    exceptions = exceptions or set()
    occurrences = [
        occurrence for occurrence in occurrences if occurrence.date() not in exceptions
    ]
    occurrences.extend(extra_dates or [])
    occurrences.sort()
    return occurrences


def expand_daily_interval_occurrences(
    start_date: date,
    start_time: time,
    interval_hours: float,
    end_date: date | None = None,
    occurrence_count_days: int | None = None,
    exceptions: set[date] | None = None,
) -> list[datetime]:
    """Expand an "N times a day" cadence into concrete occurrences.

    Every day in range gets slots at ``start_time``, ``start_time +
    interval_hours``, ``+ 2*interval_hours``, ... for as long as the slot
    still falls on that same calendar day; a slot that would roll past
    midnight is dropped, not wrapped into the next day, which restarts its
    own run from ``start_time``. Mirrors ``expand_occurrences``: exactly one
    of ``end_date``/``occurrence_count_days`` bounds the range, and
    ``exceptions`` removes matching calendar days after expansion.
    """
    if not (0 < interval_hours <= 24):
        raise ValueError("interval_hours must be greater than 0 and at most 24")
    if occurrence_count_days is None and end_date is None:
        raise ValueError(
            "requires either end_date or occurrence_count_days"
        )

    occurrences = []
    day_index = 0
    while True:
        current_date = start_date + timedelta(days=day_index)
        if end_date is not None and current_date > end_date:
            break
        if (
            occurrence_count_days is not None
            and day_index >= occurrence_count_days
        ):
            break

        day_start = datetime.combine(current_date, start_time)
        offset_hours = 0.0
        while offset_hours < 24:
            occurrence = day_start + timedelta(hours=offset_hours)
            if occurrence.date() != current_date:
                break
            if len(occurrences) >= MAX_OCCURRENCES:
                raise ValueError(
                    f"produces more than {MAX_OCCURRENCES} occurrences"
                )
            occurrences.append(occurrence)
            offset_hours += interval_hours
        day_index += 1

    exceptions = exceptions or set()
    occurrences = [
        occurrence for occurrence in occurrences if occurrence.date() not in exceptions
    ]
    occurrences.sort()
    return occurrences
