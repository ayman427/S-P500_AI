"""Exchange sessions and admissible prospective signal times."""
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=16)
def _calendar(year: int):
    return xcals.get_calendar("XNYS", start=f"{year - 1}-01-01", end=f"{year + 2}-12-31")


def execution_window(signal_date):
    session = pd.Timestamp(signal_date).normalize()
    if session.tzinfo is not None:
        raise ValueError("Signal dates must be exchange session dates without a timezone.")
    calendar = _calendar(session.year)
    if not calendar.is_session(session):
        raise ValueError(f"{session.date()} is not an exchange session.")
    execution = calendar.next_session(session)
    return calendar.session_close(session), calendar.session_open(execution), execution


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def validate_signal_time(signal_date, created_at):
    created = pd.Timestamp(created_at)
    if pd.isna(created) or created.tzinfo is None:
        raise ValueError("A timezone-aware signal creation timestamp is required.")
    close, opening, execution = execution_window(signal_date)
    if not close <= created < opening:
        raise ValueError("Paper signals must be created after the signal session closes and before the next session opens. Sync prices and wait for a valid signal window.")
    return execution
