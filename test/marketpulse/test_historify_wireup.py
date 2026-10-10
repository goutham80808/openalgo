"""Backfill resume: windows whose trading sessions are already stored are skipped."""

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "marketpulse"))

from historify_wireup import _expected_sessions, _missing_windows  # noqa: E402

# 2026-10-05..09 is a plain Mon-Fri week with no exchange holiday.
MON, FRI = dt.date(2026, 10, 5), dt.date(2026, 10, 9)
FULL_WEEK = {dt.date(2026, 10, d) for d in (5, 6, 7, 8, 9)}


def test_expected_sessions_counts_trading_days_only():
    assert _expected_sessions(MON, FRI) == 5
    assert _expected_sessions(dt.date(2026, 10, 3), dt.date(2026, 10, 4)) == 0


def test_fully_stored_window_is_skipped():
    spans = [(MON, FRI, _expected_sessions(MON, FRI))]
    assert _missing_windows(FULL_WEEK, spans) == []


def test_window_with_a_missing_session_comes_back():
    spans = [(MON, FRI, _expected_sessions(MON, FRI))]
    assert _missing_windows(FULL_WEEK - {dt.date(2026, 10, 7)}, spans) == [(MON, FRI)]


def test_empty_store_returns_every_window():
    spans = [(MON, FRI, _expected_sessions(MON, FRI))]
    assert _missing_windows(set(), spans) == [(MON, FRI)]
