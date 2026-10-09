#!/usr/bin/env python3
"""Report per-symbol candle coverage in historify.duckdb vs mp_universe.

Phase 0 acceptance gate: >= 95% of active universe members have daily candles
for >= 95% of the last 250 trading sessions.

Opens historify.duckdb read-only; safe to run any time, even mid-download.

Usage:
    uv run scripts/marketpulse/coverage_report.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database.engine_factory import create_db_engine  # noqa: E402
from database.historify_db import get_db_path  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SESSIONS = 250
COVERAGE_FLOOR = 0.95


def _trading_sessions(n: int) -> list[str]:
    from utils.trading_calendar import is_trading_day

    out: list[str] = []
    day = dt.date.today()
    while len(out) < n:
        if is_trading_day(day):
            out.append(day.isoformat())
        day -= dt.timedelta(days=1)
    return out


def main() -> int:
    import duckdb

    sessions = _trading_sessions(SESSIONS)
    # _trading_sessions walks back from today, so the list is newest-first.
    earliest = min(sessions)

    engine = create_db_engine()
    conn = duckdb.connect(get_db_path(), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT symbol, exchange,
                   COUNT(DISTINCT CAST(timestamp AS DATE)) AS days
            FROM market_data
            WHERE interval = 'D' AND CAST(timestamp AS DATE) >= ?
            GROUP BY symbol, exchange
            """,
            [earliest],
        ).fetchall()
    finally:
        conn.close()

    coverage = {(s, e): d for s, e, d in rows}
    from sqlalchemy import text

    with engine.connect() as sq:
        members = sq.execute(
            text("SELECT symbol, exchange FROM mp_universe WHERE active_to IS NULL")
        ).fetchall()
    engine.dispose()

    if not members:
        print("mp_universe has no active members; run the universe refresh first")
        return 1

    needed = min(len(sessions), SESSIONS)
    ok = 0
    for symbol, exchange in members:
        days = coverage.get((symbol, exchange), 0)
        if days >= needed * COVERAGE_FLOOR:
            ok += 1
    percent = 100.0 * ok / len(members)
    print(f"members: {len(members)}")
    print(f"candle coverage window: {earliest} .. {max(sessions)} ({needed} sessions)")
    print(f"members with >= {COVERAGE_FLOOR:.0%} coverage: {ok} ({percent:.1f}%)")
    verdict = "PASS" if percent >= 95 else "BELOW GATE (95%)"
    print(f"Phase 0 acceptance gate: {verdict}")
    return 0 if percent >= 95 else 2


if __name__ == "__main__":
    sys.exit(main())
