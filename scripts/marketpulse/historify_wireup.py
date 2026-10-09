#!/usr/bin/env python3
"""Wire the Market Pulse universe into Historify: watchlist + schedule + backfill.

What it does:
  1. Reads active members from mp_universe.
  2. Adds them to the Historify watchlist in batches of 200 (idempotent —
     existing symbols are skipped by bulk_add_to_watchlist).
  3. Creates the daily 17:30 IST EOD schedule (idempotent on schedule id).
  4. Optionally backfills daily candles over quarterly windows (--backfill),
     paced, one symbol at a time, recording failures.

The watchlist and schedule steps write to historify.duckdb directly and need
no broker session. The backfill goes through the platform's history service
and therefore needs a logged-in broker session (run while the app is up and
authenticated, or right after logging in).

Usage:
    uv run scripts/marketpulse/historify_wireup.py --dry-run
    uv run scripts/marketpulse/historify_wireup.py
    uv run scripts/marketpulse/historify_wireup.py --backfill --backfill-months 24
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database.engine_factory import create_db_engine  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SCHEDULE_ID = "mp-eod"
BATCH_SIZE = 200
BACKFILL_STEP_DAYS = 90
BACKFILL_PAUSE_SECONDS = 0.3


def load_universe(engine) -> list[dict]:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT symbol, exchange, name FROM mp_universe "
                    "WHERE active_to IS NULL ORDER BY symbol"
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def add_to_watchlist(members: list[dict], dry_run: bool) -> tuple[int, int, int]:
    from database.historify_db import bulk_add_to_watchlist

    added = skipped = 0
    failed: list[dict] = []
    for start in range(0, len(members), BATCH_SIZE):
        batch = members[start : start + BATCH_SIZE]
        if dry_run:
            logger.info(f"[dry-run] would add {len(batch)} watchlist symbols")
            added += len(batch)
            continue
        a, s, f = bulk_add_to_watchlist(batch)
        added += a
        skipped += s
        failed.extend(f)
    return added, skipped, len(failed)


def ensure_schedule(dry_run: bool) -> None:
    from database.historify_db import create_schedule

    if dry_run:
        logger.info("[dry-run] would create schedule mp-eod (daily 17:30, D, lookback 5)")
        return
    success, msg = create_schedule(
        schedule_id=SCHEDULE_ID,
        name="Market Pulse EOD universe",
        schedule_type="daily",
        data_interval="D",
        time_of_day="17:30",
        download_source="watchlist",
        lookback_days=5,
        description="Market Pulse nightly candle pull (self-healing 5-day lookback)",
    )
    if success:
        logger.info(f"schedule created: {msg}")
    else:
        # Already exists is the normal re-run path; anything else is real.
        logger.info(f"schedule: {msg}")


def backfill(members: list[dict], months: int, api_key: str) -> None:
    from services.historify_service import download_data

    end = dt.date.today()
    start = end - dt.timedelta(days=int(months * 30.5))
    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + dt.timedelta(days=BACKFILL_STEP_DAYS), end)
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + dt.timedelta(days=1)

    failures = []
    t0 = time.monotonic()
    for index, member in enumerate(members):
        for window_start, window_end in windows:
            try:
                success, response, _ = download_data(
                    symbol=member["symbol"],
                    exchange=member["exchange"],
                    interval="D",
                    start_date=window_start,
                    end_date=window_end,
                    api_key=api_key,
                )
                if not success:
                    failures.append((member["symbol"], window_start, response.get("message")))
            except Exception:
                logger.exception(f"backfill failed for {member['symbol']} window {window_start}")
                failures.append((member["symbol"], window_start, "exception"))
            time.sleep(BACKFILL_PAUSE_SECONDS)
        if index % 100 == 0:
            elapsed = time.monotonic() - t0
            logger.info(
                f"backfill progress: {index + 1}/{len(members)} symbols, "
                f"{len(failures)} failures, {elapsed:.0f}s elapsed"
            )
    logger.info(
        f"backfill complete: {len(members)} symbols x {len(windows)} windows, "
        f"{len(failures)} failures in {time.monotonic() - t0:.0f}s"
    )


def _api_key() -> str | None:
    try:
        from database.auth_db import get_first_available_api_key

        return get_first_available_api_key()
    except Exception:
        logger.exception("could not read an API key for backfill")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--backfill", action="store_true", help="backfill daily candles")
    parser.add_argument("--backfill-months", type=int, default=24)
    args = parser.parse_args()

    engine = create_db_engine()
    try:
        members = load_universe(engine)
        logger.info(f"universe: {len(members)} active members")
        added, skipped, failed = add_to_watchlist(members, args.dry_run)
        logger.info(f"watchlist: added={added} skipped={skipped} failed={failed}")
        ensure_schedule(args.dry_run)
        if args.backfill:
            if args.dry_run:
                logger.info("[dry-run] backfill skipped")
            else:
                api_key = _api_key()
                if not api_key:
                    logger.error("backfill needs an API key; generate one at /apikey")
                    return 1
                backfill(members, args.backfill_months, api_key)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
