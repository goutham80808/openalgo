"""Nightly corporate-actions cross-check for Market Pulse.

Two halves:
  1. Pull the NSE corporate-actions calendar for the lookback window and
     upsert price-relevant actions (splits, bonuses, face-value changes,
     demergers) into the mp_adjustments ledger. Dividends are deliberately
     not ledgered: the screens' dividend policy is pinned to price-return.
  2. Walk every active mp_universe member's daily candles in
     historify.duckdb (read-only) and flag overnight moves past 35% that no
     ledger row explains. Flags are logged as a trader-readable warning —
     never a crash — because one bad night must not take the pipeline down.

Runs on a real OS thread like the other marketpulse jobs. CLI:
    uv run python -m services.marketpulse_jobs.ca_check --list-recent 90
"""

import datetime as dt
import re
import sys

import pandas as pd
from sqlalchemy import text

from services.marketpulse.adjustments import detect_unexplained_gaps
from utils.logging import get_logger

logger = get_logger(__name__)

LOOKBACK_DAYS = 45
GAP_THRESHOLD = 0.35


def parse_action(subject: str) -> tuple[str | None, float | None]:
    """NSE subject line -> (action, price ratio applied to pre-ex history).

    ratio multiplies prices before the ex-date: a 10->2 sub-division is 0.2,
    a bonus 1:1 is 0.5 (b/(a+b)), a consolidation 2->10 is 5.0.
    """
    if not subject:
        return None, None
    face = re.search(r"From\s+Rs\.?\s*([\d.]+).*?To\s+Rs\.?\s*([\d.]+)", subject, re.IGNORECASE)
    if face and "split" in subject.lower():
        before, after = float(face.group(1)), float(face.group(2))
        if before > 0:
            return "split", after / before
    bonus = re.search(r"bonus\s+(\d+)\s*:\s*(\d+)", subject, re.IGNORECASE)
    if bonus:
        new, held = int(bonus.group(1)), int(bonus.group(2))
        if new + held > 0:
            return "bonus", held / (new + held)
    if "demerger" in subject.lower():
        return "demerger", None
    if "amalgamation" in subject.lower() or "scheme" in subject.lower():
        return "merger", None
    return None, None


def _iso(nse_date: str) -> str | None:
    try:
        return dt.datetime.strptime(str(nse_date).strip(), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def sync_ledger(engine, fetch=None, lookback_days: int = LOOKBACK_DAYS) -> int:
    """Upsert recent price-relevant corporate actions into mp_adjustments."""
    from services.marketpulse_jobs import nse_client

    fetch = fetch or nse_client.fetch_corporate_actions_calendar
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=lookback_days)
    rows = fetch(from_date.isoformat(), to_date.isoformat())

    ledger: list[dict] = []
    for row in rows:
        ex_date = _iso(row.get("exDate", ""))
        action, ratio = parse_action(str(row.get("subject", "")))
        symbol = str(row.get("symbol", "")).strip().upper()
        if not (ex_date and action and ratio and symbol):
            continue
        ledger.append(
            {
                "symbol": symbol,
                "exchange": "NSE",
                "ex_date": ex_date,
                "action": action,
                "ratio": ratio,
                "source": "nse",
            }
        )
    if ledger:
        stmt = text(
            """
            INSERT INTO mp_adjustments (symbol, exchange, ex_date, action, ratio, source)
            VALUES (:symbol, :exchange, :ex_date, :action, :ratio, :source)
            ON CONFLICT(symbol, exchange, ex_date, action) DO UPDATE SET
                ratio=excluded.ratio, source=excluded.source
            """
        )
        with engine.begin() as conn:
            conn.execute(stmt, ledger)
    logger.info(
        f"corporate actions: {len(ledger)} price-relevant rows synced "
        f"over {lookback_days} days ({len(rows)} calendar rows fetched)"
    )
    return len(ledger)


def cross_check(
    engine, candles_by_symbol: dict[tuple[str, str], pd.DataFrame] | None = None
) -> list[dict]:
    """Flag unexplained overnight gaps across the active universe.

    candles_by_symbol: optional injection point ({(symbol, exchange):
    DataFrame[date, close]}) so tests need no DuckDB. When absent the frames
    are read from historify.duckdb read-only.
    """
    if candles_by_symbol is None:
        candles_by_symbol = _load_candles(engine)

    with engine.connect() as conn:
        members = conn.execute(
            text("SELECT symbol, exchange FROM mp_universe WHERE active_to IS NULL")
        ).fetchall()
        ledger = conn.execute(text("SELECT symbol, ex_date, action FROM mp_adjustments")).fetchall()

    known_by_symbol: dict[str, list[dict]] = {}
    for symbol, ex_date, action in ledger:
        known_by_symbol.setdefault(symbol, []).append({"ex_date": ex_date, "action": action})

    flags: list[dict] = []
    for symbol, exchange in members:
        frame = candles_by_symbol.get((symbol, exchange))
        if frame is None or len(frame) < 2:
            continue
        frame = frame.copy()
        frame["symbol"] = symbol
        frame["exchange"] = exchange
        flags.extend(
            detect_unexplained_gaps(
                frame, known=known_by_symbol.get(symbol, []), threshold=GAP_THRESHOLD
            )
        )

    if flags:
        logger.warning(
            f"{len(flags)} stocks moved overnight in a way their adjustment "
            "history does not explain — check these before trusting screens "
            "built on them"
        )
    else:
        logger.info("corporate-action cross-check clean: no unexplained gaps")
    return flags


def _load_candles(engine) -> dict[tuple[str, str], pd.DataFrame]:
    """Daily closes per active universe member, read-only from DuckDB."""
    import duckdb
    import pandas as pd

    from database.historify_db import get_db_path

    with engine.connect() as conn:
        members = conn.execute(
            text("SELECT symbol, exchange FROM mp_universe WHERE active_to IS NULL")
        ).fetchall()
    if not members:
        return {}

    duck = duckdb.connect(get_db_path(), read_only=True)
    try:
        # historify stores UTC epoch seconds; daily candles are stamped at IST
        # midnight, so the IST offset keeps the date on the right session day
        # (same pattern as historify_db's ist_offset usage).
        frame = duck.execute(
            """
            SELECT symbol, exchange,
                   CAST(to_timestamp(timestamp + 19800) AS DATE) AS date, close
            FROM market_data
            WHERE interval = 'D'
              AND CAST(to_timestamp(timestamp + 19800) AS DATE)
                  >= CURRENT_DATE - INTERVAL 400 DAY
            ORDER BY symbol, exchange, timestamp
            """
        ).fetchdf()
    finally:
        duck.close()

    out: dict[tuple[str, str], pd.DataFrame] = {}
    if frame.empty:
        return out
    for (symbol, exchange), group in frame.groupby(["symbol", "exchange"]):
        if (symbol, exchange) in set(members):
            out[(symbol, exchange)] = group[["date", "close"]].reset_index(drop=True)
    return out


def list_recent(engine, days: int) -> list[dict]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT symbol, exchange, ex_date, action, ratio FROM mp_adjustments "
                    "WHERE ex_date >= :cutoff ORDER BY ex_date DESC, symbol"
                ),
                {"cutoff": (dt.date.today() - dt.timedelta(days=days)).isoformat()},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def run_nightly(engine=None) -> None:
    from database.engine_factory import create_db_engine

    own = engine is None
    engine = engine or create_db_engine()
    try:
        sync_ledger(engine)
        cross_check(engine)
    except Exception:
        logger.exception("corporate-action nightly check failed")
    finally:
        if own:
            engine.dispose()


if __name__ == "__main__":
    import argparse

    from database.engine_factory import create_db_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-recent", type=int, metavar="DAYS", default=None)
    args = parser.parse_args()

    _engine = create_db_engine()
    try:
        if args.list_recent is not None:
            for row in list_recent(_engine, args.list_recent):
                print(
                    f"{row['ex_date']}  {row['symbol']:<14} {row['exchange']}  "
                    f"{row['action']:<9} ratio={row['ratio']}"
                )
        else:
            run_nightly(_engine)
    finally:
        _engine.dispose()
