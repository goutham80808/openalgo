"""Nightly universe refresh. I/O lives here; decisions live in
services/marketpulse/universe.py.

Runs on a REAL OS thread (see utils/real_threading) — never a greenlet —
because it holds a long NSE fetch + SQLite write path open for the whole
universe. Phase 1 hooks this into the post-EOD scheduler; until then it is
invoked from scripts and tests via refresh()/run_nightly().
"""

import datetime as dt
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import bindparam, text

from services.marketpulse import universe
from services.marketpulse.config import adt_floor_cr
from services.marketpulse_jobs import nse_client
from utils.logging import get_logger
from utils.real_threading import Thread

logger = get_logger(__name__)

LOOKBACK_SESSIONS = 20


@dataclass
class RefreshSummary:
    sessions: int
    members: int
    departed: int = 0


def refresh(engine, nse=None, dates: list[str] | None = None) -> RefreshSummary:
    """Recompute mp_universe membership from ~20 sessions of bhavcopy."""
    nse = nse or nse_client
    dates = dates or _last_n_trading_dates(LOOKBACK_SESSIONS)
    frames = []
    for date in dates:
        frame = nse.fetch_nse_bhavcopy_with_delivery(date)
        frames.append(_to_panel(frame, date))
    panel = pd.concat(frames, ignore_index=True)
    rows = universe.compute_membership(panel, floor_cr=adt_floor_cr(), as_of=dates[-1])
    departed = _upsert(engine, rows, as_of=dates[-1])
    _enrich_flags(engine, nse)
    summary = RefreshSummary(sessions=len(dates), members=len(rows), departed=departed)
    logger.info(
        f"universe refresh: {summary.members} members over {summary.sessions} "
        f"sessions, {summary.departed} departed as of {dates[-1]}"
    )
    return summary


def _to_panel(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    """Bhavcopy frame -> the panel shape compute_membership expects.

    The NSE full bhavcopy carries no company name or ISIN; those are
    backfilled from the AMFI join (industry map refresh) and the index
    constituent files. Only the EQ series counts: counting BE/SM rows too
    would double-count symbols and mix trade-to-trade sessions in.
    """
    if "SERIES" in frame.columns:
        frame = frame[frame["SERIES"] == "EQ"]
    out = pd.DataFrame(
        {
            "symbol": frame["SYMBOL"],
            "exchange": "NSE",
            "name": None,
            "isin": None,
            "turnover_cr": frame["turnover_cr"],
            "date": date,
        }
    )
    return out


def _upsert(engine, rows: list[dict], as_of: str) -> int:
    """Upsert members; mark members absent from today's set as departed."""
    stmt = text(
        """
        INSERT INTO mp_universe (symbol, exchange, name, isin, is_fno,
            index_member, delivery_available, active_from, active_to, reasons)
        VALUES (:symbol, :exchange, :name, :isin, :is_fno, :index_member,
            :delivery_available, :active_from, :active_to, :reasons)
        ON CONFLICT(symbol, exchange) DO UPDATE SET
            name=excluded.name, isin=excluded.isin, reasons=excluded.reasons,
            active_to=NULL
        """
    )
    with engine.begin() as conn:
        if rows:
            conn.execute(stmt, rows)
        still_here = {(r["symbol"], r["exchange"]) for r in rows}
        existing = conn.execute(
            text("SELECT symbol, exchange FROM mp_universe WHERE active_to IS NULL")
        ).fetchall()
        departed = [
            {"symbol": s, "exchange": e, "as_of": as_of}
            for s, e in existing
            if (s, e) not in still_here
        ]
        if departed:
            conn.execute(
                text(
                    "UPDATE mp_universe SET active_to=:as_of "
                    "WHERE symbol=:symbol AND exchange=:exchange"
                ),
                departed,
            )
        return len(departed)


def _enrich_flags(engine, nse) -> None:
    """Set is_fno and index_member from the NSE lot file and index lists."""
    try:
        fno_symbols = set(nse.fetch_fno_lot_sizes())
    except Exception:
        logger.exception("is_fno enrichment skipped: lot size fetch failed")
        return
    index_symbols: set[str] = set()
    for slug in nse_client.INDEX_CONSTITUENT_SLUGS:
        try:
            frame = nse.fetch_index_constituents(slug)
            symbol_col = _column(frame, "Symbol")
            if symbol_col:
                index_symbols.update(frame[symbol_col].dropna().astype(str).str.strip())
        except Exception:
            logger.exception(f"index_member enrichment skipped for {slug}")
    with engine.begin() as conn:
        _set_flag(conn, "is_fno", fno_symbols)
        _set_flag(conn, "index_member", index_symbols)


def _column(frame: pd.DataFrame, wanted: str) -> str | None:
    for column in frame.columns:
        if str(column).strip().upper() == wanted.upper():
            return column
    return None


def _set_flag(conn, flag: str, symbols: set[str]) -> None:
    if not symbols:
        return
    names = sorted(symbols)
    stmt = text(f"UPDATE mp_universe SET {flag}=1 WHERE symbol = :symbol AND exchange='NSE'")
    conn.execute(stmt, [{"symbol": s} for s in names])


def _last_n_trading_dates(n: int) -> list[str]:
    from utils.trading_calendar import is_trading_day

    out: list[str] = []
    day = dt.date.today()
    while len(out) < n:
        if is_trading_day(day):
            out.append(day.isoformat())
        day -= dt.timedelta(days=1)
    return list(reversed(out))


def run_nightly() -> None:
    """Spawn refresh() on a real OS thread, fire-and-forget with logging.

    The eventlet hub must never block on the NSE fetch loop, so the caller
    (a scheduler or a blueprint) hands the work to this thread and returns.
    """
    from database.engine_factory import create_db_engine

    def _work() -> None:
        engine = create_db_engine()
        try:
            refresh(engine=engine)
        except Exception:
            logger.exception("nightly universe refresh failed")
        finally:
            engine.dispose()

    Thread(target=_work, name="marketpulse-universe-refresh", daemon=True).start()
