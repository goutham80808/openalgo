"""Thin wrapper over the NSE data fetches Market Pulse needs.

Every NSE fetch in Market Pulse goes through here, so an upstream break is
one file to fix. Library pick (Phase 0 spike, 2026-10-10):

- nsepython 2.101 for the JSON-API endpoints (participant-wise F&O OI,
  FII/DII flows, historical bulk/block deals, per-symbol corporate actions).
- The whole-market bhavcopy with delivery (sec_bhavdata_full) comes from the
  NSE archives host via a plain GET — a stable public URL that needs no
  cookies. nsepython's own get_bhavcopy is broken in 2.101, and this is the
  same GET jugaad-data's full_bhavcopy_raw performs.
- Known gap: BSE deliverable bhavcopy. bseindia.com's download host serves
  its SPA page to plain requests; neither candidate library covers it. The
  gap is documented in the Phase 0 handoff and revisited when BSE delivery
  data becomes a requirement.

Pacing: sleeps MARKETPULSE_NSE_MIN_INTERVAL (default 0.5s) between network
calls. Callers run on real OS threads (see utils/real_threading).

All dates crossing this module's public API are ISO (YYYY-MM-DD); the
DD-MM-YYYY forms NSE expects never leave this file.
"""

import datetime as dt
import io
import os
import time

import pandas as pd

from utils.httpx_client import get_httpx_client
from utils.logging import get_logger
from utils.real_threading import Lock

logger = get_logger(__name__)

BHAVCOPY_FULL_URL = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)

_MIN_INTERVAL = float(os.environ.get("MARKETPULSE_NSE_MIN_INTERVAL", "0.5"))
_last_call = 0.0
_pace_lock = Lock()


def _paced() -> None:
    global _last_call
    with _pace_lock:
        wait = _last_call + _MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _ddmmyyyy(iso_date: str) -> str:
    return dt.date.fromisoformat(iso_date).strftime("%d-%m-%Y")


def fetch_nse_bhavcopy_with_delivery(date: str) -> pd.DataFrame:
    """Whole-market NSE sec_bhavcopy with delivery for `date` (YYYY-MM-DD).

    Returns the raw frame with column names whitespace-stripped (the file's
    headers are padded) plus a computed turnover_cr column, one row per
    symbol-series.
    """
    _paced()
    stamp = dt.date.fromisoformat(date).strftime("%d%m%Y")
    url = BHAVCOPY_FULL_URL.format(ddmmyyyy=stamp)
    try:
        response = get_httpx_client().get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=30,
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
    except Exception:
        logger.exception(f"NSE bhavcopy fetch failed for {date}")
        raise
    frame.columns = [str(c).strip() for c in frame.columns]
    if "TURNOVER_LACS" in frame.columns:
        frame["turnover_cr"] = frame["TURNOVER_LACS"] / 100.0
    return frame


def fetch_participant_oi(date: str) -> pd.DataFrame:
    """Participant-wise F&O open interest for `date` (YYYY-MM-DD)."""
    _paced()
    try:
        import nsepython

        return nsepython.get_fao_participant_oi(_ddmmyyyy(date))
    except Exception:
        logger.exception(f"participant OI fetch failed for {date}")
        raise


def fetch_fii_dii_flows() -> pd.DataFrame:
    """Recent FII/DII cash-market buy/sell flows (latest days NSE serves)."""
    _paced()
    try:
        import nsepython

        return nsepython.nse_fiidii()
    except Exception:
        logger.exception("FII/DII flow fetch failed")
        raise


def fetch_bulk_deals(from_date: str, to_date: str) -> pd.DataFrame:
    """Bulk deals between two ISO dates (inclusive)."""
    return _largedeals("bulk_deals", from_date, to_date)


def fetch_block_deals(from_date: str, to_date: str) -> pd.DataFrame:
    """Block deals between two ISO dates (inclusive)."""
    return _largedeals("block_deals", from_date, to_date)


def _largedeals(mode: str, from_date: str, to_date: str) -> pd.DataFrame:
    _paced()
    try:
        import nsepython

        return nsepython.nse_largedeals_historical(
            _ddmmyyyy(from_date), _ddmmyyyy(to_date), mode=mode
        )
    except Exception:
        logger.exception(f"{mode} fetch failed for {from_date}..{to_date}")
        raise


def fetch_corporate_actions(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Exchange corporate-actions calendar rows for one symbol (ISO dates)."""
    _paced()
    try:
        import nsepython

        return nsepython.nse_mcp_get_corporate_actions(
            symbol, _ddmmyyyy(from_date), _ddmmyyyy(to_date)
        )
    except Exception:
        logger.exception(f"corporate actions fetch failed for {symbol}")
        raise
