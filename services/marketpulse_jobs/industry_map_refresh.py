"""AMFI industry map loader. I/O lives here; normalization lives in
services/marketpulse/industry_map.py.

Downloads the AMFI security classification workbook, normalizes it into
mp_industry_map rows keyed on ISIN, and backfills name/ISIN onto mp_universe
members from the workbook's symbol column (the NSE bhavcopy carries neither).

The workbook URL is a module constant overridable with MARKETPULSE_AMFI_URL.
The AMFI site moved its downloads behind a JS application during its 2026
redesign and the historical spages URL is no longer served to plain HTTP
clients — see the Phase 0 handoff. The loader is URL-agnostic: pointing
MARKETPULSE_AMFI_URL at wherever the workbook currently lives is a config
change, not a code change. Both .xlsx (openpyxl) and .xls (xlrd) parse.
"""

import datetime as dt
import io
import os
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from services.marketpulse import industry_map
from utils.httpx_client import get_httpx_client
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_AMFI_URL = "https://www.amfiindia.com/spages/AMFI_Sheet.xlsx"


@dataclass
class IndustryMapSummary:
    rows: int
    unmapped_isins: int


def amfi_url() -> str:
    return os.environ.get("MARKETPULSE_AMFI_URL", DEFAULT_AMFI_URL)


def _download() -> pd.DataFrame:
    url = amfi_url()
    response = get_httpx_client().get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=60,
        follow_redirects=True,
    )
    response.raise_for_status()
    return pd.read_excel(io.BytesIO(response.content))


def refresh(engine, fetch=None) -> IndustryMapSummary:
    """Upsert the AMFI classification; backfill universe identity.

    fetch(): returns the workbook as a DataFrame. Injectable for tests.
    """
    fetch = fetch or _download
    frame = fetch()
    updated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    rows = industry_map.normalize_amfi(frame, updated_at=updated_at)

    total = len(frame)
    unmapped = total - len(rows)
    _upsert_map(engine, rows)
    _backfill_universe_identity(engine, frame)
    summary = IndustryMapSummary(rows=len(rows), unmapped_isins=unmapped)
    logger.info(
        f"industry map refresh: {summary.rows} classified, "
        f"{summary.unmapped_isins} without a usable ISIN"
    )
    return summary


def _upsert_map(engine, rows: list[dict]) -> None:
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO mp_industry_map (isin, sector, industry, subgroup, source, updated_at)
        VALUES (:isin, :sector, :industry, :subgroup, :source, :updated_at)
        ON CONFLICT(isin) DO UPDATE SET
            sector=excluded.sector, industry=excluded.industry,
            subgroup=excluded.subgroup, source=excluded.source,
            updated_at=excluded.updated_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def _backfill_universe_identity(engine, frame: pd.DataFrame) -> None:
    """Fill name/ISIN on mp_universe rows the bhavcopy left NULL.

    The AMFI workbook's SYMBOL column is the OpenAlgo-style NSE symbol, so
    the join is on symbol; only NULL identity fields are touched, never a
    value another source already wrote.
    """
    symbol_col = _column(frame, "SYMBOL")
    isin_col = _column(frame, "ISIN")
    name_col = _column(frame, "COMPANY NAME") or _column(frame, "NAME OF THE COMPANY")
    if not (symbol_col and isin_col):
        return
    updates = []
    for _, record in frame.iterrows():
        symbol = record[symbol_col]
        isin = record[isin_col]
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        if not isinstance(isin, str) or not isin.strip():
            continue
        row: dict = {"symbol": symbol.strip(), "isin": isin.strip()}
        if name_col:
            name = record[name_col]
            row["name"] = name.strip() if isinstance(name, str) else None
        else:
            row["name"] = None
        updates.append(row)
    if not updates:
        return
    stmt = text(
        """
        UPDATE mp_universe SET
            isin=:isin,
            name=COALESCE(:name, name)
        WHERE symbol=:symbol AND isin IS NULL
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, updates)


def _column(frame: pd.DataFrame, wanted: str) -> str | None:
    for column in frame.columns:
        if str(column).strip().upper() == wanted:
            return column
    return None
