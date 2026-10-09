"""Pure compute: AMFI security-master frame -> mp_industry_map rows."""

import pandas as pd


def _find_column(frame: pd.DataFrame, wanted: str) -> str | None:
    for column in frame.columns:
        if str(column).strip().upper() == wanted:
            return column
    return None


def normalize_amfi(frame: pd.DataFrame, updated_at: str) -> list[dict]:
    isin_col = _find_column(frame, "ISIN")
    macro_col = _find_column(frame, "MACRO SECTOR")
    sector_col = _find_column(frame, "SECTOR")
    industry_col = _find_column(frame, "INDUSTRY")
    if not all([isin_col, sector_col, industry_col]):
        return []
    rows = []
    for _, record in frame.iterrows():
        isin = record[isin_col]
        if not isinstance(isin, str) or not isin.strip():
            continue
        rows.append(
            {
                "isin": isin.strip(),
                "sector": str(record[macro_col]).strip() if macro_col else None,
                "industry": str(record[sector_col]).strip(),
                "subgroup": str(record[industry_col]).strip(),
                "source": "amfi",
                "updated_at": updated_at,
            }
        )
    return rows
