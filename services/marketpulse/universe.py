"""Pure compute: EOD turnover panels -> universe membership rows.

No I/O. Input is the last ~20 sessions of bhavcopy frames; output is a list
of dict rows shaped for the mp_universe table.
"""

import json

import pandas as pd


def compute_membership(panel: pd.DataFrame, floor_cr: float, as_of: str) -> list[dict]:
    """panel columns: symbol, exchange, name, isin, turnover_cr (one row per
    symbol per session; pass ~20 sessions concatenated)."""
    if panel.empty:
        return []
    grouped = panel.groupby(["symbol", "exchange"], sort=False)
    adt = grouped["turnover_cr"].mean()
    names = grouped["name"].last()
    isins = grouped["isin"].last()
    rows = []
    for (symbol, exchange), adt20 in adt.items():
        if adt20 < floor_cr:
            continue
        rows.append(
            {
                "symbol": symbol,
                "exchange": exchange,
                "name": names.loc[(symbol, exchange)],
                "isin": isins.loc[(symbol, exchange)],
                "is_fno": 0,
                "index_member": 0,
                "delivery_available": 1 if exchange == "NSE" else 0,
                "active_from": as_of,
                "active_to": None,
                "reasons": json.dumps({"adt20": round(float(adt20), 3)}),
            }
        )
    return rows
