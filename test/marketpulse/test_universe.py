import pandas as pd

from services.marketpulse.universe import compute_membership


def _panel(turnover_by_symbol):
    return pd.DataFrame(
        {
            "symbol": list(turnover_by_symbol),
            "exchange": ["NSE"] * len(turnover_by_symbol),
            "name": ["x"] * len(turnover_by_symbol),
            "isin": [None] * len(turnover_by_symbol),
            "turnover_cr": list(turnover_by_symbol.values()),
        }
    )


def test_liquid_names_pass_the_adt_floor():
    rows = compute_membership(
        _panel({"INFY": 120.0, "DULL": 0.4}), floor_cr=1.0, as_of="2026-10-09"
    )
    symbols = {(r["symbol"], r["exchange"]) for r in rows}
    assert ("INFY", "NSE") in symbols
    assert ("DULL", "NSE") not in symbols


def test_membership_records_reasons_and_dates():
    rows = compute_membership(_panel({"INFY": 120.0}), floor_cr=1.0, as_of="2026-10-09")
    assert rows[0]["active_from"] == "2026-10-09"
    assert rows[0]["active_to"] is None
    assert "adt20" in rows[0]["reasons"]
