import pandas as pd

from services.marketpulse.industry_map import normalize_amfi


def test_amfi_columns_map_to_our_levels():
    frame = pd.DataFrame(
        {
            "ISIN": ["INE009A01021"],
            "SYMBOL \u00a0": ["INFY"],  # AMFI headers carry odd whitespace
            "MACRO SECTOR": ["Financial Services"],
            "SECTOR": ["IT"],
            "INDUSTRY": ["Computers - Software & Consulting"],
        }
    )
    rows = normalize_amfi(frame, updated_at="2026-10-09T00:00:00")
    assert rows[0] == {
        "isin": "INE009A01021",
        "sector": "Financial Services",
        "industry": "IT",
        "subgroup": "Computers - Software & Consulting",
        "source": "amfi",
        "updated_at": "2026-10-09T00:00:00",
    }


def test_unmapped_isin_is_dropped_not_crashed():
    frame = pd.DataFrame({"ISIN": [None], "SECTOR": ["x"], "INDUSTRY": ["y"]})
    assert normalize_amfi(frame, updated_at="t") == []
