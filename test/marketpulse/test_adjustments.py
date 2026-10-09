import pandas as pd

from services.marketpulse.adjustments import apply_factors, detect_unexplained_gaps


def test_split_factor_scales_history_before_ex_date():
    candles = pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "close": [2000.0, 1005.0]})
    factors = [
        {
            "symbol": "X",
            "exchange": "NSE",
            "ex_date": "2026-01-05",
            "action": "split",
            "ratio": 0.5,
        }
    ]
    out = apply_factors(candles, factors, symbol="X", exchange="NSE")
    assert out.loc[0, "close"] == 1000.0  # pre-split scaled by 0.5
    assert out.loc[1, "close"] == 1005.0


def test_unexplained_overnight_gap_is_flagged():
    candles = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "close": [100.0, 50.0],
            "symbol": "X",
            "exchange": "NSE",
        }
    )
    flags = detect_unexplained_gaps(candles, known=[{"ex_date": "2026-02-01", "action": "split"}])
    assert len(flags) == 1 and flags[0]["date"] == "2026-01-05"


def test_multiple_factors_are_cumulative():
    candles = pd.DataFrame(
        {"date": ["2026-01-02", "2026-03-02", "2026-05-04"], "close": [1000.0, 500.0, 260.0]}
    )
    factors = [
        {
            "symbol": "X",
            "exchange": "NSE",
            "ex_date": "2026-03-02",
            "action": "split",
            "ratio": 0.5,
        },
        {
            "symbol": "X",
            "exchange": "NSE",
            "ex_date": "2026-05-04",
            "action": "split",
            "ratio": 0.5,
        },
    ]
    out = apply_factors(candles, factors, symbol="X", exchange="NSE")
    assert out.loc[0, "close"] == 250.0  # both splits apply
    assert out.loc[1, "close"] == 250.0  # only the second applies
    assert out.loc[2, "close"] == 260.0


def test_gap_matching_a_known_action_within_one_session_is_not_flagged():
    candles = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "close": [100.0, 50.0, 51.0],
            "symbol": "X",
            "exchange": "NSE",
        }
    )
    flags = detect_unexplained_gaps(candles, known=[{"ex_date": "2026-01-05", "action": "split"}])
    assert flags == []
