"""Pure compute: adjustment factors over candle frames.

No I/O. `apply_factors` builds the adjusted series from the ledger;
`detect_unexplained_gaps` finds overnight moves the ledger does not explain —
the nightly cross-check that catches a split the exchange calendar missed.
"""

import pandas as pd

OHLC_COLUMNS = ("open", "high", "low", "close")


def apply_factors(
    candles: pd.DataFrame, factors: list[dict], symbol: str, exchange: str
) -> pd.DataFrame:
    """Scale OHLC (never volume) of rows strictly before each ex-date.

    Cumulative by construction: a row earlier than two ex-dates is multiplied
    by both ratios. Factors outside the candle range are no-ops. Returns a
    new frame; the input is not mutated.
    """
    out = candles.copy()
    relevant = [
        f
        for f in factors
        if f.get("symbol") == symbol
        and f.get("exchange") == exchange
        and f.get("ex_date") is not None
    ]
    if not relevant or out.empty:
        return out
    dates = out["date"].astype(str)
    lo, hi = dates.min(), dates.max()
    for factor in relevant:
        ex_date = str(factor["ex_date"])
        if not (lo <= ex_date <= hi):
            continue
        mask = dates < ex_date
        ratio = float(factor["ratio"])
        for column in out.columns:
            if str(column).strip().lower() in OHLC_COLUMNS:
                out.loc[mask, column] = out.loc[mask, column] * ratio
    return out


def detect_unexplained_gaps(
    candles: pd.DataFrame, known: list[dict], threshold: float = 0.35
) -> list[dict]:
    """Flag overnight moves past `threshold` that no known action explains.

    A gap is explained when some known ex_date falls within one session of
    the gap's date (the ledger row may carry the ex-date a day either side
    of where the candle shows the move). Returns one dict per flag with
    date, symbol, exchange and the observed return.
    """
    if candles.empty or len(candles) < 2:
        return []
    frame = candles.reset_index(drop=True)
    closes = frame["close"].astype(float)
    dates = frame["date"].astype(str)
    known_dates = {str(k.get("ex_date")) for k in known if k.get("ex_date")}

    flags: list[dict] = []
    for i in range(1, len(frame)):
        prev = closes.iloc[i - 1]
        if prev <= 0:
            continue
        change = closes.iloc[i] / prev - 1.0
        if abs(change) <= threshold:
            continue
        window = {dates.iloc[j] for j in (i - 1, i, i + 1) if 0 <= j < len(frame)}
        if window & known_dates:
            continue
        flags.append(
            {
                "date": dates.iloc[i],
                "symbol": frame["symbol"].iloc[i] if "symbol" in frame.columns else None,
                "exchange": frame["exchange"].iloc[i] if "exchange" in frame.columns else None,
                "change": round(float(change), 4),
            }
        )
    return flags
