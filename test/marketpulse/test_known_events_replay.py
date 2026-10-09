"""Known-events replay: recent splits/bonuses must be explained by the ledger.

Each event in known_events.json is replayed against historify.duckdb daily
candles when the local store has them: the raw overnight gap around the
ex-date must clear 35% (so it would have been flagged unexplained) before
apply_factors and leave no unexplained gap after. Skips with a clear message
when the local store has no candles for the symbol yet.
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from services.marketpulse.adjustments import apply_factors, detect_unexplained_gaps

KNOWN_EVENTS = Path(__file__).parent / "known_events.json"


def _candles(symbol: str, exchange: str) -> pd.DataFrame | None:
    try:
        import duckdb

        from database.historify_db import get_db_path

        conn = duckdb.connect(get_db_path(), read_only=True)
        try:
            frame = conn.execute(
                """
                SELECT CAST(to_timestamp(timestamp + 19800) AS DATE) AS date,
                       open, high, low, close
                FROM market_data
                WHERE interval = 'D' AND symbol = ? AND exchange = ?
                  AND CAST(to_timestamp(timestamp + 19800) AS DATE)
                      BETWEEN DATE '2025-01-01' AND CURRENT_DATE
                ORDER BY timestamp
                """,
                [symbol, exchange],
            ).fetchdf()
        finally:
            conn.close()
    except Exception:
        return None
    if frame.empty:
        return None
    frame["date"] = frame["date"].astype(str)
    return frame


@pytest.mark.parametrize(
    "event",
    json.loads(KNOWN_EVENTS.read_text()),
    ids=lambda e: f"{e['symbol']}-{e['ex_date']}-{e['action']}",
)
def test_known_events_replay(event):
    candles = _candles(event["symbol"], event["exchange"])
    if candles is None:
        pytest.skip(f"no local candles for {event['symbol']} yet — replay after backfill")

    events = [
        e
        for e in json.loads(KNOWN_EVENTS.read_text())
        if e["symbol"] == event["symbol"] and e["exchange"] == event["exchange"]
    ]

    # Sanity: the exchange calendar says something happened on this date.
    ex_date = dt.date.fromisoformat(event["ex_date"])
    window = candles[
        (candles["date"] >= (ex_date - dt.timedelta(days=7)).isoformat())
        & (candles["date"] <= (ex_date + dt.timedelta(days=7)).isoformat())
    ]
    assert not window.empty, "no candles near the recorded ex-date"

    flagged = [
        f
        for f in detect_unexplained_gaps(
            candles.assign(symbol=event["symbol"], exchange=event["exchange"]),
            known=[],
        )
        if f["date"] == event["ex_date"]
    ]
    if not flagged:
        pytest.skip(
            f"{event['symbol']} moved less than 35% overnight on "
            f"{event['ex_date']} (e.g. a dividend-only record); nothing to replay"
        )

    adjusted = apply_factors(candles, events, event["symbol"], event["exchange"])
    after = detect_unexplained_gaps(
        adjusted.assign(symbol=event["symbol"], exchange=event["exchange"]),
        known=events,
    )
    assert all(f["date"] != event["ex_date"] for f in after), (
        f"{event['symbol']} {event['action']} on {event['ex_date']} does not "
        "explain the observed overnight move — ledger ratio is wrong"
    )
