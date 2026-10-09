"""Record how far back Kotak serves 1m candles. Answer goes to the handoff
and master spec section 12-B (backfill vs live-record decision).

Uses the platform's own unified API so the answer reflects exactly what
OpenAlgo gets. Needs the app running and a Kotak session logged in:
    uv run scripts/marketpulse/kotak_intraday_spike.py <api_key>
"""

import datetime as dt
import os
import sys

import httpx

BASE = os.environ.get("OPENALGO_URL", "http://127.0.0.1:5000")
SYMBOL = os.environ.get("MARKETPULSE_SPIKE_SYMBOL", "RELIANCE")
STEPS = (7, 30, 90, 180, 365)


def probe(api_key: str, days: int) -> tuple[str, int]:
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    response = httpx.post(
        f"{BASE}/api/v1/history",
        json={
            "apikey": api_key,
            "symbol": SYMBOL,
            "exchange": "NSE",
            "interval": "1m",
            "start_date": start,
            "end_date": dt.date.today().isoformat(),
        },
        timeout=120,
    )
    data = response.json().get("data") or []
    if isinstance(data, list) and data:
        first = data[0]
        stamp = first.get("timestamp") if isinstance(first, dict) else first
        return str(stamp), len(data)
    return "none", 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for days in STEPS:
        stamp, count = probe(sys.argv[1], days)
        print(f"{days}d request -> oldest candle: {stamp} ({count} rows)")
