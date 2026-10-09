# Market Pulse Phase 0 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the data foundation: universe + industry map + adjustment ledger tables, nightly universe refresh on a real thread, Historify EOD schedules for the whole liquid universe, 2-year backfill, corporate-actions verification, and the two spikes (Kotak intraday depth, NSE library pick).

**Architecture:** Pure-compute modules in `services/marketpulse/` (no I/O); I/O jobs in `services/marketpulse_jobs/` running on real OS threads via `utils/real_threading`; SQLite writes via `database.engine_factory.create_db_engine()`; NSE public files fetched through a thin wrapper around the community library picked in Task 3. Master design: `docs/superpowers/specs/2026-10-09-marketpulse-design.md` (§3, §4, §8 Phase 0).

**Tech Stack:** Python 3.12, uv, pandas, SQLAlchemy (NullPool), APScheduler (existing Historify scheduler), NSE community library (jugaad-data or nsepython — Task 3 decides).

**Conventions:** Read `docs/superpowers/specs/marketpulse/README.md` first. uv only. `get_logger(__name__)`. `logger.exception()` on errors. No emojis. Conventional Commits. Run `uv run ruff check . --fix && uv run ruff format .` before each commit.

---

### Task 1: Migration — the four foundation tables

**Files:**
- Create: `upgrade/migrate_marketpulse.py`
- Modify: `upgrade/migrate_all.py` (MIGRATIONS list)
- Test: `test/marketpulse/test_migrate_marketpulse.py`

- [ ] **Step 1: Write the failing test**

```python
# test/marketpulse/test_migrate_marketpulse.py
"""Migration idempotency: applied twice, tables identical, --status quiet."""
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "upgrade" / "migrate_marketpulse.py"


def _tables(db):
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mp_%'"
    ).fetchall()
    return sorted(r[0] for r in rows)


def test_creates_tables_and_is_idempotent(tmp_path):
    db_path = tmp_path / "openalgo.db"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA foreign_keys=ON")
    db.commit()
    db.close()
    env = {"DATABASE_PATH": str(db_path)}
    import os

    full_env = {**os.environ, **env}
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], env=full_env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    db = sqlite3.connect(db_path)
    assert _tables(db) == [
        "mp_adjustments",
        "mp_events",
        "mp_industry_map",
        "mp_universe",
    ]
    db.close()


def test_status_reports_without_applying(tmp_path):
    db_path = tmp_path / "openalgo.db"
    import os

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--status"],
        env={**os.environ, "DATABASE_PATH": str(db_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mp_universe" in result.stdout
    db = sqlite3.connect(db_path)
    assert _tables(db) == []
    db.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest test/marketpulse/test_migrate_marketpulse.py -v`
Expected: FAIL — `upgrade/migrate_marketpulse.py` does not exist.

- [ ] **Step 3: Write the migration**

```python
#!/usr/bin/env python3
"""Market Pulse foundation tables: universe, industry map, adjustments, events.

Idempotent: checks PRAGMA table_info and returns quietly when present.
Supports --status to report without applying. Tested on fresh and copied
real databases (see test/marketpulse/test_migrate_marketpulse.py).
"""

import argparse
import os
import sqlite3
import sys

TABLES = {
    "mp_universe": """
        CREATE TABLE IF NOT EXISTS mp_universe (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            name TEXT,
            isin TEXT,
            is_fno INTEGER NOT NULL DEFAULT 0,
            index_member INTEGER NOT NULL DEFAULT 0,
            delivery_available INTEGER NOT NULL DEFAULT 0,
            active_from TEXT NOT NULL,
            active_to TEXT,
            reasons TEXT,
            PRIMARY KEY (symbol, exchange)
        )""",
    "mp_industry_map": """
        CREATE TABLE IF NOT EXISTS mp_industry_map (
            isin TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            subgroup TEXT,
            source TEXT NOT NULL DEFAULT 'amfi',
            updated_at TEXT NOT NULL
        )""",
    "mp_adjustments": """
        CREATE TABLE IF NOT EXISTS mp_adjustments (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            ex_date TEXT NOT NULL,
            action TEXT NOT NULL,
            ratio REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'nse',
            PRIMARY KEY (symbol, exchange, ex_date, action)
        )""",
    "mp_events": """
        CREATE TABLE IF NOT EXISTS mp_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            date TEXT NOT NULL,
            kind TEXT NOT NULL,
            detail TEXT
        )""",
}
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mp_events_symbol_date ON mp_events (symbol, date)",
]


def _connect():
    path = os.environ.get("DATABASE_PATH", os.path.join("database", "openalgo.db"))
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def main(apply_changes: bool) -> int:
    conn = _connect()
    try:
        for name, ddl in TABLES.items():
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [name]
            ).fetchone()
            state = "present" if exists else "missing"
            print(f"  mp table {name}: {state}")
            if apply_changes and not exists:
                conn.execute(ddl)
                print(f"  [OK] created {name}")
        if apply_changes:
            for ddl in INDEXES:
                conn.execute(ddl)
            conn.commit()
            print("[OK] Market Pulse foundation migration complete")
        else:
            print("[STATUS] report only — nothing applied")
        return 0
    except Exception:
        import logging

        logging.exception("marketpulse migration failed")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    sys.exit(main(apply_changes=not args.status))
```

- [ ] **Step 4: Register in migrate_all.py**

In `upgrade/migrate_all.py`, append inside `MIGRATIONS` (after the alert-log entry):

```python
    ("migrate_marketpulse.py", "Market Pulse Foundation Tables"),
```

- [ ] **Step 5: Run tests + verify on a copy of the real DB**

Run: `uv run pytest test/marketpulse/test_migrate_marketpulse.py -v`
Expected: 2 PASS.
Run: `cp database/openalgo.db /tmp/opencode/openalgo-copy.db && DATABASE_PATH=/tmp/opencode/openalgo-copy.db uv run upgrade/migrate_marketpulse.py --status && DATABASE_PATH=/tmp/opencode/openalgo-copy.db uv run upgrade/migrate_marketpulse.py`
Expected: creates the four tables on the populated copy without touching any non-mp table.

- [ ] **Step 6: Commit**

```bash
git add upgrade/migrate_marketpulse.py upgrade/migrate_all.py test/marketpulse/
git commit -m "feat: marketpulse foundation tables migration"
```

---

### Task 2: Pure-compute package — universe + industry map + config

**Files:**
- Create: `services/marketpulse/__init__.py` (empty), `services/marketpulse/config.py`, `services/marketpulse/universe.py`, `services/marketpulse/industry_map.py`
- Test: `test/marketpulse/test_universe.py`, `test/marketpulse/test_industry_map.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/marketpulse/test_universe.py
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
    rows = compute_membership({"INFY": 120.0, "DULL": 0.4}, floor_cr=1.0)
    symbols = {(r["symbol"], r["exchange"]) for r in rows}
    assert ("INFY", "NSE") in symbols
    assert ("DULL", "NSE") not in symbols


def test_membership_records_reasons_and_dates():
    rows = compute_membership({"INFY": 120.0}, floor_cr=1.0, as_of="2026-10-09")
    assert rows[0]["active_from"] == "2026-10-09"
    assert rows[0]["active_to"] is None
    assert "adt20" in rows[0]["reasons"]
```

```python
# test/marketpulse/test_industry_map.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/marketpulse/test_universe.py test/marketpulse/test_industry_map.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# services/marketpulse/config.py
"""Market Pulse configuration. Every threshold lives here or in .env."""
import os


def adt_floor_cr() -> float:
    """Minimum 20-day average daily traded value, in crore rupees."""
    return float(os.environ.get("MARKETPULSE_ADT_FLOOR_CR", "1.0"))
```

```python
# services/marketpulse/universe.py
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
```

```python
# services/marketpulse/industry_map.py
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest test/marketpulse/test_universe.py test/marketpulse/test_industry_map.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/marketpulse/ test/marketpulse/test_universe.py test/marketpulse/test_industry_map.py
git commit -m "feat: marketpulse pure-compute universe and industry map"
```

---

### Task 3: NSE library spike + thin fetch wrapper

**Files:**
- Create: `services/marketpulse_jobs/__init__.py` (empty), `services/marketpulse_jobs/nse_client.py`, `scripts/marketpulse/nse_bakeoff.py`
- Modify: `pyproject.toml` (add the winning library via `uv add`)

- [ ] **Step 1: Write the bake-off script and run it for both candidates**

```python
# scripts/marketpulse/nse_bakeoff.py
"""One-shot bake-off: can the library fetch the five file classes we need?

Run for each candidate (after `uv add <lib>`):
    uv run python scripts/marketpulse/nse_bakeoff.py <lib>
Records PASS/FAIL per capability. The winner goes into the Phase 0 handoff.
"""
import sys


def main(lib: str) -> int:
    results = {}
    try:
        if lib == "jugaad-data":
            from jugaad_data.nse import full_bhavcopy_download  # type: ignore

            results["bhavcopy NSE"] = callable(full_bhavcopy_download)
        elif lib == "nsepython":
            import nsepython  # type: ignore

            results["bhavcopy NSE"] = callable(
                getattr(nsepython, "bhav_copy", None)
            )
        else:
            print(f"unknown lib {lib}")
            return 2
    except Exception as exc:  # noqa: BLE001 - spike reports, never crashes
        results["import"] = f"FAILED: {exc}"
    for capability, ok in results.items():
        print(f"{capability}: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

Extend the script during the session to check: NSE bhavcopy with delivery, BSE deliverable bhavcopy, participant-wise F&O OI, FII/DII flows, block/bulk deals. For each capability call the real function for one recent trading day and print row counts. Keep every network call in this script — it is a spike, not product code.

- [ ] **Step 2: Pick the winner and record it**

Criteria: covers all five file classes (or the most, with the gap documented), maintained in the last 6 months, works without an NSE cookie dance. Record the pick + per-capability coverage in the Session handoff below and in the master spec §12-B area if it changes assumptions.

- [ ] **Step 3: Write the thin wrapper**

```python
# services/marketpulse_jobs/nse_client.py
"""Thin wrapper over the chosen NSE community library.

Every NSE fetch in Market Pulse goes through here, so an upstream break is
one file to fix. Pacing: sleep MARKETPULSE_NSE_MIN_INTERVAL (default 0.5s)
between file downloads.
"""
import time

from utils.logging import get_logger

logger = get_logger(__name__)

_MIN_INTERVAL = 0.5
_last_call = 0.0


def _paced():
    global _last_call
    wait = _last_call + _MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def fetch_nse_bhavcopy_with_delivery(date: str):
    """Return a pandas DataFrame for NSE sec_bhavdata_full for `date`
    (YYYY-MM-DD). Implementation binds the winning library from Task 3."""
    _paced()
    raise NotImplementedError("bound to the winning library in-session")
```

Bind each fetch function to the winning library during the session; every function follows the same shape (pace, fetch, normalize to a DataFrame, return; `logger.exception` + raise on failure). BSE deliverable bhavcopy, participant OI, FII/DII, deals get one function each — five in total, all named `fetch_*`.

- [ ] **Step 4: Smoke the wrapper**

Run: `uv run python -c "from services.marketpulse_jobs.nse_client import fetch_nse_bhavcopy_with_delivery as f; d=f('2026-10-07'); print(d.shape)"`
Expected: a row count in the thousands (an NSE trading day).

- [ ] **Step 5: Commit**

```bash
git add services/marketpulse_jobs/ scripts/marketpulse/ pyproject.toml uv.lock
git commit -m "feat: marketpulse nse fetch wrapper (library pick from spike)"
```

---

### Task 4: Nightly universe refresh job (real thread)

**Files:**
- Create: `services/marketpulse_jobs/universe_refresh.py`
- Test: `test/marketpulse/test_universe_refresh.py`

- [ ] **Step 1: Write the failing test**

```python
# test/marketpulse/test_universe_refresh.py
import pandas as pd

from services.marketpulse_jobs.universe_refresh import refresh


class FakeNse:
    def __init__(self, frame):
        self._frame = frame
        self.calls = []

    def fetch_nse_bhavcopy_with_delivery(self, date):
        self.calls.append(date)
        return self._frame.copy()


def test_refresh_writes_membership(tmp_path, monkeypatch):
    from database.engine_factory import create_db_engine
    from sqlalchemy import text

    db_url = f"sqlite:///{tmp_path}/openalgo.db"
    engine = create_db_engine(db_url)
    from upgrade.migrate_marketpulse import TABLES

    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        conn.commit()

    panel = pd.DataFrame(
        {
            "symbol": ["INFY", "INFY"],
            "exchange": ["NSE", "NSE"],
            "name": ["Infosys", "Infosys"],
            "isin": ["INE009A01021", "INE009A01021"],
            "turnover_cr": [110.0, 130.0],
        }
    )
    fake = FakeNse(panel)
    summary = refresh(engine=engine, nse=fake, dates=["2026-10-08", "2026-10-09"])
    assert summary.members == 1
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM mp_universe")).scalar()
    assert count == 1
    assert len(fake.calls) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/marketpulse/test_universe_refresh.py -v` — expected FAIL (no module).

- [ ] **Step 3: Implement the job**

```python
# services/marketpulse_jobs/universe_refresh.py
"""Nightly universe refresh. I/O lives here; decisions live in
services/marketpulse/universe.py.

Runs on a REAL OS thread (see utils/real_threading) — never a greenlet —
because it holds a DuckDB/SQLite write path open for the whole universe.
"""
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from services.marketpulse import universe
from services.marketpulse_jobs import nse_client
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefreshSummary:
    sessions: int
    members: int


def refresh(engine, nse=None, dates: list[str] | None = None) -> RefreshSummary:
    nse = nse or nse_client
    dates = dates or _last_n_trading_dates(20)
    frames = []
    for date in dates:
        frame = nse.fetch_nse_bhavcopy_with_delivery(date)
        frames.append(
            frame.assign(
                turnover_cr=frame["TURNOVER_LAKHS"] / 100.0
                if "TURNOVER_LAKHS" in frame.columns
                else frame["turnover_cr"]
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    from services.marketpulse.config import adt_floor_cr

    rows = universe.compute_membership(
        panel, floor_cr=adt_floor_cr(), as_of=dates[-1]
    )
    _upsert(engine, rows)
    return RefreshSummary(sessions=len(dates), members=len(rows))


def _upsert(engine, rows: list[dict]) -> None:
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO mp_universe (symbol, exchange, name, isin, is_fno,
            index_member, delivery_available, active_from, active_to, reasons)
        VALUES (:symbol, :exchange, :name, :isin, :is_fno, :index_member,
            :delivery_available, :active_from, :active_to, :reasons)
        ON CONFLICT(symbol, exchange) DO UPDATE SET
            name=excluded.name, isin=excluded.isin, reasons=excluded.reasons
        """
    )
    with engine.begin() as conn:
        for row in rows:
            conn.execute(stmt, row)


def _last_n_trading_dates(n: int) -> list[str]:
    import datetime as dt

    from utils.market_calendar import is_trading_day  # ground exact helper in-session

    out: list[str] = []
    day = dt.date.today()
    while len(out) < n:
        if is_trading_day(day):
            out.append(day.isoformat())
        day -= dt.timedelta(days=1)
    return list(reversed(out))
```

Ground in-session: the trading-calendar helper (grep `def is_trading_day` or the holiday table used by `migrate_market_holidays.py`) and the bhavcopy column name for traded value; adjust the two named places. The is_fno / index_member enrichment: after `_upsert`, fetch the NSE F&O lot file and index constituents through `nse_client`, and update the flags in one `UPDATE ... FROM` pass — add as `_enrich_flags(engine, nse)` with its own test.

- [ ] **Step 4: Run test**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/marketpulse_jobs/universe_refresh.py test/marketpulse/test_universe_refresh.py
git commit -m "feat: marketpulse nightly universe refresh job"
```

---

### Task 5: AMFI industry map loader

**Files:**
- Create: `services/marketpulse_jobs/industry_map_refresh.py`
- Test: `test/marketpulse/test_industry_map_refresh.py`

Same shape as Task 4, one page: download the AMFI security master XLSX (URL in module constant, configurable via `MARKETPULSE_AMFI_URL`), `pd.read_excel`, `industry_map.normalize_amfi`, upsert into `mp_industry_map` keyed on isin. Test with a fake fetcher returning a synthetic frame; assert unmapped ISINs counted in the summary and no crash. Commit: `feat: marketpulse amfi industry map loader`.

---

### Task 6: Historify wiring — watchlist, schedule, backfill

**Files:**
- Create: `scripts/marketpulse/historify_wireup.py`

- [ ] **Step 1: Ground the watchlist API**

Run: `grep -n "watchlist" blueprints/historify.py database/historify_db.py | head -40`
Identify the function that bulk-adds symbols to the Historify watchlist (schedules download from the watchlist — see `create_schedule(download_source="watchlist")` in database/historify_db.py:3235). Note exact names for the handoff.

- [ ] **Step 2: Write the wire-up script**

Reads active members from `mp_universe`, maps OpenAlgo symbols to broker symbols via `SymToken` (broker/*/database/master_contract_db.py — 18 callers show the pattern; use `instruments_service` if it exposes a lookup), adds them to the Historify watchlist in batches of 200, then creates the daily schedule:

```python
# scripts/marketpulse/historify_wireup.py (core call, after grounding)
from database.historify_db import create_schedule

create_schedule(
    schedule_id="mp-eod",
    name="Market Pulse EOD universe",
    schedule_type="daily",
    data_interval="D",
    time_of_day="17:30",
    download_source="watchlist",
    lookback_days=5,
    description="Market Pulse nightly candle pull (self-healing 5-day lookback)",
)
```

Time-of-day note: 17:30 IST is after NSE EOD files settle; the marketpulse scan itself does not run on this cron — Phase 1 hooks `historify_schedule_execution_complete`.

- [ ] **Step 3: Backfill two years**

Historify caps `lookback_days` at 365 (blueprints/historify.py:1310). Use the manual download endpoint (find the exact route in blueprints/historify.py — the download/preview API) to pull the universe in quarterly windows over ~24 months, paced. Record total bars and wall time in the handoff. If the manual endpoint also caps at 365 days per call, quarterly windows already satisfy it.

- [ ] **Step 4: Coverage check (this is the acceptance gate)**

```python
# scripts/marketpulse/coverage_report.py
"""Report per-symbol candle coverage in historify.duckdb vs mp_universe."""
```

Open `historify.duckdb` read-only (duckdb.connect(path, read_only=True)), count distinct trading dates per symbol for the last 250 sessions, join against active `mp_universe`, print the percent of members with >= 95% coverage. Acceptance: >= 95% of members.

- [ ] **Step 5: Commit**

```bash
git add scripts/marketpulse/
git commit -m "feat: marketpulse historify wire-up and coverage report"
```

---

### Task 7: Corporate actions — ledger, nightly cross-check, known-events suite

**Files:**
- Create: `services/marketpulse/adjustments.py` (pure), `services/marketpulse_jobs/ca_check.py`, `test/marketpulse/known_events.json`, `test/marketpulse/test_adjustments.py`

- [ ] **Step 1: Failing tests (pure part)**

```python
# test/marketpulse/test_adjustments.py
import pandas as pd

from services.marketpulse.adjustments import apply_factors, detect_unexplained_gaps


def test_split_factor_scales_history_before_ex_date():
    candles = pd.DataFrame(
        {"date": ["2026-01-02", "2026-01-05"], "close": [2000.0, 1005.0]}
    )
    factors = [{"symbol": "X", "exchange": "NSE", "ex_date": "2026-01-05",
                "action": "split", "ratio": 0.5}]
    out = apply_factors(candles, factors, symbol="X", exchange="NSE")
    assert out.loc[0, "close"] == 1000.0  # pre-split scaled by 0.5
    assert out.loc[1, "close"] == 1005.0


def test_unexplained_overnight_gap_is_flagged():
    candles = pd.DataFrame(
        {"date": ["2026-01-02", "2026-01-05"], "close": [100.0, 50.0],
         "symbol": "X", "exchange": "NSE"}
    )
    flags = detect_unexplained_gaps(candles, known=[{"ex_date": "2026-02-01",
                                                     "action": "split"}])
    assert len(flags) == 1 and flags[0]["date"] == "2026-01-05"
```

- [ ] **Step 2: Implement `services/marketpulse/adjustments.py`**

`apply_factors(candles, factors, symbol, exchange)`: for each factor with `ex_date` inside the candle range, multiply OHLC columns (not volume) of strictly-earlier rows by `ratio`; cumulative when multiple actions. `detect_unexplained_gaps(candles, known)`: any overnight |return| > 35% whose date matches no known action within ±1 session is a flag (bonus/split-size move without a ledger row).

- [ ] **Step 3: Nightly cross-check job**

`services/marketpulse_jobs/ca_check.py`: fetch the exchange CA calendar via `nse_client` for the last 30 sessions, upsert new rows into `mp_adjustments`, then run `detect_unexplained_gaps` across all `mp_universe` members from historify.duckdb; print and log the flag count; flags > 0 write a trader-readable warning ("N stocks moved overnight in a way their adjustment history does not explain") via `logger.warning`, never a crash.

- [ ] **Step 4: Populate the known-events suite**

Run `uv run python -m services.marketpulse_jobs.ca_check --list-recent 90` (implement the flag in this task), pick 10 well-known corporate actions from the output (prefer splits/bonuses of large names), and write them to `test/marketpulse/known_events.json` as `[{"symbol", "exchange", "ex_date", "action", "ratio"}]`. Add `test_known_events_replay()` that replays each event against historify.duckdb candles when present and asserts the gap disappears after `apply_factors`; skip with a clear message when the local store has no candles for that symbol yet.

- [ ] **Step 5: Run + commit**

Run: `uv run pytest test/marketpulse/ -v` — all PASS.

```bash
git add services/marketpulse/adjustments.py services/marketpulse_jobs/ca_check.py test/marketpulse/
git commit -m "feat: marketpulse corporate-action ledger and cross-check"
```

---

### Task 8: Kotak intraday depth spike

**Files:**
- Create: `scripts/marketpulse/kotak_intraday_spike.py`

- [ ] **Step 1: Write the spike**

Uses the platform's own unified API (generate an API key at `/apikey` if none): call `/api/v1/history` on one liquid symbol with `interval=1m` and `start_date` stepping back 7d / 30d / 90d / 180d / 365d; print the oldest candle timestamp returned at each step.

```python
# scripts/marketpulse/kotak_intraday_spike.py
"""Record how far back Kotak serves 1m candles. Answer goes to the handoff
and master spec §12-B (backfill vs live-record decision)."""
import os
import sys

import httpx

BASE = os.environ.get("OPENALGO_URL", "http://127.0.0.1:5000")


def probe(api_key: str, days: int) -> str:
    import datetime as dt

    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    response = httpx.post(
        f"{BASE}/api/v1/history",
        json={
            "apikey": api_key,
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "interval": "1m",
            "start_date": start,
            "end_date": dt.date.today().isoformat(),
        },
        timeout=60,
    )
    data = response.json().get("data", [])
    return data[0]["timestamp"] if data else "none"


if __name__ == "__main__":
    for days in (7, 30, 90, 180, 365):
        print(f"{days}d request -> oldest candle: {probe(sys.argv[1], days)}")
```

- [ ] **Step 2: Run with the server up and record the answer**

Run: `uv run python scripts/marketpulse/kotak_intraday_spike.py <api_key>`
Write the result into the Session handoff and master spec §12-B.

- [ ] **Step 3: Commit**

```bash
git add scripts/marketpulse/kotak_intraday_spike.py
git commit -m "chore: marketpulse kotak intraday depth spike"
```

---

### Task 9: Close-out

- [ ] fd-audit skill over the diff (DB engines, threads, files).
- [ ] `uv run ruff check . --fix && uv run ruff format .`
- [ ] `uv run pytest test/marketpulse/ -v` green; full suite `uv run pytest test/ -q` no new failures.
- [ ] Fill "Session handoff" below; update the README status table.
- [ ] Ask the user whether to commit any remaining changes.

---

## Session handoff

Executed 2026-10-10 on branch `feat/marketpulse-phase-0` (10 commits).

- Completed: Tasks 1-8 in full. Migration registered in `migrate_all.py` and
  verified idempotent + `--status`-quiet on a copy of the real `db/openalgo.db`;
  pure-compute package (`services/marketpulse/`); NSE wrapper
  (`services/marketpulse_jobs/nse_client.py`, 9 fetch functions, paced);
  universe refresh (writes membership, marks departures inactive, enriches
  `is_fno`/`index_member`); AMFI loader (upserts map + backfills universe
  name/ISIN); Historify wire-up + coverage report scripts; corporate-action
  ledger + nightly cross-check + known-events suite; Kotak spike script.
  Live-smoked against real NSE data: 1674 universe members from 3 sessions
  (design expects ~1,600-1,900), 213 F&O, 494 index members; 20+ price-relevant
  CA rows synced over 120 days (HDFCBANK bonus, ADANIPOWER split, etc.).
  Test suite: 16 passed, 11 known-event replays skipped (no local candles);
  full-repo run has 39 non-marketpulse failures, all verified pre-existing
  on main or flaky-in-isolation-pass.
- Deviations from the master spec (and why):
  - Migration reads `DATABASE_URL` (repo convention, default
    `sqlite:///db/openalgo.db` resolved against project root) instead of the
    plan's `DATABASE_PATH`/`database/openalgo.db` — the plan's default pointed
    at a database the app never opens.
  - Bhavcopy turnover column is `TURNOVER_LACS` (plan guessed
    `TURNOVER_LAKHS`); the wrapper computes `turnover_cr = TURNOVER_LACS/100`.
    Archives CSV headers AND values are space-padded — the reader uses
    `skipinitialspace=True` or the EQ filter silently drops every row.
  - `is_trading_day` lives at `utils/trading_calendar.py` (plan said
    `utils.market_calendar`).
  - NSE bhavcopy carries no name/ISIN: universe rows get NULL identity at
    refresh, backfilled by the AMFI loader's symbol join (only NULL fields
    are touched).
  - Universe refresh filters SERIES == EQ only, and marks members missing
    from today's set `active_to` (survivorship-aware history from day one).
  - DuckDB `market_data.timestamp` is UTC epoch seconds; date conversion
    must be `CAST(to_timestamp(timestamp + 19800) AS DATE)` (IST offset,
    same pattern as historify_db). Direct `CAST(timestamp AS DATE)` is
    unimplemented in DuckDB and fails at runtime.
  - Dividends are not ledgered (price-return policy); splits parse from
    "From Rs X To Rs Y", bonuses from "a:b" -> ratio b/(a+b).
- Spike answers: NSE library picked: **nsepython 2.101** — participant OI
  (dated), FII/DII, historical bulk/block deals, market-wide CA calendar via
  `nsefetch(<full URL>)` all verified live; its whole-market bhavcopy is
  broken, so that one file comes from a plain archives GET (the same GET
  jugaad-data performs). jugaad-data covers only bhavcopy + current-day bulk
  deals. BSE deliverable bhavcopy: NOT covered by either (bseindia.com
  download host serves its SPA page to plain requests) — documented gap.
  Kotak 1m depth: NOT MEASURED — needs a running server with a Kotak session;
  run `uv run scripts/marketpulse/kotak_intraday_spike.py <api_key>` on the
  live instance and record the answer in master spec 12-B. AMFI URL that
  worked: none from this network (2026 site redesign moved the workbook;
  historical spages URLs 404) — loader is URL-agnostic via
  `MARKETPULSE_AMFI_URL`, point it at wherever the workbook currently lives
  and re-run `industry_map_refresh.refresh()`.
- Gotchas the next session must know:
  - `uv run ruff check . --fix` is a trap on this repo: HEAD has 786
    pre-existing violations; the blanket command rewrites ~98 unrelated
    files. Scope ruff to the files you touched (all marketpulse files are
    clean).
  - nsepython's `nsefetch` needs a FULL URL (bare `/api/...` paths fail with
    a curl "no host" error); its MCP-backed `nse_mcp_get_corporate_actions`
  hangs on multi-symbol probes — use the market-wide calendar endpoint.
  - `pyproject.toml` dependency additions must also land in
    `requirements.txt` AND `requirements-nginx.txt` or
    `test_requirements_parity` fails.
  - The known-events replay tests skip (by design) until historify has
    candles; the replay plumbing itself was verified end-to-end against a
    synthetic duckdb.
- Coverage report number (acceptance gate): **0%** of universe with >= 95%
  coverage over 250 sessions — `db/historify.duckdb` here is a fresh install
  with zero candles. GATE PENDING: on the live instance run
  `uv run scripts/marketpulse/historify_wireup.py` then
  `--backfill --backfill-months 24`, then
  `uv run scripts/marketpulse/coverage_report.py`.
- Known-events suite size and status: 11 events (10 distinct corporate
  actions incl. NAZARA's same-day split+bonus), all real NSE records synced
  2026-10-10; green as 16 passed + 11 skipped-pending-candles.
