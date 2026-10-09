"""Migration idempotency: applied twice, tables identical, --status quiet."""

import os
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
    env = {"DATABASE_URL": f"sqlite:///{db_path}"}

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
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--status"],
        env={**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mp_universe" in result.stdout
    db = sqlite3.connect(db_path)
    assert _tables(db) == []
    db.close()
