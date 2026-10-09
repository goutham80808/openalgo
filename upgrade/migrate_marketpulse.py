#!/usr/bin/env python3
"""Market Pulse foundation tables: universe, industry map, adjustments, events.

Idempotent: checks sqlite_master and returns quietly when present.
Supports --status to report without applying. Tested on fresh and copied
real databases (see test/marketpulse/test_migrate_marketpulse.py).

Uses the repo-wide DATABASE_URL convention (default sqlite:///db/openalgo.db,
resolved against the project root) rather than a private path so a run from
any directory touches the database the app actually opens.
"""

import argparse
import logging
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def resolve_sqlite_path(db_url):
    """Make a relative sqlite:/// path absolute against the project root.

    See upgrade/migrate_agent.py for the rationale: DATABASE_URL is relative
    by default and must not resolve against the current directory.
    """
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return db_url
    path = db_url[len(prefix) :]
    if os.path.isabs(path):
        return db_url
    return prefix + os.path.join(PROJECT_ROOT, path).replace("\\", "/")


def get_database_url():
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    return resolve_sqlite_path(os.getenv("DATABASE_URL", "sqlite:///db/openalgo.db"))


def _sqlite_file(db_url):
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return None
    return db_url[len(prefix) :]


def _connect():
    path = _sqlite_file(get_database_url())
    if path is None:
        raise SystemExit("Market Pulse tables are SQLite-only; DATABASE_URL is not sqlite")
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
        logging.exception("marketpulse migration failed")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    sys.exit(main(apply_changes=not args.status))
