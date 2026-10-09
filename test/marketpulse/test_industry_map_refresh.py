import pandas as pd

from services.marketpulse_jobs.industry_map_refresh import refresh


def _amfi_frame():
    return pd.DataFrame(
        {
            "ISIN": ["INE009A01021", None, "INE466L01038"],
            "SYMBOL": ["INFY", "GARBAGE", "360ONE"],
            "MACRO SECTOR": ["Financial Services", "x", "Financial Services"],
            "SECTOR": ["IT", "y", "Financial Services"],
            "INDUSTRY": ["Computers - Software & Consulting", "z", "Asset Management"],
        }
    )


def test_refresh_upserts_map_and_counts_unmapped(tmp_path):
    from sqlalchemy import text

    from database.engine_factory import create_db_engine

    engine = create_db_engine(f"sqlite:///{tmp_path}/openalgo.db")
    from upgrade.migrate_marketpulse import TABLES

    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        # a universe member waiting for its identity backfill
        conn.execute(
            text(
                "INSERT INTO mp_universe (symbol, exchange, active_from) "
                "VALUES ('INFY', 'NSE', '2026-10-09')"
            )
        )
        conn.commit()

    summary = refresh(engine=engine, fetch=lambda: _amfi_frame())
    assert summary.rows == 2  # the None-ISIN row is dropped
    assert summary.unmapped_isins == 1

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM mp_industry_map")).scalar()
        infy = conn.execute(
            text("SELECT sector, industry FROM mp_industry_map WHERE isin='INE009A01021'")
        ).first()
        identity = conn.execute(
            text("SELECT name, isin FROM mp_universe WHERE symbol='INFY'")
        ).first()
    assert count == 2
    assert infy == ("Financial Services", "IT")
    # identity backfill from the AMFI symbol column
    assert identity == (None, "INE009A01021")
    engine.dispose()


def test_refresh_is_idempotent(tmp_path):
    from sqlalchemy import text

    from database.engine_factory import create_db_engine

    engine = create_db_engine(f"sqlite:///{tmp_path}/openalgo.db")
    from upgrade.migrate_marketpulse import TABLES

    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        conn.commit()

    refresh(engine=engine, fetch=lambda: _amfi_frame())
    refresh(engine=engine, fetch=lambda: _amfi_frame())
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM mp_industry_map")).scalar()
    assert count == 2
    engine.dispose()
