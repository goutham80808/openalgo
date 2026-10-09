import pandas as pd

from services.marketpulse_jobs.universe_refresh import refresh


class FakeNse:
    def __init__(self, frame):
        self._frame = frame
        self.calls = []
        self.lot_calls = 0
        self.index_calls = []

    def fetch_nse_bhavcopy_with_delivery(self, date):
        self.calls.append(date)
        return self._frame.copy()

    def fetch_fno_lot_sizes(self):
        self.lot_calls += 1
        return {"INFY": 1250, "DULL": 60000}

    def fetch_index_constituents(self, slug):
        self.index_calls.append(slug)
        return pd.DataFrame({"Symbol": ["INFY"], "ISIN Code": ["INE009A01021"]})


def test_refresh_writes_membership(tmp_path, monkeypatch):
    from sqlalchemy import text

    from database.engine_factory import create_db_engine

    engine = create_db_engine(f"sqlite:///{tmp_path}/openalgo.db")
    from upgrade.migrate_marketpulse import TABLES

    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        conn.commit()

    frame = pd.DataFrame(
        {
            "SYMBOL": ["INFY", "INFY"],
            "SERIES": ["EQ", "EQ"],
            "turnover_cr": [110.0, 130.0],
        }
    )
    fake = FakeNse(frame)
    summary = refresh(engine=engine, nse=fake, dates=["2026-10-08", "2026-10-09"])
    assert summary.members == 1
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM mp_universe")).scalar()
        row = dict(conn.execute(text("SELECT * FROM mp_universe")).mappings().first())
    assert count == 1
    assert len(fake.calls) == 2
    # enrichment: INFY is in the F&O list and the index constituents
    assert row["is_fno"] == 1
    assert row["index_member"] == 1
    assert row["active_from"] == "2026-10-09"
    engine.dispose()


def test_refresh_marks_departures_inactive(tmp_path):
    from sqlalchemy import text

    from database.engine_factory import create_db_engine

    engine = create_db_engine(f"sqlite:///{tmp_path}/openalgo.db")
    from upgrade.migrate_marketpulse import TABLES

    with engine.connect() as conn:
        for ddl in TABLES.values():
            conn.execute(text(ddl))
        # a pre-existing member that today's panel no longer includes
        conn.execute(
            text(
                "INSERT INTO mp_universe (symbol, exchange, active_from) "
                "VALUES ('GONE', 'NSE', '2026-09-01')"
            )
        )
        conn.commit()

    frame = pd.DataFrame({"SYMBOL": ["INFY"], "SERIES": ["EQ"], "turnover_cr": [50.0]})
    summary = refresh(engine=engine, nse=FakeNse(frame), dates=["2026-10-09"])
    assert summary.members == 1
    with engine.connect() as conn:
        gone = conn.execute(text("SELECT active_to FROM mp_universe WHERE symbol='GONE'")).scalar()
        infy = conn.execute(text("SELECT active_to FROM mp_universe WHERE symbol='INFY'")).scalar()
    assert gone == "2026-10-09"
    assert infy is None
    engine.dispose()
