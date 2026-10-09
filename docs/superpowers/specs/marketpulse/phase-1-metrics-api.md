# Market Pulse Phase 1 — Metrics, nightly compute, API

**Goal:** The EOD compute pass: per-symbol daily metrics, industry/index rollups, breadth, rotation quadrants — computed nightly and served as JSON.

**Prerequisites (verify before planning):**
- Phase 0 status in `docs/superpowers/specs/marketpulse/README.md` is "done"; its handoff section is filled.
- `uv run python scripts/marketpulse/coverage_report.py` shows >= 95% coverage.
- `sqlite3 database/openalgo.db "SELECT COUNT(*) FROM mp_universe WHERE active_to IS NULL"` returns ~1,600-1,900.

**Scope (master spec §3, §5, §6, §8-Phase 1):**
- `services/marketpulse/metrics.py`, `industries.py`, `breadth.py`, `rotation.py` — pure compute on pandas frames; indicators via the `openalgo` SDK's `openalgo.ta`, nothing hand-rolled.
- Studies registry per master §3: entries are `(name, cadence, declared inputs, produced schema)`; cross-sectional by design.
- `services/marketpulse_jobs/eod_scan.py`: real OS thread (`utils/real_threading`), triggered by the `historify_schedule_execution_complete` event (see services/historify_service.py:1797 for the emit site; add a listener — do not poll a cron), opens historify.duckdb read-only per run, retry-once, writes `mp_symbol_daily`, `mp_industry_daily`, `mp_index_daily`, `mp_scan_runs`; idempotent for the same date (delete+rewrite the date partition inside one transaction).
- Replay mode: `--from/--to` per study, transactional partition rewrite.
- `r_s_rating`: point-in-time IBD-style percentile (3m/6m/9m/12m blend) against the universe active on that date (master §5).
- Migrations for the three daily tables + mp_scan_runs (idempotent, registered in migrate_all.py, --status, tested on a copied real DB).
- `blueprints/marketpulse.py` with the summary/breadth/rotation/indices endpoints of master §6 (session-auth, `{status, data|message}` envelopes).
- Frontend: none in this phase.

**Done when:** job twice on one date = no row drift; replay over a range reproduces identical rows; each endpoint `curl`s a full JSON for yesterday; `mp_scan_runs` records a run per pass; golden-day pandas-fixture tests green.

**Session protocol:** per `README.md` — write the detailed plan to `docs/superpowers/plans/<date>-marketpulse-phase-1.md` (writing-plans skill), get approval, execute, fd-audit, fill handoff below.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Gotchas the next session must know:
- Endpoint smoke results:
