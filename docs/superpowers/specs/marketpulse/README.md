# Market Pulse — phase execution kit

One session per phase. The master design doc is the contract; these files are
the session briefs. A fresh session needs: this README, the master design doc,
and its own phase file — nothing else from past conversations.

## Master design doc

`docs/superpowers/specs/2026-10-09-marketpulse-design.md` — architecture,
invariants, data plan, all phases. Read §3 (invariants) and your phase's
entry in §8 before writing any code.

## How to run a phase

1. Open a fresh session in the repo root.
2. Say exactly: `Execute docs/superpowers/specs/marketpulse/phase-N-*.md`
3. The session then, in order:
   - Reads this README, the master design doc, and its phase file.
   - Verifies the prerequisites listed in the phase file (commands provided).
   - If the phase file is a brief (phases 1-7): writes the detailed
     implementation plan to
     `docs/superpowers/plans/<date>-marketpulse-phase-N.md` using the
     writing-plans skill, gets user approval, then executes.
     Phase 0 ships with its full plan already written.
   - Implements with `uv run` only; `uv run ruff check . --fix` +
     `uv run ruff format .` before finishing; `uv run pytest test/ -v` green.
   - Runs the fd-audit skill if the phase touched DB, scheduler, threads,
     files, or sockets (phases 0, 1, 3, 4, 5, 7).
   - Fills the "Session handoff" section at the bottom of its phase file:
     what landed, what deviated from the master spec (and why), spike
     answers, gotchas the next session must know.
   - Updates the status table below.
4. Do not commit unless the user explicitly asks in that session.

## Status

| Phase | File | Status |
| --- | --- | --- |
| 0 | phase-0-data-foundation.md | executed 2026-10-10 (branch feat/marketpulse-phase-0) — code landed, tests green; PENDING on live instance: Historify 2-year backfill + 95% coverage gate, Kotak 1m-depth spike, AMFI workbook URL via MARKETPULSE_AMFI_URL |
| 1 | phase-1-metrics-api.md | brief — not started |
| 2 | phase-2-market-ui.md | brief — not started |
| 3 | phase-3-pattern-engine.md | brief — not started |
| 4 | phase-4-screens-ui.md | brief — not started |
| 5 | phase-5-institutional-joins.md | brief — not started |
| 6 | phase-6-backtester.md | brief — not started |
| 7 | phase-7-fundamentals.md | brief — not started |

## Conventions that apply to every phase

From the repo's CLAUDE.md — repeated here because a fresh session must not
miss them: uv only; `logger = get_logger(__name__)` and `logger.exception()`
for errors (never print/traceback); SQLite engines only via
`database.engine_factory.create_db_engine()` (NullPool, never StaticPool);
schema changes ship as idempotent scripts in `upgrade/` registered in
`migrate_all.py` with `--status` support, tested on a copy of a real DB; new
pages need all three registrations (App.tsx route, react_app.py route,
navigation.ts entry); no asyncio in request code; anything crossing the
eventlet boundary uses `utils/real_threading`; no icons/emojis anywhere;
Conventional Commits.

## Why phases 1-7 are briefs, not full plans

Detailed implementation plans written months ahead go stale: Phase 3's
thresholds get tuned in Phase 3; Phase 5 depends on which NSE library won
the Phase 0 spike; Phase 6 depends on Phase 3's stage grammar. Each brief
pins the contract (goal, prerequisites, scope, acceptance) and the session
writes the detailed plan against the real, current code on day one of that
phase. Phase 0 is fully planned because it is next and its inputs are known.
