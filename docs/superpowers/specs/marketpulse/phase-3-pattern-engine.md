# Market Pulse Phase 3 — Pattern engine

**Goal:** Bases, pivots, stages: the state machine that classifies every liquid stock as Forming / Fresh breakout / Climbing / Played out, plus the transition log that feeds the change feed. The schedule's critical path — the validation harness does not compress.

**Prerequisites (verify before planning):**
- Phases 0-2 done; `mp_symbol_daily` populated for >= 250 sessions.
- Phase 0 handoff notes any corporate-action gotchas (they poison base detection).

**Scope (master spec §8-Phase 3, §2 observed spec):**
- `services/marketpulse/patterns.py` + `scan.py` (pure compute): swing/lid detection (zigzag shared foundation — SMC detectors later reuse it), base segmentation B1..Bn with durations, base max depth, tightening ATR ratio across successive bases, volume dry-up, up/down volume ratio, pivot + band (5% / 20%), now-vs-pivot, powering-up, squat / failed poke, stage classification, transition log with BP-feed grammar ("closed 20% above its pivot — the first time since ...", "five sessions since its breakout — moved to Climbing", "poked above and closed back under — its 4th test of the lid").
- **Start from community-tuned parameters**: mine SCREENi-py / PKScreener / breakout-stock-screener rule formulations and defaults (parameters only — GPL code is reference, never copied). Cross-eyeball against a free public scan in a browser tab (manual, not an integration).
- Migration: `mp_pattern_state` (idempotent, registered, --status, copied-real-DB test).
- Nightly: extend the Phase 1 eod_scan pass with the pattern studies; replay mode must recompute pattern history for backtesting.
- **Validation harness**: render 100 random universe charts with base/pivot annotations to PNGs (matplotlib is fine — this is a dev tool), iterate rules until annotation quality is acceptable to the user; pin ~15 golden cases as pytest fixtures.

**Done when:** stage counts stable across re-runs; the 100-chart set reads correctly on user review; transitions match the grammar; golden tests green; fd-audit run.

**Session protocol:** per `README.md`. Plan the harness early — it is the iteration loop for every rule.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Parameter sources actually used:
- Harness verdict (user eyeball result):
- Golden cases pinned:
