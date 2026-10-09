# Market Pulse Phase 6 — Screen backtester

**Goal:** Backtest any saved screen over stored history, regime-sliced, with honest India costs — the layer that proves or kills the rules.

**Prerequisites (verify before planning):**
- Phase 3 done (pattern history replayable); Phase 5 done (institutional filters joinable into historical screens if their data was backfilled — backfill participant OI/deals history as part of this phase's first task if not).
- **Pre-task:** verify the vectorbt dependency license is compatible with this repo (it is already exemplified at `examples/python/backtesting_vectorbt.py`). If incompatible, fall back to a minimal internal vector loop — decide before adding any dependency.

**Scope (master spec §8-Phase 6, §13):**
- Thin alignment layer: screen membership dates from replayed pattern history → entry at next open, 2xATR stop, chandelier trail, 1.5% risk sizing, 0.35-0.45% cost model (STT + exchange + GST + stamp + impact), ADT-capped slippage and position caps.
- **Point-in-time everything**: universe and RS as of each date (master §5) — survivorship-free or the tearsheet lies.
- Regime slicing: bull/bear/chop labels from our own never-pruned breadth aggregates; per-regime returns, drawdowns, counts.
- Tearsheet per screen (per-preset at minimum); documented cost model; results page or exported report — decide in planning.
- Bridge (only after the backtest earns it): screen rows → planned limit orders at pivot ± band, **sandbox mode first** per platform norms; live routing stays out of scope (master §11).

**Done when:** the VCP preset's backtest produces a tearsheet with sensible per-regime numbers; a deliberately survivorship-biased control (today's universe applied backwards) produces visibly different, worse numbers — proving the point-in-time discipline works.

**Session protocol:** per `README.md`.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- vectorbt verdict (license + used?):
- VCP tearsheet summary numbers:
- Sandbox bridge status:
