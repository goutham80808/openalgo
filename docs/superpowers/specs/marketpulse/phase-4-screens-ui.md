# Market Pulse Phase 4 — Screens UI

**Goal:** `/screens` — the daily setup scanner surface: stage cards, change feed, presets, filters, custom screens.

**Prerequisites (verify before planning):**
- Phase 3 done: `mp_pattern_state` populated for yesterday; transitions flowing.
- `mp_events` populated (results/board-meeting/IPO dates) — if the fetcher does not exist yet, it is added in this phase (master §4: NSE financial-calendar sources via the nse_client wrapper).

**Scope (master spec §6 screens endpoints, §7, §8-Phase 4):**
- `GET /screens/api/today?stage=&preset=&sort=&filters...`, `GET /screens/api/feed?date=`, `GET /screens/api/stock/<symbol>`, `POST /screens/api/custom` (saved custom screens).
- UI: stage summary cards with counts; what-changed feed (stage moves + mp_events join); card and list views; per-card fields per master §2 (band, OHLC + %chg, volume, RS rating, pivot line, base annotations with durations, depth, tightening, VDU, up/down volume, now-vs-pivot, turnover, mcap, squat/failed-poke flags, results dates); sort keys (RS rating, now vs pivot, tightening, VDU, up/down volume, base age, distance from 52w high, mcap); filters (announcement, circuit/F&O, results date, watchlist, industry/index, quadrant); presets VCP / Blue sky / Multi-year breakout / IPO base; combined screens; Download/Copy.
- Three registrations for `/screens` and children.
- Trader-language copy; least-read-time-per-decision is the design judge (master §13).

**Done when:** today's screen shows plausible annotated names; every filter/preset/sort works from the API; custom screens save and reload; component tests green.

**Session protocol:** per `README.md`; fd-audit if the events fetcher added a scheduled job.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Presets implemented (and their parameter sources):
- Events fetcher: added in this phase? source + refresh cadence:
