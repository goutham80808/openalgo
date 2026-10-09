# Market Pulse — design document

Date: 2026-10-09
Status: finalized 2026-10-09 — execution model: one session per phase via
docs/superpowers/specs/marketpulse/ (README = protocol, phase-0 = full plan,
phases 1-7 = session briefs); Phase 0 ready to execute
Owner: Goutham + agent

## 1. What we are building

A BananaPatterns-class market analytics layer inside OpenAlgo, customized to our
strategy work: market breadth, sector rotation, relative strength, industry and
index dashboards first; a VCP/Minervini-style pattern engine and screens second;
screen backtesting last. v1 is long-only cash-market analytics on EOD cadence
and carries no advice surfaces; institutional-flow data, intraday cadence, SMC
studies and the options-selling shortlist are planned extensions — see §12.

Two product surfaces:

- **Market** (`/market`) — how the market and its industries are doing today:
  Breadth (over time + across industries), Rotation (RRG quadrant map, which
  doubles as the Strength view), Compare, Industries & Indices detail, All
  indices at a glance.
- **Screens** (`/screens`) — the daily setup scanner: every liquid stock
  classified into a stage (Forming / Fresh breakout / Climbing / Played out),
  per-stock cards with base anatomy, a "what changed since last close" feed,
  preset screens (VCP, Blue sky, Multi-year breakout, IPO base), combined and
  custom screens, list and card views, filters.

Naming: internal module `marketpulse`. Nav labels "Market" and "Screens". No
borrowed branding.

**Reuse doctrine — own the atomic data, borrow everything else.** *Own*:
candles (Historify/Kotak) and the NSE/BSE public files (bhavcopy, delivery,
participant OI, deals, filings) landed nightly into our tables — backtests
and live decisions cannot depend on someone else's API window or data
retention. *Borrow*: indicator maths from the pinned `openalgo` SDK's
`openalgo.ta` (no hand-rolled DMAs/ATRs); NSE file fetching from a
maintained community library (jugaad-data / nsepython — pick one in the
Phase 0 spike) so upstream keeps the endpoint churn, not us; backtest
machinery in the vectorbt style already exemplified at
`examples/python/backtesting_vectorbt.py`; screener rule parameters mined
from SCREENi-py / PKScreener / breakout-stock-screener — parameters and
rule formulations only, their GPL code is reference, never copied in;
options analytics from the existing `/tools` suite; execution from
`/strategy` and the sandbox. *Ingest where it earns its keep*: Screener.in
fundamentals arrive via screener-mcp (MIT, PyPI — see §14), which also
carries an NSE client (announcements, bulk deals). *Link out for
eyeballs*: every stock row deep-links to its Screener.in page — one
anchor. Nothing is ever ingested from Chartink or needed from it: we own
candles (Kotak/Historify) with better provenance, our own chart terminal
renders them with our annotations, and their scan engine is the thing
we're building.

**Honest framing.** v1's screens are workflow parity, not edge — the same
Minervini rules thousands run on Chartink. The edge lives in three places
the free tools do not have: joining institutional datasets onto the screens
(Phase 5), backtesting them honestly on point-in-time data (Phase 6), and
executing from them in-platform. See §13.

## 2. Reference material

Screenshots (design reference only, stays out of the repo — external copyrighted
material):

- `/tmp/opencode/bp-shots/` — 16 anonymous pages (01-home … 16-about)
- `/tmp/opencode/bp-shots/logged-in/` — 20 authenticated pages
  (`m01`-`m09` market views, `s01`-`s07` screens states, `x01`-`x03`
  backtest/indices/stock)

Observed spec extracted from them (our v1 target parity):

- Breadth over time: % in confirmed uptrend, % within 2% of ATH, % above
  50 DMA / 200 DMA, % near 52-wk high/low, up/down on day and from open,
  breakout and squat counts. Period tabs 1D 1W 1M 3M 6M YTD 1Y. Liquid
  universe vs all-stocks toggle.
- Breadth across industries: median gain of the group's stocks + "n of m up",
  bars sorted high to low, period selector, Industries vs Subgroups grouping,
  groups with <= 3 members rendered dimmed with "too few for a ranking to mean
  anything".
- Rotation: scatter, x = strength now (RS), y = momentum (change vs last
  month), four labelled quadrants Turning up / Powering up / Falling back /
  Cooling off with member counts, ranked side list with RS + change, layers
  Industries / Subgroups / Stocks / Watchlist, Industries vs Indices toggle.
  The menu item "Strength" points at this view (the four quadrant labels are
  literally strong/weak/heating/cooling). No separate strength route exists
  (404) — we mirror that.
- Screens: stage summary cards with counts; what-changed feed whose entries
  are state-machine transitions with reasons ("closed 20% above its pivot —
  the first time since 04/09/26", "five sessions since its breakout — moved to
  Climbing", "poked above and closed back under — its 4th test of the lid");
  per-stock cards showing symbol, name, band (5% / 20%), trade-for-trade tag,
  industry links, OHLC + %chg, volume, RS rating, pivot line and base
  annotations B1..B8 with durations (3w..153w), base max depth %, tightening
  ATR ratio, volume dry-up, up/down volume ratio, now vs pivot %, 1-day
  turnover, market cap, squat / failed poke flags, "Powering up" badge,
  earlier-bases-running note, board meeting / results dates; cards vs list
  views; sort by RS rating, now vs pivot, tightening, volume dry-up,
  up/down volume, base age, distance from 52-wk high, market cap; filters for
  announcement, circuit/F&O, results date, watchlist, industry/index, quadrant.
- Stock page: full chart + the same metric table per stock.

## 3. Architecture (approach A — native platform module)

```
Kotak Neo (working) 
  └─ Historify schedules (existing) ──> historify.duckdb   (daily candles, universe)
                                            │
                                            ▼  nightly APScheduler job (after EOD pull)
        services/marketpulse/        pure compute, no I/O
          ├─ universe.py             liquidity floor, exclusions
          ├─ metrics.py              per-symbol daily metrics (pandas)
          ├─ industries.py           industry/subgroup rollups
          ├─ rotation.py             RS + momentum quadrants
          ├─ breadth.py              market + industry breadth series
          ├─ patterns.py             bases, pivots, stages   (phase 3)
          └─ scan.py                 stage classification + change feed (phase 3)
                                            │
                                            ▼
        openalgo.db (new tables, via database/engine_factory)
          mp_universe, mp_industry_map, mp_symbol_daily,
          mp_industry_daily, mp_index_daily, mp_scan_runs,
          mp_pattern_state (phase 3)
                                            │
                                            ▼
        blueprints/marketpulse.py   JSON API for the React pages
                                            │
                                            ▼
        frontend: /market/* and /screens pages, nav entries
```

Invariants honoured:

- `services/marketpulse/` does no I/O — every input is an argument, every
  output is a return value, same contract as `services/risk/`. The job in
  `services/` (or the blueprint) owns reading DuckDB and writing SQLite.
- **Cadence is a parameter, not a constant**: the job runner takes
  `cadence` (eod | intraday) and every study declares which cadence it
  consumes. v1 ships only the eod pass; the intraday pass (§12-B) reuses
  the same runner, so adding it is configuration plus studies — not a new
  pipeline.
- **The EOD pass runs on a real OS thread, never a greenlet.** A
  full-universe 2-year DuckDB read plus pandas compute is tens of seconds
  of C code; under the default eventlet worker that blocks the hub and
  freezes the whole app — invisible on `uv run app.py`, fatal on deploy.
  The job runs via `utils/real_threading`, opens its DuckDB connection
  read-only per run and closes it, and is triggered by the existing
  `historify_schedule_execution_complete` event — not a wall-clock cron,
  which could fire mid-download and hit DuckDB's single-writer lock.
- **The module is organized as a studies registry**, and the registry is
  cross-sectional by design: a study registers `(name, cadence, declared
  inputs, produced schema)` — inputs are panels (whole-universe frames, OI
  tables), not single-symbol candles, because breadth and rotation rank
  across the universe. Per-symbol studies are the special case, not the
  interface. Adding a study is one file plus one registration line; a
  migration only if it persists a genuinely hot-queried column.
- All dates in `mp_*` tables are IST trading dates; the scheduler timezone
  is `Asia/Kolkata`.
- All engines via `database.engine_factory.create_db_engine()` (NullPool).
- Schema changes ship as idempotent migration scripts in `upgrade/` registered
  in `migrate_all.py`, `--status` supported, safe on populated DBs.
- Pages follow the three registrations: route in `frontend/src/App.tsx`, path
  in `blueprints/react_app.py`, nav entry in `frontend/src/config/navigation.ts`.
- `logger = get_logger(__name__)`, `logger.exception()` for errors, uv + ruff.
- After any phase touching DB/scheduler/jobs: run the fd-audit skill.

## 4. Data plan

- Source of truth for EOD candles: **Historify schedules over the working
  Kotak connection**, landing in `historify.duckdb`. Universe watchlist =
  all NSE+BSE equities passing the ADT floor (default: 20-day average traded
  value > ₹1 Cr — config `MARKETPULSE_ADT_FLOOR_CR`, matching BP's "liquid
  universe" scale of ~1,600-1,900 names; mcap filter config exists but is off
  by default per user decision).
- Pacing: ~1,600-1,900 symbols / night. Kotak's candle API previously needed
  rate-limit fixes; the schedule must batch (e.g. 200-symbol chunks), pace,
  and resume on failure. Backfill 2 years of daily candles first (weekly
  cadence for the tail beyond 2y is acceptable: metrics need ~260 sessions).
- Delivery %: **both exchanges** — NSE `sec_bhavdata_full` and BSE's
  deliverable-quantity bhavcopy (it exists; skipping it drops the signal
  exactly where it is scarcest). Fetched through the chosen community NSE
  library; lands in `mp_symbol_daily`; never blocks the scan.
- Corporate actions — the #1 correctness risk, hardened: store **raw and
  adjusted series plus per-(symbol, ex-date) adjustment factors** as
  first-class rows (`mp_adjustments`); a nightly automated cross-check of
  the full universe against the exchange CA calendar; a **known-events
  regression suite** (recent splits/bonuses/mergers with expected factors —
  HDFC twins, AB group restructurings) that must stay green; dividend
  policy pinned to price-return for screens. Broker EOD in India is
  routinely mis-adjusted — one missed split fabricates a −80% "base" that
  the screener flags with confidence.
- Market cap + events: mcap from NSE's monthly mcap files (or
  shares-outstanding x price — decided in Phase 0); results/board-meeting
  calendar and IPO listing dates land in an `mp_events` table — Phase 4's
  filters are vaporware without it.
- FII/DII flows, participant-wise F&O OI, block/bulk deals (Phase 5):
  fetched by the same community library — we write no NSE scraping of our
  own.
- Index data: all NSE/BSE index closes for the "all indices at a glance" and
  index-vs-industry toggles (Historify supports index symbols; fall back to
  NSE archives CSV if coverage is thin).
- Industry/subgroup classification: AMFI security master (free, covers every
  NSE+BSE ISIN) loaded into `mp_industry_map`. AMFI's MacroSector / Sector /
  Industry columns map to our three grouping levels — Sector (top),
  Industry (the 50-60 row ranking breadth uses), Subgroup (the ~190-row fine
  grouping rotation offers); refreshed monthly; unmapped symbols land in an
  "Unclassified" group that is excluded from industry rankings but visible in
  all-stocks views.

## 5. Database sketch (all SQLite via engine_factory, migration script per table)

- `mp_universe(symbol, exchange, name, isin, is_fno, index_member,
  delivery_available, active_from, active_to, reasons)`
  — nightly membership snapshot; symbols leave but rows persist (survivorship-
  aware screens later). The `is_fno` / `index_member` / `delivery_available`
  flags are cheap now and a populated-DB migration later — Phase 0 fills them.
- `mp_industry_map(isin, sector, industry, subgroup, source, updated_at)`.
- `mp_symbol_daily(date, symbol, exchange, close, pct_chg, volume, turnover_cr,
  adt20_cr, r_s_rating, above_dma20/50/200 (bools), pct_from_52w_high,
  pct_from_52w_low, updown_vol_ratio, vol_dryup, atr_ratio_tightening,
  delivery_pct, r1m/r3m/r6m/r12m, extras JSON)` — one row per symbol per
  session. Typed columns are the hot-query fields; `extras` carries new or
  custom metrics so a study can persist a value **without shipping a
  migration** — the main "customize it later" seam.
- `mp_industry_daily(date, industry, subgroup, members, median_gain_1d/1w/1m/…,
  pct_above_dma50, net_new_highs, rs_now, rs_mom_1m, quadrant)`.
- `mp_index_daily(date, index_name, close, pct_chg, rs_now, rs_mom_1m, quadrant)`.
- `mp_scan_runs(run_date, status, symbols_scanned, started_at, finished_at,
  error)`.
- `mp_adjustments(symbol, ex_date, action, ratio)` — the adjustment ledger
  behind the adjusted series; replayable, auditable.
- `mp_events(symbol, date, kind, detail)` — results, board meetings, IPO
  listing dates.
- `r_s_rating` definition (pinned): IBD-style weighted multi-horizon
  percentile (3m/6m/9m/12m blend) computed against the **point-in-time
  universe as of that date** from `mp_universe` history — never today's
  universe applied backwards, or Phase 6's backtests silently inherit
  survivorship bias.
- Phase 3: `mp_pattern_state(date, symbol, stage, base_count, current_base_no,
  base_start, base_depth_pct, pivot, band_pct, tightening_ratio, vol_dryup,
  updown_vol, now_vs_pivot_pct, powering_up, squat, failed_poke, change_reason)`.

Indexes on (date, symbol) / (date, industry). Retention: industry, index and
breadth **aggregates are never pruned** (tiny — and multi-year breadth
history doubles as the regime labeller for Phase 6); `mp_symbol_daily` kept
~3 years daily (pruning to weekly at 2y would break 52-week metrics at the
boundary).

## 6. API surface (`blueprints/marketpulse.py`, session-auth web routes)

- `GET /market/api/summary?date=` — run status, universe size, "updated EOD" stamp.
- `GET /market/api/breadth/timeseries?days=&universe=` — over-time series.
- `GET /market/api/breadth/industries?period=1D|1W|1M|3M|6M|YTD|1Y&grouping=industries|subgroups&universe=`
- `GET /market/api/rotation?layer=industries|subgroups|indices&universe=`
- `GET /market/api/indices` — all indices glance.
- `GET /market/api/industry/<slug>` and `GET /market/api/index/<name>` — member
  lists stock by stock.
- `GET /market/api/compare?symbols=a,b,c` — normalized performance.
- Phase 4: `GET /screens/api/today?stage=&preset=&sort=&filters…`,
  `GET /screens/api/feed?date=`, `GET /screens/api/stock/<symbol>`,
  `POST /screens/api/custom` (saved custom screens).

## 7. Frontend

New pages (React 19, existing component patterns, `Layout` shell, TanStack
Query for server state, openalgo-charts for the time series and later stock
charts):

- `/market` — redirects to `/market/breadth`.
- `/market/breadth` — two tabs (Over time | Across industries), period tabs,
  universe toggle, grouping + metric selectors, small-group dimming.
- `/market/rotation` — quadrant scatter (labelled Turning up / Powering up /
  Falling back / Cooling off — trader words, no "arming"-style jargon), ranked
  side list, layer tabs, dot click-through to industry detail.
- `/market/indices` — all indices at a glance (stage + change grid).
- `/market/compare` — pick any symbols/indices, normalized lines.
- `/market/industry/<slug>`, `/market/index/<name>` — stock-by-stock member
  tables (the "Industries & Indices" surface).
- Phase 4 `/screens` — stage cards + change feed + card/list renders + filters
  + presets + custom screens.
- Phase 5 stock page `/market/stock/<symbol>` — chart + full metric table,
  linked from every table/card.

All copy in trader language per repo convention; no status codes or jargon in
user-facing errors.

## 8. Phases, acceptance criteria, estimates

**Execution model:** one fresh session per phase — see
`docs/superpowers/specs/marketpulse/README.md` for the protocol, the
per-phase files, and the running status table.

### Phase 0 — Data foundation (≈ 5-6 days)
Universe builder — including the **`is_fno`, index-membership and
delivery-availability flags on `mp_universe`** — AMFI industry map, Kotak
Historify schedules for the liquid universe, 2-year daily backfill,
`mp_universe` + `mp_industry_map` + `mp_adjustments` migrations.
**Spikes** (same week): (a) how much intraday history the Kotak candle API
actually returns (brokers commonly cap at 30–90 days) — answer recorded in
§12-B, it decides backfill-vs-live-record for the intraday pass; (b) pick
the NSE-fetch community library (jugaad-data vs nsepython) and prove it
against bhavcopy + delivery + participant OI + FII/DII + deals endpoints.
**Done when**: nightly candle coverage >= 95% of universe for 250 sessions;
the corporate-actions known-events regression suite is green and the nightly
full-universe CA cross-check reports zero unexplained adjustments;
`--status` migration verified on a copied real DB; intraday-history depth
and library pick written down.

### Phase 1 — Metrics + nightly compute + API (≈ 4 days)
`metrics.py` (built on `openalgo.ta` + pandas — no hand-rolled indicators),
`industries.py`, `breadth.py`, `rotation.py`, the nightly job on a real
thread triggered by `historify_schedule_execution_complete` (retry-once,
`mp_scan_runs` audit), API endpoints for summary/breadth/rotation/indices,
and a **replay mode** (`--from`/`--to`, per study) that rewrites partitions
transactionally — metric definitions *will* change, and silent
mixed-formula history is an analytics bug. Golden-day unit tests with
pandas fixtures.
**Done when**: running the job twice for the same date is idempotent; a
replay over a range reproduces identical rows; a `curl` of each endpoint
returns a full JSON for yesterday.

### Phase 2 — Market UI (≈ 6 days)  ← first user-visible milestone
All `/market/*` pages per section 7 with real data. Rotation ships as a
static quadrant scatter with click-through — **no hover trails in v1**
(BP's trails are polish, not information). Every symbol row carries the
Screener.in out-link.
**Done when**: breadth, rotation, indices, industry detail and compare render
from the API on your instance and match the reference screenshots' information
density; three registrations complete for every route.

### Phase 3 — Pattern engine (≈ 9 days)
`patterns.py` + `scan.py`: swing/lid detection, base segmentation and counting
with durations, base max depth, tightening ATR ratio across successive bases,
volume dry-up, up/down volume ratio, pivot + band, now-vs-pivot, powering-up,
squat / failed poke, stage classification (Forming / Fresh breakout / Climbing /
Played out) and the transition log that feeds the change feed.
**Start from community-tuned parameters, not the books alone**: SCREENi-py,
PKScreener and breakout-stock-screener have iterated thresholds with real
NSE users for years — mine their rule formulations and defaults (parameters
only; their GPL code is reference, never copied into this repo). During
validation, cross-eyeball our output against an equivalent free public
scan in a browser tab (Chartink works fine for this — manual eyeballing,
not an integration) as a third-party sanity check.
Includes a **validation harness**: render 100 random universe charts with base
annotations to PNGs for eyeball verification; iterate rules until annotation
quality is acceptable; pin ~15 golden cases as unit tests (this step is the
schedule's critical path and does not compress).
**Done when**: stage counts on a given date are stable across re-runs; the 100
chart set reads correctly to you on review; transitions match the BP feed's
grammar ("closed above its pivot…", "moved to Climbing…").

### Phase 4 — Screens UI (≈ 6 days)
`/screens` page: stage cards, what-changed feed (stage moves + `mp_events`
join), card/list views with all sort keys, filters (industry, quadrant,
results date, watchlist, circuit/F&O), presets VCP / Blue sky / Multi-year
breakout / IPO base, combined screens, saved custom screens, Download/Copy.
**Done when**: today's screen on your instance shows plausible names with
annotated cards; every filter and preset works from the API.

### Phase 5 — Institutional flow joins + stock pages (≈ 6 days) ← the first edge layer
Promoted from §12-A because the community NSE library does the fetching:
`mp_participant_oi_daily` (FII/DII/client long-short ratios), FII/DII cash
flows, `mp_deals` (block + bulk), per-symbol futures OI build-up vs price
(Historify candles already carry `oi`). Derived signals joined onto the
screens: breakout-day delivery vs its own 20-day average; fresh breakout +
futures long build-up (and the weak short-covering variant filtered down);
the participant-OI risk-on gate (FII long-short trend + Nifty vs 50DMA)
that decides whether breakouts are worth taking at all; block deals in
the last N sessions + now basing. Plus `/market/stock/<symbol>` full pages,
watchlist integration, and a Telegram digest of watchlist stage moves
(platform Telegram infrastructure already exists — config, not build).
**Done when**: the screens answer "who else is on this trade" per name;
the risk gate renders on `/market`; every symbol deep-links to a full
page; the digest lands in Telegram.

### Phase 6 — Screen backtester (≈ 4-5 days)
Backtest any saved screen over stored history **on vectorbt** — the
approach `examples/python/backtesting_vectorbt.py` already exemplifies; we
write only the thin alignment layer (screen membership dates → entry at
next open, 2xATR stop, chandelier trail, 1.5% risk sizing, 0.35-0.45%
India cost model, ADT-capped slippage), not an event loop from scratch.
Results are **regime-sliced** using our own never-pruned breadth history
as the bull/bear/chop labeller. Dependency-license check is a Phase 6
pre-task; fallback is a minimal internal vector loop. Bridge (only after
the backtest earns it): screen rows → planned limit orders at pivot ±
band, **sandbox mode first** per platform norms.
**Done when**: the VCP preset's backtest produces a tearsheet with sensible
per-regime numbers and the cost model documented.

### Phase 7 — Fundamentals ingestion via screener-mcp (≈ 3 days; can be pulled forward beside Phase 5)
Two modes, one MIT-licensed dependency (`screener-mcp`, on PyPI — its
client and HTML parsers are importable, not MCP-only):
1. **Agent tools (config, near-zero build)**: register the screener MCP
   server with the platform agent — 30 tools (overview, financials,
   quarterly results, shareholding with promoter pledge, peer comparison,
   rule-based red flags; no Screener.in login needed for research) answer
   "why is this on my screen" conversationally.
2. **ETL for joins**: import its paced client (0.25s min interval,
   concurrency cap 3, shared 429 cooldown) to fetch **only screen-hit and
   watchlist names** — never 1,800-page sweeps — into
   `mp_fundamentals(symbol, fetched_at, payload JSON)`, quarterly refresh
   plus on-demand miss. Screen filters then include promoter pledge %,
   FII/DII quarterly change, ROCE, D/E, market cap; the stock page renders
   the condensed view from the same payload.
Caveat: the data lags ~1 quarter (their own documentation) — filters and
context, never timing signals.

Total ≈ 46 working days end-to-end; first milestone (market views live) ≈ day 15.
Reuse trims roughly a week and a half off the naive build: NSE-fetch
library (Phases 0 and 5), `openalgo.ta` (Phase 1), community-tuned
parameters (Phase 3), vectorbt (Phase 6), screener-mcp (Phase 7).

## 9. Error handling

- Nightly job: partial-data days are recorded in `mp_scan_runs` and flagged in
  the UI's "Updated EOD" stamp ("partial — 87% of universe"); never silently
  stale.
- Kotak failures: retry once, then mark the run failed and keep yesterday's
  tables; UI shows last-good date.
- DuckDB contention: the marketpulse job opens the file read-only and only
  after Historify's completion event fires; if the lock is nonetheless
  held, it backs off and retries rather than failing the run.
- Holidays / no-trade days: job no-ops cleanly.
- Symbol churn (new listings, name changes, suspended): universe snapshotting
  keeps history honest; new listings join with < 6 months of history flagged
  and excluded from RS percentile until 120 sessions exist.
- Blueprint endpoints always return `{status, data|message}` envelopes;
  `logger.exception()` on the server, trader-language message to the client.

## 10. Testing

- Pure-compute unit tests (pandas fixtures) for metrics, industries, rotation
  quadrant assignment, pattern golden cases.
- Idempotency test: run the nightly job twice on the same date, assert no row
  drift.
- API smoke tests against a seeded DB.
- Frontend: component tests for the breadth/rotation tables (existing vitest
  patterns).
- fd-audit after phases 1, 3, 4 (scheduler + DB + new blueprint).

## 11. Non-goals (v1)

v1 ships no intraday screens, no advice/recommendation copy, no scraping of
BananaPatterns' proprietary data and no republishing of their annotations —
our implementation computes everything from our own Kotak/NSE data.
**Options-selling analytics are not rebuilt here**: the option chain, OI
tracker, Max Pain, volatility surface, IV scanner and strategy builder
already live under `/tools`, and multi-leg execution under `/strategy`.
Market Pulse's future contribution is the underlying-level regime shortlist
that feeds them (§12-D). Backtesting of screens is phase 6; live signal
routing to the strategy module is explicitly out of scope until the
backtester proves the rules.

## 12. Extension paths (v1.1+)

The four directions deferred from v1, each with the seam that makes it cheap.
None changes the §3 pipeline — all reuse it.

### A. Institutional flow — "follow the big money"

**Promoted into v1 as Phase 5** (the community NSE library made it cheap).
Remaining here: `mp_iv_daily` — nightly ATM IV + IV rank per F&O underlying,
captured from the Kotak chain (feeds §12-D).

### B. Intraday cadence

- The §3 job runner already takes `cadence`; the intraday pass reuses it.
- **Storage rule**: raw intraday bars stay in `historify.duckdb` (already 1m
  + OI capable, with computed intervals); SQLite keeps only intraday
  *snapshots* with a short retention window (e.g. 5 sessions) — never
  1m × full-universe rows in `mp_*`.
- Source depends on the Phase 0 spike: if broker history depth permits
  backfill → pull it; otherwise build bars from the live ZMQ tick pipeline
  during market hours (a market-hours recorder of its own — eventlet green
  rules, fd-audit, scheduled as a separate small phase).
- Drives: intraday shortlisting for swing entries, intraday SMC (§12-C),
  intraday F&O OI snapshots.

### C. Smart Money Concepts studies

- Same pure-compute contract; each detector is a studies-registry module:
  swings/zigzag (**shared with the Phase 3 base detector**), order blocks,
  fair-value gaps, BOS/CHoCH, liquidity sweeps, premium/discount zones.
- Daily-TF SMC is available as soon as the studies registry exists;
  meaningful intraday SMC waits on §12-B.
- Persistence: annotations computed on demand from duckdb candles; only
  latest-state summaries land in SQLite (`extras` or one small table).

### D. Options-selling shortlist

- New study over `mp_iv_daily` + `mp_symbol_daily` + results calendar:
  a "sellable this week" list — IV rank high, trend with the sell side, no
  earnings before expiry, F&O + liquid.
- Deep-links into the existing `/tools` chain/vol pages and `/strategy`
  builder; Market Pulse never renders chain analytics itself.
- v1 groundwork already laid: universe F&O flags (Phase 0), results-date
  filter (Phase 4), `extras` metrics (§5).

### Customization mechanics (applies to all four)

Studies registry + `extras JSON` + config thresholds
(`MARKETPULSE_ADT_FLOOR_CR` pattern) + custom screens ⇒ adding a metric or a
screen is a new study file plus a registration line. A migration is needed
only when a new persisted column is genuinely hot-queried.

## 13. Edge statement and the daily flow

**Where the edge is (and is not).** The screens themselves are parity — the
same rules thousands run on Chartink. The edge lives where free tools do
not go: (1) **joins** — stage x delivery x futures OI build-up x
participant OI x block deals on one page (Phase 5); (2) **honest
backtests** on point-in-time data with India costs and regime slicing
(Phase 6); (3) **execution in-platform** — screen row to planned order,
sandbox first; (4) **attention discipline** — the change feed converts a
1,800-stock scan into a 15-item read, and every screen is judged on
least-read-time-per-decision; (5) **IV rank against our own captured
history** for the options-selling regime gate (§12-D).

**Explicitly rejected** (recorded so it is not re-litigated): news/NLP
sentiment (the platform agent covers ad-hoc questions; per-headline latency
is a pro-data game), tape/order-flow reading (no retail edge at available
depth), indicator count as a feature, scraping Screener.in or Chartink
(deep links only — their terms, our convenience).

**The daily flow we optimize for.** 16:05 job completes → Telegram digest
lands (regime colour, watchlist stage moves, fresh breakouts worth a look)
→ five-minute screen review, fundamentals one question away (agent with
screener tools) or one click out to Screener.in → morning: planned levels
ready, orders (sandbox until backtested) one click away. The product's job
is that this takes minutes, not evenings.

## 14. Fundamentals (powered by screener-mcp)

No clone, no parser of our own:
[LogeshR15/screener-mcp](https://github.com/LogeshR15/screener-mcp) (MIT,
on PyPI) already paces, authenticates and parses Screener.in — financials,
ratios, 8-quarter shareholding with promoter pledge, peer comparison and
rule-based red flags, plus NSE announcements and bulk deals on the same
client. Company research needs no Screener.in login; only fundamental
screening clauses need a free account. Integration is Phase 7: MCP
registration for the agent (conversational) + library import for ETL into
`mp_fundamentals` (filters and joins). Boundaries: personal-instance use,
paced fetches of names we are actually deciding on, no republishing of
their content. If the project ever dies, the parse target is stable HTML
and a thin own parser is the fallback.
