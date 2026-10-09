# Market Pulse Phase 5 — Institutional flow joins + stock pages

**Goal:** The first edge layer: institutional datasets joined onto the screens, plus per-stock full pages and the Telegram digest.

**Prerequisites (verify before planning):**
- Phase 4 done. Phase 0 handoff records the NSE library pick and its per-capability coverage — confirm participant OI, FII/DII flows, and block/bulk deals were covered (or note the gap and the fallback used).

**Scope (master spec §8-Phase 5, §12-A remainder):**
- Tables via registered migration: `mp_participant_oi_daily` (client/HNI/DII/FII/prop long-short), `mp_deals` (block + bulk), FII/DII cash-flow table.
- Fetchers extend `services/marketpulse_jobs/nse_client.py` (paced); futures OI build-up comes from the `oi` column already on historify candles — no new feed.
- Derived signals joined onto the screens: breakout-day delivery vs its own 20-day average; fresh breakout + futures long build-up (and the weak short-covering variant filtered down); the participant-OI risk-on gate (FII long-short trend + Nifty vs 50DMA) rendered on `/market`; block deals in last N sessions + now basing.
- `/market/stock/<symbol>` full page (chart via openalgo-charts + metric table + institutional panel + Screener.in out-link); watchlist integration (existing platform watchlists); Telegram digest of watchlist stage moves (platform Telegram infra is config, not build).

**Done when:** the screens answer "who else is on this trade" per name; the risk gate renders on `/market`; every symbol deep-links to a full page; the digest lands in Telegram; fd-audit run.

**Session protocol:** per `README.md`.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Institutional signal definitions as implemented:
- Telegram digest schedule + sample:
