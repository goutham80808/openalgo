# Market Pulse Phase 2 — Market UI

**Goal:** All `/market/*` pages rendering live data — the first user-visible milestone.

**Prerequisites (verify before planning):**
- Phase 1 done: `curl -b <session> http://127.0.0.1:5000/market/api/summary` returns success with a fresh EOD stamp.
- Breadth/rotation/indices endpoints return populated JSON for yesterday.

**Scope (master spec §7, §8-Phase 2):**
- Pages: `/market` (redirect to breadth), `/market/breadth` (over time | across industries tabs, period tabs, universe toggle, grouping + metric selectors, small-group dimming for <= 3 members), `/market/rotation` (static quadrant scatter — Turning up / Powering up / Falling back / Cooling off — ranked side list, layer tabs, click-through; no hover trails in v1), `/market/indices`, `/market/compare`, `/market/industry/<slug>`, `/market/index/<name>`.
- React 19 + TanStack Query, existing Layout shell, openalgo-charts for time series; the scatter is a plain SVG/Canvas component.
- Every symbol row carries the Screener.in out-link.
- Three registrations for every route: `frontend/src/App.tsx`, `blueprints/react_app.py`, `frontend/src/config/navigation.ts` (nav under profileMenuItems per master).
- Trader-language copy only; errors name cause + next action, never codes.

**Done when:** all pages render from the API on the live instance at the reference screenshots' information density; three registrations complete for every route; component tests (vitest, existing patterns) for breadth table and rotation list.

**Session protocol:** per `README.md` — detailed plan first, approval, execute, fill handoff below. fd-audit not expected to apply (no DB/thread changes) — run it anyway if the diff touches any.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Gotchas the next session must know:
- Routes registered (all three registrations listed):
