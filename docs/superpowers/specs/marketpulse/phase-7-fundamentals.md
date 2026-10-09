# Market Pulse Phase 7 — Fundamentals via screener-mcp

**Goal:** Screener.in fundamentals in the product: conversational agent tools + parsed fundamentals joined onto the screens.

**Prerequisites (verify before planning):**
- Phase 4-5 done (screens exist to join onto).
- `uv add screener-mcp` installs; `uv run python -c "import screener_mcp"` passes. Repo: https://github.com/LogeshR15/screener-mcp (MIT, PyPI).

**Scope (master spec §8-Phase 7, §14):**
- **Agent tools (config-level)**: register the screener MCP server with the platform agent (see `services/agent/mcp/registry.py` for the existing external-MCP hook). Company research needs no Screener.in login; only screening clauses do (free account; credentials via env, never committed).
- **ETL**: import its paced client (0.25s min interval, concurrency 3, shared 429 cooldown) to fetch **only screen-hit and watchlist names** — never universe-wide sweeps — into `mp_fundamentals(symbol, fetched_at, payload JSON)` via a registered migration; quarterly refresh plus on-demand miss.
- Screen filters from the payload: promoter pledge %, FII/DII quarterly change, ROCE, D/E, market cap; stock page renders the condensed fundamentals view from the same payload.
- Boundaries: personal-instance use, paced fetches of names actually being decided on, no republishing. Data lags ~1 quarter — filters and context, never timing signals.

**Done when:** asking the agent "why is X on my screen" returns fundamentals + pledge + red flags; the pledge/FII-DII/ROCE filters work on /screens; the stock page shows the condensed view.

**Session protocol:** per `README.md`; fd-audit if the fetcher added a scheduled job.

## Session handoff

- Completed:
- Deviations from the master spec (and why):
- Agent registration details (how wired, which tools exposed):
- Filter definitions as implemented:
