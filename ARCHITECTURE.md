# Architecture

## Overview

Vibe Trading is a multi-agent crypto swing-trading bot that runs on a 4-hour cadence.
A cron-driven scheduler fetches trending symbols and OHLCV candles from Binance via CCXT,
runs a **TechnicalVolumeAnalyst** LLM agent (tool-use loop over 6 data-fetching tools)
followed by a **HeadTrader** LLM agent to produce structured qualitative decisions, and
then hands those decisions to a deterministic **RiskManager** that computes all precise
entry/stop/take-profit prices and position sizes using Python `Decimal` arithmetic.
The resulting order is submitted to a **PaperBroker** (or live `CoinbaseBroker`), with
state persisted to a single-file DuckDB read-cache and a Supabase Postgres transactional
store. A FastAPI + Next.js dashboard surfaces metrics, decisions, and cost data in
real time.

---

## Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  TradingScheduler  (runtime/scheduler.py)  — every 4 hours  │
│  APScheduler cron: hour='*/4', minute=1                      │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  DataFetcher  (data/fetcher.py)  │  ◄── Binance CCXT (OHLCV, funding rate, OI)
│  - incremental_update()          │       CoinGecko (trending symbols)
│  - bootstrap_if_needed()         │
└──────────────┬───────────────────┘
               │  writes candles
               ▼
┌─────────────────────────────────────────────┐
│  DuckDB  (data/vibe_trading.db)             │  ← read-cache
│  Tables: candles · features                 │
│          trades · portfolio_state           │
│          open_positions · decision_log      │
└─────────────┬───────────────────────────────┘
              │  reads candles / features
              │
     ┌────────┴──────────────────────────┐
     │                                   │
     ▼                                   ▼
┌────────────────────┐      ┌──────────────────────────────────┐
│  FeaturePipeline   │      │  ToolExecutor  (agents/tools.py)  │
│ (features/pipeline)│      │  6 tools fed to the analyst:     │
│  TA-Lib indicators │      │  · get_candles (4h + 1d)         │
│  scipy S/R         │      │  · get_indicators                │
│  CDLENGULFING etc. │      │  · get_support_resistance        │
└────────┬───────────┘      │  · get_candlestick_patterns      │
         │ snapshot dict    │  · get_derivatives               │
         │ (legacy path)    │  · get_market_sentiment          │
         │                  └───────────┬──────────────────────┘
         │                              │ tool results (JSON)
         └──────────────┬───────────────┘
                        │
                        ▼
       ┌────────────────────────────────────────┐
       │  TechnicalVolumeAnalyst (agents/analyst)│
       │  Tool-use loop via client.call_llm_     │
       │  with_tools()  (≤ 10 iterations)        │
       │                                         │
       │  Output: AnalystOutput (Pydantic)        │
       │  · market_bias  enum (bullish/bearish/  │
       │                       neutral)          │
       │  · volume_confirmation enum             │
       │  · thesis  (free text)                  │
       │  · nearest_support / nearest_resistance │
       │  · confluence_score  [0.0 – 1.0]        │
       └──────────────────┬─────────────────────┘
                          │ AnalystOutput
                          ▼
       ┌────────────────────────────────────────┐
       │  HeadTrader  (agents/trader.py)         │
       │  Single-shot structured call            │
       │                                         │
       │  Output: HeadTraderOutput (Pydantic)    │
       │  · action  enum (long/short/flat/close) │
       │  · stop_loss_strategy  enum             │
       │  · take_profit_strategy  enum           │
       │  · risk_reward_ratio  float             │
       │  · reasoning_summary  (free text)       │
       └──────────────────┬─────────────────────┘
                          │ proposal dict
                          ▼
       ┌────────────────────────────────────────┐
       │  RiskManager  (brokers/risk.py)         │
       │  Pure Python / Decimal arithmetic       │
       │  - Drawdown circuit breaker (15%)       │
       │  - Max concurrent positions (5)         │
       │  - ATR-based stop/TP price resolution   │
       │  - Position sizing (1% equity risk)     │
       │  - 50% max per-asset exposure cap       │
       │                                         │
       │  Output: entry_price · stop_price       │
       │          take_profit_price · size_usd   │
       └──────────────────┬─────────────────────┘
                          │ approved order
                          ▼
       ┌────────────────────────────────────────┐
       │  PaperBroker  (brokers/paper.py)        │
       │  (or CoinbaseBroker in LIVE_SANDBOX)    │
       │  Tracks positions, OCO simulation,      │
       │  PnL settlement                         │
       └──────────────────┬─────────────────────┘
                          │ closed_trades / state
                          ▼
       ┌────────────────────────────────────────┐
       │  Supabase Postgres  (data/db.py)        │  ← transactional store
       │  Tables: portfolio_state · open_        │
       │  positions · trades · decision_log      │
       │  · llm_cost_log                         │
       └────────────────────────────────────────┘

  Web Layer (web/main.py + web/):
  ┌────────────────────────┐   ┌────────────────────────┐
  │  FastAPI  :8008        │   │  Next.js  :3001         │
  │  /api/status           │◄──│  Dashboard              │
  │  /api/metrics          │   │  (equity curve, costs,  │
  │  /api/positions        │   │   positions, decisions) │
  │  /api/trades           │   └────────────────────────┘
  │  /api/decisions        │
  │  /api/costs            │
  │  /api/candles          │
  │  POST /api/trigger-tick│
  └────────────────────────┘
```

---

## 1. The Stochastic / Deterministic Split

The central design principle is that **LLM agents produce qualitative decisions; Python
computes all numbers.**

The LLMs never see raw floats in their outputs — they emit `Literal` Pydantic enums:

| Layer | What the LLM emits | Type |
|---|---|---|
| Analyst | `market_bias` | `"bullish" \| "bearish" \| "neutral"` |
| Analyst | `volume_confirmation` | `"confirmed" \| "divergent" \| "weak"` |
| Trader | `action` | `"long" \| "short" \| "flat" \| "close"` |
| Trader | `stop_loss_strategy` | `"1.5_atr" \| "2.0_atr" \| "swing_low" \| "tight_atr"` |
| Trader | `take_profit_strategy` | `"3.0_atr" \| "4.0_atr" \| "next_resistance" \| "risk_reward_multiplier"` |

The **RiskManager** (`brokers/risk.py`) then resolves those strategy labels into exact
`Decimal` prices using TA-Lib ATR on the 4h candles and the FeaturePipeline's support/
resistance levels. All arithmetic uses Python's `Decimal` type to avoid floating-point
rounding errors.

Rationale: LLMs are stochastic text predictors. Asking an LLM to output `"stop_loss_price":
61834.72" from a raw price series produces hallucinated or inconsistent arithmetic.
Asking it to choose between a small enum of validated strategies, then delegating the
math to deterministic Python, eliminates an entire class of numeric bugs while keeping
the decision itself interpretable.

The feature pipeline (`features/pipeline.py`) applies the same principle to inputs:
raw indicator values (RSI 14 → `"overbought"`, MACD histogram → `"bullish_momentum_expanding"`)
are translated to categorical regime labels before being passed to the LLM, so the model
votes on regimes rather than arithmetic on floats.

---

## 2. Agent Layer

### TechnicalVolumeAnalyst (`agents/analyst.py`)

**Tool-use loop path (production):**

1. `LLMClient.call_llm_with_tools()` is invoked with `ANALYST_TOOLS` (6 JSON-schema tool
   definitions) and a `ToolExecutor` instance.
2. The loop runs up to 10 iterations (`max_iterations=10`). Each turn:
   - LiteLLM sends the conversation to the model with `tool_choice="auto"`.
   - If the model emits `tool_calls`, `ToolExecutor.execute()` dispatches to the
     appropriate Python handler and appends the result as a `"tool"` role message.
   - If the model stops emitting `tool_calls`, its `assistant.content` is returned as
     the raw JSON string.
3. `_extract_json()` strips optional markdown code fences (some models wrap output in
   ` ```json ``` `) before `json.loads`.
4. The result is validated into `AnalystOutput` (Pydantic).

**Legacy snapshot path (used by eval harness):**

When `TechnicalVolumeAnalyst` is constructed without `db` and `fetcher` (as in
`eval/runner.py`), or when an explicit `snapshot` dict is passed, the analyst falls
back to a single-shot `call_llm()` with `response_format=AnalystOutput`. The snapshot
is the pre-computed dict from `FeaturePipeline.run()`.

**Known divergence:** The eval harness always exercises the snapshot/legacy path
(`TechnicalVolumeAnalyst(db=None, fetcher=None)` in `runner.run_case()`), while
production uses the tool-loop path. This means eval scores measure the analyst's
ability to synthesize a pre-computed feature dict, not its tool-selection and
data-retrieval reasoning. The two paths share the same prompt and output schema, so
improvements to the bias/volume-confirmation methodology are still validated by eval.

### The 6 Analyst Tools (`agents/tools.py`)

All tools are dispatched by `ToolExecutor`, which holds a `Database`, a `DataFetcher`,
and an internal `FeaturePipeline`. The `current_timestamp` is pinned before each
tool-loop call to prevent look-ahead bias during backtests.

| Tool | Handler | Data source |
|---|---|---|
| `get_candles` | `_get_candles` | DuckDB `candles` table |
| `get_indicators` | `_get_indicators` | FeaturePipeline (TA-Lib on DuckDB candles) |
| `get_support_resistance` | `_get_support_resistance` | FeaturePipeline (`scipy.signal.find_peaks`) |
| `get_candlestick_patterns` | `_get_candlestick_patterns` | FeaturePipeline (TA-Lib CDLENGULFING etc.) |
| `get_derivatives` | `_get_derivatives` | `DataFetcher.fetch_funding_rate_and_oi()` → Binance Futures CCXT |
| `get_market_sentiment` | `_get_market_sentiment` | alternative.me Fear & Greed API |

### HeadTrader (`agents/trader.py`)

Single-shot `call_llm()` with `response_format=HeadTraderOutput`. Receives:
- The `AnalystOutput` as a JSON dict.
- The current market price (used to evaluate S/R proximity within 2%).
- A historical accuracy scorecard dict.
- Current open positions.

The House Methodology embedded in the system prompt deterministically maps S/R proximity
to stop-loss enum (`swing_low` when support is within 2% below entry for longs; `1.5_atr`
otherwise) and take-profit enum (`next_resistance` for longs; `3.0_atr` for shorts).
The LLM applies the methodology but makes the final call.

### Pydantic Contracts

Both `AnalystOutput` and `HeadTraderOutput` are Pydantic `BaseModel` subclasses.
Every field uses `Literal` types (for categoricals) or `float` with doc descriptions.
Structural output is enforced via `response_format=<ModelClass>` in `litellm.completion`,
which maps to OpenAI's structured-output API on supported models.

---

## 3. Multi-Provider LLM Client (`agents/client.py`)

`LLMClient` wraps LiteLLM and provides two call surfaces:

- **`call_llm()`** — single turn with optional `response_format` for structured output.
- **`call_llm_with_tools()`** — agentic loop up to `max_iterations=10`.

**Provider routing:** The `_LITELLM_PROVIDER_PREFIXES` dict maps the `LLM_PROVIDER` env
var (`gemini | openai | anthropic | groq | ollama`) to a LiteLLM model prefix string.
The resulting model id is `{prefix}/{model}` (e.g. `gemini/gemma-4-31b-it`). Unknown
providers pass the model string verbatim to LiteLLM's auto-detection.

**Per-agent model overrides:** `TechnicalVolumeAnalyst` reads
`{PROVIDER}_ANALYST_MODEL`; `HeadTrader` reads `{PROVIDER}_TRADER_MODEL`. Both fall
back to `LLM_MODEL` if unset.

**Throttle:** A class-level `_throttle_lock` and `_last_call_at` timestamp gate all
instances globally. `LLM_MIN_CALL_INTERVAL_SECONDS` controls the minimum spacing.
Production defaults to `0` (no throttle). The eval harness sets `4.5s` to stay under
Gemini's 15 RPM free-tier limit.

**Retry:** 5 attempts with exponential backoff (4 → 8 → 16 → 32 → 60s) via `tenacity`.

**Cost capture:** After every call (both single and tool-loop), `_emit_cost()` calls
the class-level `_cost_sink`. In production this is a `PostgresCostLogger`; in
tests/eval it is `None` (no-op).

---

## 4. Data Layer

### DuckDB — Read Cache (`data/db.py`, `Database`)

Single-file embedded database at `data/vibe_trading.db` (configurable via
`DATABASE_PATH`). Used as a time-series read cache for candles and computed features.

Tables:
- **`candles`** — OHLCV data keyed on `(symbol, timeframe, timestamp)`.
- **`features`** — Pre-computed indicator snapshot keyed on `(symbol, timestamp)`.
- **`portfolio_state`** — Balance and peak-balance history (also written by PaperBroker).
- **`open_positions`** — Active paper-trading positions (PaperBroker's durability store).
- **`trades`** — Closed trade history with PnL.
- **`decision_log`** — Every analyst+trader decision with `agent_transcripts` (the
  FeaturePipeline snapshot JSON).

The `Database` class connects/disconnects on every use. DuckDB does not allow multiple
writers; the code pattern is always `connect() → query → close()` to minimize lock
hold time. Connection retries with exponential backoff handle transient `IOException`
during lock contention between the bot process and the API process.

### Supabase Postgres — Transactional State (`data/db.py`, `PostgresDatabase`)

Postgres is the authoritative store for live state and audit data. Uses a psycopg2
`ThreadedConnectionPool(1, 15)` shared as a class-level singleton.

Tables mirror the DuckDB schema for state tables (portfolio_state, open_positions, trades,
decision_log) plus:
- **`llm_cost_log`** — Per-call LLM cost and token usage.

### `PostgresConnectionWrapper` — Dialect Translation

Since the codebase was written against DuckDB's SQL dialect (`?` placeholders,
`INSERT OR IGNORE`, `INSERT OR REPLACE`), a thin wrapper (`translate_query()` +
`PostgresConnectionWrapper`) rewrites SQL on the fly before executing it against psycopg2:
- `?` → `%s`
- `INSERT OR IGNORE INTO decision_log` → `INSERT INTO ... ON CONFLICT (decision_id) DO NOTHING`
- `INSERT OR IGNORE INTO llm_cost_log` → `INSERT INTO ... ON CONFLICT (call_id) DO NOTHING`
- `INSERT OR REPLACE INTO open_positions` → `INSERT INTO ... ON CONFLICT (symbol) DO UPDATE SET ...`

This lets the business logic (scheduler, broker, web API) write a single SQL string
that works against both backends.

### `DataFetcher` (`data/fetcher.py`)

Wraps two CCXT Binance clients: spot (OHLCV, trending tickers) and futures (funding
rate, open interest). Key methods:

- **`bootstrap()`** — Initial warm-up: 2 years of 1D candles, 6 months of 4H candles,
  paginated in 1000-candle batches.
- **`bootstrap_if_needed()`** — Checks candle count per symbol/timeframe; bootstraps
  only missing pairs (threshold: 200 candles).
- **`incremental_update()`** — Fetches the latest `limit=15` candles per symbol/timeframe
  at each tick. Deliberately separates network fetch from DB write to keep DB lock time
  short.
- **`fetch_trending_symbols()`** — Tries CoinGecko `/search/trending` first; falls back
  to Binance top-USDT-volume sort; last resort: static 5-symbol list.

### `FeaturePipeline` (`features/pipeline.py`)

Computes indicators on demand from DuckDB candles. Used both by the prod scheduler
(to build the `snapshot` dict for the decision log) and by `ToolExecutor` (for the
analyst's on-demand tool calls).

Indicators computed via TA-Lib: SMA(20/50/200), RSI(14), MACD(12/26/9), ADX(14), OBV.
Support/resistance via `scipy.signal.find_peaks` on highs/lows with `distance=10`
and `prominence=1%` of mean.
Candlestick patterns via TA-Lib: CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR,
CDLHAMMER, CDLSHOOTINGSTAR.

---

## 5. Eval Harness (`eval/`)

### Golden Set

20 hand-labeled YAML cases under `evals/snapshots/`. Each case (`EvalCase`) contains:
- A symbol + timestamp pointing to real DuckDB candle history.
- `analyst_label` with expected `market_bias`, `volume_confirmation`, `nearest_support`,
  `nearest_resistance`, `confluence_score`, and a `thesis_rubric`.
- `trader_label` with expected `action`, strategy enums, `risk_reward_ratio`, and a
  `reasoning_rubric`.

Cases were generated from real candle history via `evals/scan_candidates.py` (regime
bucketing) and labeled by `evals/build_golden_set.py` (deterministic Murphy-rule voting).

### Runner (`eval/runner.py`)

`run_case()` builds a `FeaturePipeline` snapshot, runs the analyst on it (snapshot/
legacy path — see divergence note below), then runs the trader fed the **labeled**
analyst output (not the actual analyst output). This isolates trader regressions from
analyst regressions.

**Known eval/prod divergence:** `run_case()` constructs `TechnicalVolumeAnalyst(db=None,
fetcher=None)`, which forces the legacy snapshot path. Production always uses the
tool-use loop path. The two paths share the same bias-selection methodology and output
schema, so prompt changes that affect bias/volume-confirmation logic are covered. What
the eval does *not* cover is tool-selection behavior, tool result interpretation, or
multi-turn reasoning in the analyst.

### Scorer (`eval/scorer.py`)

Two scoring strategies:
- **`score_categorical()`** — Exact match for enum fields (`market_bias`, `action`,
  `stop_loss_strategy`, `take_profit_strategy`).
- **`score_numeric_tolerance()`** — Linear degradation between `ok` and `zero`
  thresholds. For S/R levels: `ok=2%`, `zero=5%` (percentage of expected). For
  `confluence_score`: `ok=0.15`, `zero=0.30` (absolute).
- **`score_rubric()`** — LLM-as-judge for free-text fields (`thesis`, `reasoning_summary`).
  Passes the actual text and the case's `must_mention` / `must_not_mention` rubric to
  a separate `build_judge()` LLM call. Score = fraction of criteria passed.

`stop_loss_strategy`, `take_profit_strategy`, and `risk_reward_ratio` are only scored
when the label action is not `"flat"` (a flat label has no entry strategy to validate).
`hold_period_bias` is excluded from scoring entirely — the golden-set default of
`"medium"` is a hardcoded heuristic, not a derivable signal.

### Baseline and CI

`evals/baseline.json` — committed regression yardstick. Current baseline:
`overall_score=0.867`, `analyst_score=0.956`, `trader_score=0.731` (22 cases,
produced on `gemma-4-31b-it`). `eval.py` exits non-zero when the current run's score
regresses below baseline. `--update-baseline` overwrites it after a reviewed improvement.

The harness runs cases in parallel (`ThreadPoolExecutor`, default 6 workers) with the
class-level `LLMClient` throttle bounding the actual RPM regardless of worker count.

---

## 6. Cost Tracking and Guardrails (`agents/cost.py`)

### LLM Cost Logging

Every `call_llm()` and `call_llm_with_tools()` call in `LLMClient._emit_cost()`:
1. Reads token usage from the LiteLLM response.
2. Calls `usage_cost()`: LiteLLM's `get_model_info()` first; `PRICE_OVERRIDES` dict
   as a fallback for free-tier Gemma models LiteLLM doesn't price.
3. Calls `PostgresCostLogger.record()` → inserts a `CostEvent` into `llm_cost_log`.

Cost logging is best-effort: exceptions are caught and logged as warnings so a logging
failure never interrupts a trade.

### Daily Summary and Alarms

`daily_summary(conn)` aggregates `llm_cost_log` since UTC midnight: total USD, call
count, tokens, average cost per call, projected $/month, per-model breakdown.

The scheduler calls `_check_cost_alarm()` at the top of every tick:
- If today's spend > `LLM_DAILY_COST_ALARM_USD` (default $5) and no alarm was sent
  today, fires a Discord webhook notification. Tracks alarm state in
  `_cost_alarmed_on: date` to send once per UTC day.

The scheduler calls `_trading_blocked_by_cost()` before any LLM calls:
- If today's spend >= `LLM_DAILY_COST_CAP_USD` (default $10; disable with `<= 0`),
  returns `True` and blocks new-entry evaluation for the rest of the UTC day.
- Existing open positions continue to be managed (SL/TP simulation is deterministic
  and requires no LLM calls).
- Fail-open: a DB read error returns `False` (never halts trading on accounting hiccup).

Named control: `should_block_trading(today_usd, cap_usd)` in `cost.py` — a
single-responsibility, testable function.

### RiskManager Guardrails (`brokers/risk.py`)

| Guardrail | Value | Enforcement |
|---|---|---|
| Max drawdown circuit breaker | 15% from peak balance | Vetoes new entries; existing positions continue |
| Max concurrent positions | 5 | Scheduler skips evaluation when portfolio is full |
| Max per-asset exposure | 50% of balance | Caps raw position size before submission |
| Risk per trade | 1% of current balance | Drives position sizing: `size = risk_amount / stop_distance_pct` |
| Dust filter | $10 minimum position | Rejects positions below $10 |
| Fee + slippage buffer | 0.5% (0.4% maker + 0.1% slip) | Applied as haircut to position size |

All arithmetic is done in `Decimal` to avoid float rounding in PnL and size calculations.

---

## 7. Observability

**Langfuse tracing:** `@observe()` decorators from the `langfuse` library wrap:
- `TradingScheduler.sync_and_evaluate()` — top-level tick trace.
- `TechnicalVolumeAnalyst.analyze()` — per-symbol analyst trace.
- `HeadTrader.decide()` — per-symbol trader trace.

`propagate_attributes()` context managers annotate traces with symbol tags and metadata.

**Discord alerts:** `TradingScheduler._send_discord_alert()` via `DISCORD_WEBHOOK_URL`
on: new trade entered, trade closed (with PnL), risk veto, LLM cost alarm, cost cap
reached, scheduler tick errors.

**Web dashboard:** `GET /api/costs` → `daily_summary()` from Postgres. The API
process accesses Postgres via the shared `ThreadedConnectionPool`; DuckDB is read
with `read_only=True` connections.

---

## 8. End-to-End Decision Flow (One Symbol)

```
1. Scheduler tick fires (APScheduler, every 4h)
   └── TradingScheduler.sync_and_evaluate()  [scheduler.py:70]

2. DataFetcher.incremental_update()  [fetcher.py:114]
   └── Fetches latest 15 candles (4h + 1d) from Binance via CCXT
   └── Writes to DuckDB candles table — short connect/close

3. PaperBroker.update_positions()  [paper.py:155]
   └── Checks SL/TP hit for each open position at latest close price
   └── Settled trades → written to Postgres `trades` table
   └── Discord "TRADE CLOSED" alert if any hit

4. Cost gate: _trading_blocked_by_cost()  [scheduler.py:275]
   └── Reads today's llm_cost_log from Postgres
   └── Returns True (skip) if today_usd >= LLM_DAILY_COST_CAP_USD

5. TechnicalVolumeAnalyst.analyze(symbol, timestamp)  [analyst.py:113]
   └── ToolExecutor.set_timestamp(last_candle_ts)
   └── LLMClient.call_llm_with_tools() — up to 10 turns:
       ├── Analyst calls get_candles("4h"), get_candles("1d")
       ├── Analyst calls get_indicators("4h"), get_indicators("1d")
       ├── Analyst calls get_support_resistance()
       ├── Analyst calls get_candlestick_patterns()
       ├── Analyst calls get_derivatives() → Binance Futures CCXT
       ├── Analyst calls get_market_sentiment() → alternative.me API
       └── Analyst emits final AnalystOutput JSON
   └── Returns AnalystOutput (Pydantic-validated)

6. FeaturePipeline.run(symbol, timestamp)  [pipeline.py:17]
   └── Builds deterministic snapshot dict (for decision_log audit)
   └── Uses DuckDB candles — short connect/close

7. HeadTrader.decide(symbol, analyst_output, ...)  [trader.py:84]
   └── LLMClient.call_llm() — single-shot, response_format=HeadTraderOutput
   └── Applies House Methodology (S/R proximity → stop enum selection)
   └── Returns proposal dict: action, strategy enums, risk_reward_ratio

8. Postgres: INSERT INTO decision_log  [scheduler.py:194]
   └── decision_id, symbol, action, strategies, reasoning_summary,
       agent_transcripts (snapshot JSON)

9. if proposal.action == "flat": continue to next symbol

10. RiskManager.evaluate_proposal()  [risk.py:36]
    ├── Drawdown circuit breaker check
    ├── Concurrent-position cap check
    ├── ATR(14) from 4h candles (talib.ATR)
    ├── Resolves stop_loss_strategy → exact stop_price (Decimal)
    ├── Resolves take_profit_strategy → exact take_profit_price (Decimal)
    └── Sizes position: size_usd = (balance × 1%) / stop_distance_pct

11. if risk_res.approved:
    └── PaperBroker.submit_order()  [paper.py:113]
        ├── Records position in memory + DuckDB open_positions
        ├── Fills immediately at risk_res["entry_price"]
        └── Discord "NEW TRADE ENTERED" alert

12. if not approved:
    └── Discord "RISK VETO" alert + log warning
```

---

## 9. Deployment

Docker Compose services (`docker-compose.yml`):

| Service | Container | Port | Command |
|---|---|---|---|
| `vibe-bot` | `vibe_trading_bot` | — | `python -m vibe_trading.cli live` — blocking 4h scheduler |
| `vibe-api` | `vibe_trading_api` | **8008 → 8000** | `uvicorn vibe_trading.web.main:app --reload` |
| `vibe-web` | `vibe_trading_web` | **3001 → 3000** | `npm run dev` (Next.js; proxies API at `http://vibe-api:8000`) |
| `backtester` | `vibe_backtester` | — | `python -m vibe_trading.cli backtest --start <date>` (one-shot) |
| `trade-once` | `vibe_trade_once` | — | `python -m vibe_trading.cli trade-once` (one-shot) |

All services share a bind-mounted `./data` volume so the single-file DuckDB database
(`data/vibe_trading.db`) is visible to both the bot and the API. The bot writes DuckDB
as writer; the API opens it as `read_only=True`. Both connect to the same Supabase
Postgres instance via `POSTGRES_URL`.

Environment is loaded from `.env` (`TRADING_MODE`, `LLM_PROVIDER`, `LLM_MODEL`,
`GEMINI_API_KEY`, `POSTGRES_URL`, `LANGFUSE_SECRET_KEY`, `DISCORD_WEBHOOK_URL`,
`LLM_DAILY_COST_ALARM_USD`, `LLM_DAILY_COST_CAP_USD`).

Live Coinbase trading is gated behind `TRADING_MODE=LIVE_SANDBOX`, which swaps
`PaperBroker` for `CoinbaseBroker`. The default is `PAPER`.

---

## Key Source Files Reference

| File | Role |
|---|---|
| `src/vibe_trading/runtime/scheduler.py` | Main tick loop, cost gates, Discord alerts |
| `src/vibe_trading/agents/analyst.py` | TechnicalVolumeAnalyst, AnalystOutput schema |
| `src/vibe_trading/agents/trader.py` | HeadTrader, HeadTraderOutput schema |
| `src/vibe_trading/agents/client.py` | LLMClient, tool-use loop, throttle, cost emit |
| `src/vibe_trading/agents/tools.py` | ANALYST_TOOLS definitions, ToolExecutor handlers |
| `src/vibe_trading/agents/cost.py` | CostEvent, PostgresCostLogger, daily_summary, guardrail fns |
| `src/vibe_trading/brokers/risk.py` | RiskManager — all Decimal price/size math |
| `src/vibe_trading/brokers/paper.py` | PaperBroker — position state, OCO simulation |
| `src/vibe_trading/data/db.py` | Database (DuckDB), PostgresDatabase, dialect translation |
| `src/vibe_trading/data/fetcher.py` | DataFetcher — CCXT Binance, CoinGecko |
| `src/vibe_trading/features/pipeline.py` | FeaturePipeline — TA-Lib, scipy S/R, candlestick |
| `src/vibe_trading/eval/eval.py` | Eval CLI entry point, parallel runner |
| `src/vibe_trading/eval/runner.py` | run_case(), EvalCase / CaseResult models |
| `src/vibe_trading/eval/scorer.py` | Deterministic + LLM-as-judge scoring |
| `src/vibe_trading/web/main.py` | FastAPI REST endpoints |
| `evals/snapshots/` | 20 golden-set YAML cases |
| `evals/baseline.json` | Committed regression yardstick (overall 0.867) |
| `docker-compose.yml` | Service definitions and port mappings |
