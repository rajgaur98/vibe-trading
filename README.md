# Vibe Trading 🌊📈

[Architecture overview](ARCHITECTURE.md)

A systematic crypto swing-trading bot powered by **Google Gemini multi-agent reasoning** and **deterministic Python risk management**. Designed to trade major cryptocurrencies (BTC/USD, ETH/USD) on a 4-hour timeframe using Google AI Studio's Free Tier ($0 operational cost).

---

## Key Features

1. **Multi-Agent Reasoning:**
   - **Technical & Volume Analyst (`gemini-3.5-flash`):** Evaluates indicators, moving average stacks, volume flows, and candlestick pattern confluences.
   - **Head Trader (`gemini-3.1-pro`):** Synthesizes analyst data and historical scorecards to propose structured trade setups.
2. **Zero-Hallucination Safe Rules:** Converts raw metrics to qualitative categories in Python before sending inputs to the LLM. 
3. **Deterministic Python Risk Manager:** Computes exact entry, stop-loss, and take-profit targets via ATR, capping risk at 1% of equity per trade (max 3 concurrent positions).
4. **Single-File DuckDB Storage:** Unified database for OHLCV history, features, trades, and agent decision logs.
5. **Evaluation Harness:** Backtests trading decisions chronologically with sub-candle (1m/5m) resolution to prevent bracket execution lookahead bias, generating `QuantStats` HTML performance reports.
6. **Observability & Alerts:** Uses Langfuse Cloud for LLM tracing and Discord webhooks for instant trade notifications.

---

## Directory Structure

```
├── data/                      # Database storage & reports
├── src/
│   └── vibe_trading/
│       ├── __init__.py
│       ├── cli.py             # CLI Entry Point
│       ├── agents/
│       │   ├── client.py      # Gemini API Wrapper
│       │   ├── analyst.py     # Technical Analyst
│       │   └── trader.py      # Head Trader
│       ├── brokers/
│       │   ├── base.py        # Abstract Broker
│       │   ├── paper.py       # Paper/Simulated Broker
│       │   ├── coinbase.py    # Coinbase Advanced Broker
│       │   └── risk.py        # Risk Manager
│       ├── data/
│       │   ├── db.py          # DuckDB Schema/Connection
│       │   └── fetcher.py     # CCXT Historical Fetcher
│       ├── eval/
│       │   └── backtest.py    # Backtest engine
│       ├── features/
│       │   └── pipeline.py    # Tech indicators & regimes
│       └── runtime/
│           └── scheduler.py   # recurring chron trading loops
├── tests/                     # Unit test suite
└── pyproject.toml             # Dependencies & packaging metadata
```

---

## Setup Instructions

### 1. Requirements
Ensure you have the following installed on your machine:
- Python 3.11+
- `ta-lib` (required for TA-Lib Python wrapper. On macOS, run `brew install ta-lib` first)

### 2. Installation
Install the project dependencies in your environment:
```bash
pip install -e .
```
Or if developing, install dev requirements:
```bash
pip install -e ".[dev]"
```

### 3. Environment Config
Copy the environment variables template and fill in your Gemini API Key:
```bash
cp .env.example .env
```
Open `.env` and enter your credentials:
```env
# Provider selection (default: gemini). LiteLLM routes everything through
# LLMClient, so you can swap providers via env vars without code changes.
LLM_PROVIDER=gemini                              # gemini | openai | anthropic | groq | ollama
LLM_MODEL=gemini-3.1-flash-lite                  # provider-native model id

# Optional per-agent overrides (only honored when LLM_PROVIDER=gemini).
# These take precedence over LLM_MODEL for the respective agent.
# GEMINI_ANALYST_MODEL=gemma-4-31b-it
# GEMINI_TRADER_MODEL=gemma-4-31b-it

# Provide the key for whichever provider you selected
GEMINI_API_KEY=your_google_ai_studio_api_key     # for LLM_PROVIDER=gemini
# GROQ_API_KEY=your_groq_console_api_key         # for LLM_PROVIDER=groq
# OPENAI_API_KEY=...                             # for LLM_PROVIDER=openai
# ANTHROPIC_API_KEY=...                          # for LLM_PROVIDER=anthropic

DISCORD_WEBHOOK_URL=your_discord_webhook_url
TRADING_MODE=PAPER
```

#### Choosing a model — free-tier rate vs token limits

Different free tiers fail in different ways. Check your provider's actual
usage dashboard (e.g. <https://aistudio.google.com/usage>) — published docs
often don't match what your account is actually provisioned. The eval suite
is the stress case: one full run is ~56 calls / ~280K tokens in a short burst.

| Option | Limits that bite | Best when |
|--------|------------------|-----------|
| **Gemini `gemma-4-31b-it`** (recommended) | 15 RPM, **Unlimited TPM**, 1,500 RPD | You want no token caps; bursty eval runs. Structured output + tool-calling both verified. |
| Gemini `gemini-3.1-flash-lite` | 15 RPM, 250K TPM, 500 RPD | Live scheduler (~250 calls/day fits easily); rock-solid structured output. |
| Gemini `2.5-flash` / `3.5-flash` | **20 RPD** | Avoid for eval — a single run exhausts the daily request cap. |
| Groq `llama-3.3-70b-versatile` | ~30 RPM, but a **daily token cap (~100K TPD)** | High request rate with *small* prompts. The TPD cap exhausts fast on this project's indicator-heavy prompts. |

**Recommended:** `LLM_PROVIDER=gemini` with `LLM_MODEL=gemma-4-31b-it`. Gemma 4
on AI Studio has **no per-minute token cap** and 1,500 requests/day — the
profile that suits this project's bursty, token-heavy eval workload. Only the
15 RPM rate limit applies, which `--throttle-seconds 5` keeps you under. Both
the trader's structured output and the analyst's tool-use loop are verified
working on it.

> **Note on Groq:** Groq's LPU hardware is genuinely fast (~280 tok/s) and its
> *request* limits are generous, but its free tier imposes a **daily token
> cap** that this project's ~5K-token analyst prompts blow through after a
> couple of eval runs. Prefer it only for high-request, low-token workloads.

---

## CLI Usage

### Bootstrap Historical Data
Populates 2 years of 1D candles and 6 months of 4h candles in DuckDB to warm up indicators:
```bash
python -m vibe_trading.cli bootstrap
```

### Run Backtesting Simulation
Runs historical backtesting over the selected range, exporting an HTML report to `data/reports/backtest_report.html`:
```bash
python -m vibe_trading.cli backtest --start 2026-01-01 --end 2026-05-01
```

### Start the Live Scheduler
Starts the blocking cron scheduler that runs evaluations every 4 hours:
```bash
python -m vibe_trading.cli live
```

### Run On-Demand Trade (Bypass Schedule)
Runs a single evaluation and execution window immediately:
```bash
python -m vibe_trading.cli trade-once
```

---

## Docker Usage 🐳

Using Docker allows running the bot and backtests without manually installing system-level dependencies like compiling `ta-lib` C libraries.

### 1. Build the Docker Image
```bash
docker compose build
```

### 2. Run Historical Data Bootstrap
```bash
docker compose run --rm vibe-bot bootstrap
```

### 3. Run Backtester
Runs the backtest and generates the HTML report in `data/reports/` (persisted on the host):
```bash
docker compose run --rm backtester
```

### 4. Run On-Demand Trade (Bypass Schedule)
Runs a single evaluation window immediately:
```bash
docker compose run --rm trade-once
```

### 5. Start the Live Scheduler Bot (Background)
Runs the live scheduler container in the background:
```bash
docker compose up -d vibe-bot
```

---

## Live Demo Execution (Binance USDⓂ Futures Demo Trading)

Set `TRADING_MODE=LIVE_TESTNET` to execute real orders on **Binance Futures Demo Trading**
(`demo.binance.com` → REST `demo-fapi.binance.com`) with native exchange brackets. On entry
the broker places a market order plus two `closePosition` orders — `TAKE_PROFIT_MARKET` and
`STOP_MARKET` — so the exchange fills whichever triggers first and cancels the sibling,
**even if the bot is offline** (verified live). Leverage is pinned to 1× so risk/sizing
semantics match the paper model. Futures (not spot) is used because the trader emits
**short** as well as long.

> Demo Trading is **not** the deprecated futures testnet (`testnet.binancefuture.com`) and
> not production. ccxt 4.5 dropped `set_sandbox_mode` for futures, so the broker routes the
> fapi URLs to `demo-fapi.binance.com` directly (override via `BINANCE_DEMO_FAPI_URL`). The
> demo TP/SL are **conditional orders** — read back with `fetch_open_orders(..., {'stop': True})`.

- **Setup:** create demo-trading API keys at `demo.binance.com` (API Management; enable
  Reading + Futures), put them in `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET`
  (names kept for back-compat — they hold demo keys), and set `TRADING_MODE=LIVE_TESTNET`.
- **Dry run:** `BINANCE_TESTNET_DRY_RUN=true` logs intended orders without placing any —
  a safe way to verify wiring before sending real testnet orders.
- **Dashboard:** in this mode `/api/positions` reads open positions **directly from the
  exchange** (always accurate), falling back to the Postgres ledger on any exchange error.
- **Bookkeeping:** the Postgres `open_positions` table is the reconciliation ledger; each
  tick compares it to live exchange positions and records any bracket-closed trade.
- **TA still uses spot candles** — only the execution price aligns to the futures mark.
- **Smoke test (manual):** `python scripts/binance_testnet_smoke.py` opens a tiny position
  with a bracket, prints it, and closes it (requires your testnet keys).

> **Real-time bookkeeping:** a User Data Stream websocket listener (ccxt.pro `watch_orders`,
> a daemon thread started with the scheduler) records bracket-closed trades to `trades` +
> Discord within seconds of the fill, instead of at the next 4h tick. The 4h reconcile
> remains the safety net, and an atomic ledger claim-delete prevents any double-recording.

---

## Running Tests
Run unit verification tests using pytest:
```bash
PYTHONPATH=src pytest
```

## Eval Harness

Score the analyst + trader prompts against the hand-labeled golden set under `evals/snapshots/`:

```bash
# Run the eval against the current baseline, prints summary, non-zero exit on regression
uv run python -m vibe_trading.eval.eval

# After a prompt change you've reviewed and approved:
uv run python -m vibe_trading.eval.eval --update-baseline

# Override the judge model (defaults to your configured LLM_MODEL — a different
# family here gives cross-model-family bias mitigation). Any LiteLLM id works.
EVAL_JUDGE_MODEL=claude-3-5-haiku-20241022 uv run python -m vibe_trading.eval.eval

# Increase the per-case throttle if you hit provider RATE limits
# (default 3s; the committed baseline was produced on Gemma 4 31B at 5s)
uv run python -m vibe_trading.eval.eval --throttle-seconds 5

# Evaluate the SAME tool-use path production runs (slower, more LLM calls).
# Default is the fast snapshot path (what the committed baseline measures);
# the tool-loop path produces different scores and needs its own --update-baseline.
uv run python -m vibe_trading.eval.eval --analyst-path tool-loop
```

**Eval vs prod path.** By default the eval exercises the analyst's fast single-call
*snapshot* path (cheap, deterministic — the regression-gate default the committed
baseline is measured on). Production runs the multi-turn *tool-use* path. To verify
exactly what ships, run `--analyst-path tool-loop`; it makes ~6× the LLM calls per case
(much slower on a high-latency model) and yields different scores, so seed it with its
own `--update-baseline` rather than comparing against the snapshot baseline.

The judge defaults to whatever `LLM_MODEL` is set to, so switching `LLM_PROVIDER`
flips all four call sites (analyst, trader, and both judges) together — no risk of
the judge pointing at a model the active provider doesn't host. Set
`EVAL_JUDGE_MODEL` only when you deliberately want a different judge.

The golden-set YAMLs live in `evals/snapshots/` — 14 real cases derived from DuckDB
candle history via `evals/scan_candidates.py` (regime bucketing) and labeled by
`evals/build_golden_set.py` (deterministic Murphy-rule voting). Both generator scripts
are committed so the derivation logic is reviewable; rerun them after curating new
candidate timestamps.

Reports land in `data/reports/eval-<timestamp>.json` (gitignored). The regression
yardstick is `evals/baseline.json`, committed to git so prompt-impact diffs are
reviewable in PRs. The current baseline was produced on `gemma-4-31b-it` (chosen for
its unlimited TPM — see the model-selection table above). To seed the baseline on a
fresh checkout, run with `--update-baseline` once. If your provider has a tight
**token** cap (e.g. Groq's TPD), prefer Gemma; if it has a tight **request** cap,
raise `--throttle-seconds` so the per-minute bucket has time to refill.

## Cost Tracking

Every LLM call's tokens, dollar cost, and latency are logged to the `llm_cost_log`
Postgres table (capture happens in `LLMClient`, the single call chokepoint — so eval and
prod share the same accounting code). Cost is computed from LiteLLM's model pricing, with
a shadow-price fallback (`PRICE_OVERRIDES` in `agents/cost.py`) for models LiteLLM doesn't
price — e.g. the free-tier Gemma models — so projected $/month stays meaningful.

- **Dashboard:** a cost tile shows today's spend, projected $/month, and call count.
- **API:** `GET /api/costs` returns the daily summary + per-model breakdown.
- **Alarm:** the scheduler sends a Discord alert once per UTC day when spend exceeds
  `LLM_DAILY_COST_ALARM_USD` (default $5) — a warning.
- **Kill switch (hard cap):** once today's spend reaches `LLM_DAILY_COST_CAP_USD`
  (default $10; set `<= 0` to disable), the scheduler **blocks new-entry evaluation**
  (the expensive analyst/trader LLM calls) for the rest of the UTC day, stopping further
  spend. Existing positions keep being managed (deterministic SL/TP, no LLM). The gate
  is the named, testable `should_block_trading()` control, enforced before the LLM calls
  so it actually saves cost. Fail-open: a spend-read error never halts trading.

Cost tracking itself is observational — a logging failure never interrupts a trade.

