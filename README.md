# Vibe Trading 🌊📈

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

# Provide the key for whichever provider you selected
GEMINI_API_KEY=your_google_ai_studio_api_key     # for LLM_PROVIDER=gemini
# GROQ_API_KEY=your_groq_console_api_key         # for LLM_PROVIDER=groq (free tier, faster, way higher rate limits)
# OPENAI_API_KEY=...                             # for LLM_PROVIDER=openai
# ANTHROPIC_API_KEY=...                          # for LLM_PROVIDER=anthropic

DISCORD_WEBHOOK_URL=your_discord_webhook_url
TRADING_MODE=PAPER
```

**Recommended for fewer rate-limit headaches:** set `LLM_PROVIDER=groq` +
`LLM_MODEL=llama-3.3-70b-versatile` (a 70B open-weights model on Groq's LPU
hardware — ~280 tok/s, native tool-calling, and the free tier comfortably
covers both the live scheduler and the full eval suite).

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

# Use a different judge model (any LiteLLM-compatible identifier)
EVAL_JUDGE_MODEL=claude-3-5-haiku-20241022 uv run python -m vibe_trading.eval.eval

# Increase the per-case throttle if you hit provider rate limits
# (default 3s; raise to 8-10s for rate-limited tiers)
uv run python -m vibe_trading.eval.eval --throttle-seconds 10
```

The golden-set YAMLs live in `evals/snapshots/` — 14 real cases derived from DuckDB
candle history via `evals/scan_candidates.py` (regime bucketing) and labeled by
`evals/build_golden_set.py` (deterministic Murphy-rule voting). Both generator scripts
are committed so the derivation logic is reviewable; rerun them after curating new
candidate timestamps.

Reports land in `data/reports/eval-<timestamp>.json` (gitignored). The regression
yardstick is `evals/baseline.json`, committed to git so prompt-impact diffs are
reviewable in PRs. To seed the baseline on a fresh checkout, run with
`--update-baseline` once — and use `--throttle-seconds 10` if your LLM provider's
rate limit is tight, otherwise the first few cases may fail with `RateLimitError`
and pollute the baseline.

