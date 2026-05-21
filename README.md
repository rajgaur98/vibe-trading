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
GEMINI_API_KEY=your_google_ai_studio_api_key
DISCORD_WEBHOOK_URL=your_discord_webhook_url
TRADING_MODE=PAPER
```

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

---

## Running Tests
Run unit verification tests using pytest:
```bash
PYTHONPATH=src pytest
```
