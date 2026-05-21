# Vibe Trading — Gemini Design Spec (v2)

**Date**: 2026-05-20
**Author**: Raj (with structural optimization updates)
**Status**: Approved design draft

## Purpose

Build a paper-trading system for crypto swing trades that uses Google Gemini (via Google AI Studio) as a multi-agent reasoner over Murphy-style technical-analysis features. The dual goal:

1. **Trading goal** — system that, when graduated to real money, demonstrably trades better than buy-and-hold over a 3-month paper window.
2. **Learning goal** — exercise production AI engineering muscles end-to-end: structured outputs, agent orchestration, prompt caching, evals, observability, cost tracking, guardrails, and reproducible backtests.

The learning goal is primary; the trading goal is the forcing function that makes the learning real.

---

## Scope (v1)

### In scope

- Paper trading on BTC/USD and ETH/USD using 4h candles for decisions and 1D candles for trend context.
- Deterministic feature pipeline computing trend, momentum, volume, market structure, chart patterns, and candlestick patterns.
- Multi-agent LLM reasoning system (Technical & Volume Analyst, Head Trader) with deterministic Python-based Risk Manager.
- Backtest harness that replays historical candles through the exact same pipeline used in live paper, resolving OCO orders via sub-candle (1m/5m) intervals.
- Eval harness with golden datasets, TimeSeriesSplit, and QuantStats performance reports.
- Observability via Langfuse Cloud (Free Tier) + append-only decision audit log.
- Streamlit dashboard, Discord alerts.
- A `CoinbaseBroker` implemented and tested against the sandbox with client-side OCO bracket order emulation, gated off in v1 by `LIVE_TRADING_ENABLED=false`.

### Out of scope (v1)

- Real-money execution (gated until graduation criteria pass — see Component 8).
- ML-based pattern detection (designed in as a future swap; rule-based only at v1).
- Heavy news/social sentiment NLP engines (replaced by lightweight macro/derivatives event triggers).
- Order types beyond market entries with bracket OCO stop + take-profit.
- Multi-exchange smart routing.
- Markets other than crypto.
- Sub-4h timeframes for decisions.

---

## Architecture

```
[Data Layer] ──► [Feature Pipeline] ──► [Analyst Agent] ──► [Head Trader] ──► [Risk Manager] ──► [Execution]
                                                                                   │
                                                                                   ▼
               [Observability + Eval Harness] ◄─────────────────────────────── [Outcomes]
```

**Key invariant**: the same `MarketSnapshot → Decision` function works in backtest, live paper, and (eventually) real money. No mode-specific forks. This is what makes evals trustworthy and graduation meaningful.

---

## Component 1 — Data Layer

- **Source**: `ccxt` unified API. Coinbase Advanced primary; Binance secondary for deeper OHLCV history.
- **Universe**: BTC/USD, ETH/USD. Expandable via config.
- **Timeframes**: 4h (decision cadence), 1D (trend bias context).
- **Storage**: DuckDB single-file DB for OHLCV + computed features. Parquet exports for portable backtest replay.
- **Scheduler**: Powered by `APScheduler` to run the pipeline reliably at every 4h candle close, eliminating timing drift and handling API retry loops.
- **Bootstrap**: pull ≥ 2 years of 1D and ≥ 6 months of 4h on first run.

---

## Component 2 — Feature Pipeline

Pure-Python, deterministic. All indicators are pre-computed in vectorized format across the historical DB before backtests run, eliminating iterative pipeline overhead. 

### Numeric Hallucination Safety Rule
To prevent LLM mathematical reasoning errors, **all raw numerical metrics are categorized into qualitative regimes in Python** before being passed to the agents.

- **Trend**: MA20/50/200 stack, ADX (trend strength), price-vs-MA position, 1D-timeframe trend bias.
  * *Categorization:* Raw prices are converted to relative terms (e.g., `price_vs_ma200: "above"`, `ma_stack_status: "bullish_alignment"`).
- **Momentum**: RSI14, MACD (12/26/9), Stochastic (14/3/3) (calculated using `ta-lib`).
  * *Categorization:* Raw values mapped to bands (e.g., `rsi_14: { "value": 72.0, "regime": "overbought" }`).
- **Volume**: OBV trend, volume MA20, volume spike flags.
  * *Categorization:* Volume trend mapped to `volume_confirmation: "weak"` or `"strong"`.
- **Structure**: swing highs/lows, support/resistance via fractal pivots (calculated using `scipy.signal.find_peaks` prominence).
  * *Categorization:* Support/resistance levels passed with percentage distance relative to price (e.g., `nearest_support: { "price": 67200.0, "distance_pct": 1.2, "proximity": "very_close" }`).
- **Patterns**: Rule-based detectors for head-and-shoulders, double tops/bottoms, triangles, flags.
- **Candlesticks**: Pin bars, engulfing, morning/evening stars near structural levels, recognized using `ta-lib` C-bindings.
- **Derivatives & Macro FA Context**:
  * *Funding Rates:* Categorized into risk regimes (e.g., `funding_rate: "0.08% (extremely_high_long_crowding)"`).
  * *Open Interest (OI) Change:* Categorized into trend indicators (e.g., `open_interest_trend: "rising_capital_inflow"`).
  * *Macro Event Calendar:* Binary flag mapping major macro calendar events (`is_macro_event_today: True/False`).

**Output**: A Pydantic v2 `MarketSnapshot` per symbol per candle close. The schema is the contract — agents only ever see this categorized snapshot, never raw candle arrays.

---

## Component 3 — Multi-Agent Reasoning

Two agents running on Google AI Studio Free Tier.

| Agent | Model | Reads | Outputs |
|---|---|---|---|
| **Technical & Volume Analyst** | `gemini-3.5-flash` | Trend + momentum + structure + volume + macro features | Bias (bullish/bearish/neutral), volume confirmation strength, thesis, structural zones |
| **Head Trader** | `gemini-3.1-pro` | Analyst outputs + deterministic agent scorecard + open positions | Final qualitative `TradeProposal`: action (long/short/flat), stop_loss_strategy, take_profit_strategy, risk_reward_ratio |

- **Orchestration**: Linear pipeline. The `Technical & Volume Analyst` outputs its analysis, which is combined with a Python-calculated agent accuracy scorecard (`accuracy_last_20_trades: "70%"`) and passed to the `Head Trader`.
- **Structured outputs**: Native Gemini structured outputs using strict Pydantic schemas.
- **Qualitative Risk Strategy**: The Head Trader **never** outputs raw prices for stop-loss or take-profit. It outputs the chosen strategy (e.g., `stop_loss_type: "2.0_atr"`, `take_profit_type: "next_resistance"`), forcing the execution engine to compute the exact prices.

---

## Component 4 — Decision Schema

The contract between reasoning and execution.

```python
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from decimal import Decimal
from typing import Literal

class TradeProposal(BaseModel):
    decision_id: UUID
    timestamp: datetime
    symbol: str
    action: Literal["long", "short", "flat", "close"]
    stop_loss_strategy: Literal["1.5_atr", "2.0_atr", "swing_low", "tight_atr"]
    take_profit_strategy: Literal["3.0_atr", "4.0_atr", "next_resistance", "risk_reward_multiplier"]
    risk_reward_ratio: Decimal = Field(..., max_digits=4, decimal_places=2) # e.g., 2.00
    hold_period_bias: Literal["short", "medium", "long"]
    reasoning_summary: str
```

---

## Component 5 — Execution & Risk Layer

- **`RiskManager` (Deterministic Python Code)**: Receives the `TradeProposal` from the Head Trader and performs mathematical evaluations:
  1. **Circuit Breakers:** Rejects proposal if max drawdown (>15%), max concurrent positions (>3), or max asset exposure (>50%) is breached.
  2. **Price Calculations:** Computes the exact entry price, resolves the qualitative stop-loss strategy to a decimal (e.g., Entry - 2 * ATR), and computes take-profit distance.
  3. **Position Sizing:** Calculates safe position size in USD using a fixed fractional risk model (risking exactly 1% of account equity per trade).
- **`Broker` interface**: `place_order(decision)`, `get_positions()`, `get_balance()`, `cancel(order_id)`.
- **`PaperBroker`**: in-process, fills at next-candle open with realistic slippage + fees.
- **`CoinbaseBroker`**: Client-side emulation of bracket OCO orders. Automatically monitors the active stop/limit legs via websockets and cancels the opposite leg upon execution.

---

## Component 6 — Evaluation Harness

- **Backtest runner**: Replays historical candles through the *exact same* production code path.
  * *Sub-candle Fill Resolution:* If a 4h candle's range triggers both the stop-loss and take-profit levels, the backtester will fetch **1-minute or 5-minute data** from DuckDB for that period to resolve the exact sequence of events, eliminating bar-penetration execution bias.
- **Walk-forward Splitting**: Handled deterministically using `scikit-learn`'s `TimeSeriesSplit` to prevent chronological information leakage.
- **Metrics**: Powered by the **`QuantStats`** library. Reconstructs a daily equity curve from the trade ledger to generate metrics (Sharpe, Sortino, max drawdown, win rate) and comparison benchmarks (vs. buying and holding BTC).

---

## Component 7 — Observability

- **LLM tracing**: **Langfuse Cloud (Free Tier)**. Automatically traces prompts, structured outputs, latencies, and token costs without self-hosting overhead.
- **Decision audit log**: append-only Parquet, joinable to executions and outcomes.
- **Live dashboard**: Streamlit, embedding the `QuantStats` HTML tear sheet. Shows open positions, recent decisions with reasoning, current drawdown, and agent-disagreement rate.
- **Alerts**: Discord webhooks on execution, drawdown breach, or cost deviations.

---

## Component 8 — Graduation Criteria

Real money is enabled only when ALL pass over a 3-month live paper window:

1. **Sharpe > 1.0** (calculated via `QuantStats`) net of fees and slippage.
2. **Max drawdown < 15%**.
3. **≥ 30 closed trades** (verified via bootstrap/Monte Carlo simulations in python to prove statistical significance above random drift).
4. **Cost-per-decision below ceiling** — target is $0.00 due to Gemini Free Tier usage.
5. **Manual review** — read the last 20 trade transcripts.

Initial real-money allocation: **$500**.

---

## Component 9 — Tech Stack

| Layer | Tool | Rationale |
|---|---|---|
| Language | Python 3.12 | Domain default |
| Package manager | `uv` | Fast, modern, lockfile-based |
| Data & Storage | `pandas`, `ccxt`, `duckdb` | Standard quant stack |
| Indicators | `ta-lib` (C-wrapper) | Extremely fast indicator & candle pattern calculations |
| S/R Peaks | `scipy.signal` | Prominence-based peak detection |
| LLM SDK | `google-genai` | Native structured outputs, free tier access |
| Observability | Langfuse Cloud | Free tier tracing, zero server maintenance |
| Performance Metrics | `QuantStats` | Professional tear sheets, benchmark plots |
| Scheduler | `APScheduler` / `Rocketry` | Eliminates execution drift, concurrency-safe |
| Backtest Splits | `scikit-learn` (`TimeSeriesSplit`) | Prevents future information leakage |
| Tests | `pytest`, `hypothesis` | Property tests on backtester |
