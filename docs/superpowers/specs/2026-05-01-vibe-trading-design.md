# Vibe Trading — Design Spec

**Date**: 2026-05-01
**Author**: Raj (with brainstorming assistance)
**Status**: Draft for review

## Purpose

Build a paper-trading system for crypto swing trades that uses Claude as a multi-agent reasoner over Murphy-style technical-analysis features. The dual goal:

1. **Trading goal** — system that, when graduated to real money, demonstrably trades better than buy-and-hold over a 3-month paper window.
2. **Learning goal** — exercise production AI engineering muscles end-to-end: structured outputs, agent orchestration, prompt caching, evals, observability, cost tracking, guardrails, and reproducible backtests.

The learning goal is primary; the trading goal is the forcing function that makes the learning real.

## Scope (v1)

### In scope

- Paper trading on BTC/USD and ETH/USD using 4h candles for decisions and 1D candles for trend context.
- Deterministic feature pipeline computing trend, momentum, volume, market structure, chart patterns, and candlestick patterns.
- Four-agent LLM reasoning system (Technical Analyst, Volume Analyst, Risk Manager, Head Trader).
- Backtest harness that replays historical candles through the exact same pipeline used in live paper.
- Eval harness with golden datasets and trading metrics.
- Observability via Langfuse + append-only decision audit log.
- Streamlit dashboard, Discord alerts.
- A `CoinbaseBroker` implemented and tested against the sandbox, gated off in v1 by `LIVE_TRADING_ENABLED=false`.

### Out of scope (v1)

- Real-money execution (gated until graduation criteria pass — see Component 8).
- ML-based pattern detection (designed in as a future swap; rule-based only at v1).
- Sentiment analysis from news/social.
- Order types beyond market entries with bracket OCO stop + take-profit.
- Multi-exchange smart routing.
- Markets other than crypto.
- Sub-4h timeframes.

## Architecture

```
[Data Layer] ──► [Feature Pipeline] ──► [Multi-Agent Reasoning] ──► [Decision] ──► [Execution]
                                                                                       │
                                                                                       ▼
              [Observability + Eval Harness] ◄─────────────────────────────────── [Outcomes]
```

**Key invariant**: the same `MarketSnapshot → Decision` function works in backtest, live paper, and (eventually) real money. No mode-specific forks. This is what makes evals trustworthy and graduation meaningful.

## Component 1 — Data Layer

- **Source**: `ccxt` unified API. Coinbase Advanced primary; Binance secondary for deeper OHLCV history.
- **Universe**: BTC/USD, ETH/USD. Expandable via config.
- **Timeframes**: 4h (decision cadence), 1D (trend bias context).
- **Storage**: DuckDB single-file DB for OHLCV + computed features. Parquet exports for portable backtest replay.
- **Refresh**: scheduler runs the pipeline at every 4h candle close.
- **Bootstrap**: pull ≥ 2 years of 1D and ≥ 6 months of 4h on first run.

## Component 2 — Feature Pipeline

Pure-Python, deterministic. Murphy-derived features per `MarketSnapshot`:

- **Trend**: MA20/50/200 stack, ADX (trend strength), price-vs-MA position, 1D-timeframe trend bias.
- **Momentum**: RSI14, MACD (12/26/9), Stochastic (14/3/3), classical and hidden divergences flagged.
- **Volume**: OBV trend, volume MA20, volume spike flags (Murphy's confirmation principle).
- **Structure**: swing highs/lows, support/resistance via fractal pivots, recent breakouts (with retests flagged).
- **Patterns**: rule-based detectors for head-and-shoulders, double tops/bottoms, triangles, flags. (ML detectors slot in here later.)
- **Candlesticks**: pin bars, engulfing, morning/evening stars when occurring near a structural level.

**Output**: a Pydantic v2 `MarketSnapshot` per symbol per candle close. The schema is the contract — agents only ever see this, never raw OHLCV.

## Component 3 — Multi-Agent Reasoning

Four agents. Default model: Claude Sonnet 4.6. Head Trader may use Opus 4.7 if cost budget allows. Each agent has a tightly-scoped Pydantic input/output schema.

| Agent | Reads | Outputs |
|---|---|---|
| **Technical Analyst** | Trend + momentum + structure features | Bias (bullish/bearish/neutral), thesis, key levels, confluence count |
| **Volume Analyst** | Volume features + price action | Confirmation strength, divergences, "is the move real?" |
| **Risk Manager** | Recent ATR, open positions, account state, prior agents' bias | Max position size, stop placement, veto power on drawdown breach |
| **Head Trader** | All three above + recent agent track record | Final `Decision`: action (long/short/flat), size, entry, stop, take-profit, hold-period bias, reasoning summary |

- **Orchestration**: custom mini graph runner (~150 LOC) using the Anthropic SDK directly. Three parallel agent calls fan into the Head Trader. No LangGraph dependency in v1 — the topology is small and the learning value of building it is high.
- **Tool use**: Technical Analyst gets a `lookup_levels(symbol, timeframe)` tool to fetch precise S/R on demand instead of stuffing every level into the prompt.
- **Prompt caching**: system prompts and the static portion of the snapshot (indicator definitions, schema docs) cached using Anthropic's prompt-cache controls. Cache-hit rate is a tracked metric.
- **Structured outputs**: tool-use mode with strict Pydantic schemas. Schema-compliance rate is a tracked metric.

## Component 4 — Decision Schema

The contract between reasoning and execution.

```python
class Decision(BaseModel):
    decision_id: UUID
    timestamp: datetime
    symbol: str
    action: Literal["long", "short", "flat", "close"]
    size_quote_ccy: Decimal           # USD notional
    entry_type: Literal["market", "limit"]
    entry_price: Decimal | None
    stop_price: Decimal
    take_profit_price: Decimal
    hold_period_bias: Literal["short", "medium", "long"]  # days, days-to-week, weeks
    reasoning_summary: str
    agent_transcripts: dict[str, AgentOutput]  # full audit trail
```

## Component 5 — Execution Layer

- **`Broker` interface**: `place_order(decision)`, `get_positions()`, `get_balance()`, `cancel(order_id)`.
- **`PaperBroker`**: in-process, fills at next-candle open with realistic slippage + fees, ledger persisted to DuckDB.
- **`CoinbaseBroker`**: implemented from day one, exercised against Coinbase sandbox in CI, gated behind `LIVE_TRADING_ENABLED=false`.
- **Order types**: bracket OCO (stop + TP) where supported by exchange; simulated client-side in paper.

## Component 6 — Evaluation Harness

Most of the AI engineering learning lives here.

- **Backtest runner**: replays historical candles through the *exact same* pipeline. Only the Data Layer is mocked to a Parquet replay; everything downstream — feature pipeline, agents, broker — is the production code path. Outputs full trade ledger + per-decision agent transcripts.
- **Walk-forward**: train/holdout splits for any ML elements and prompt-change A/B testing with proper held-outs (no peeking).
- **Agent golden datasets**:
  - 50–100 hand-labeled `MarketSnapshot` → expected `(bias, key_levels)` for the Technical Analyst.
  - ~30 conflicting-signal scenarios with labeled "correct" `Decision` for the Head Trader.
- **Regression suite**: any prompt change must not regress golden-set scores beyond a configurable threshold. Runs in CI.
- **Metrics**:
  - **Trading**: Sharpe, Sortino, max drawdown, hit rate, expectancy, exposure %, profit factor.
  - **Agent**: schema-compliance rate, reasoning grounded-ness (LLM-judge: does the reasoning cite facts that are actually in the snapshot?), decision-stability under prompt perturbation, agent-disagreement rate.

## Component 7 — Observability

- **LLM tracing**: Langfuse, self-hosted via Docker Compose. Every prompt, completion, latency, token cost attached to a `decision_id`.
- **Decision audit log**: append-only Parquet, joinable to executions and outcomes. Enables queries like "win rate of trades where Volume Analyst flagged divergence."
- **Cost tracker**: $/decision, $/day, projected $/month. Alarm when daily spend deviates >2σ from rolling baseline.
- **Live dashboard**: Streamlit. Open positions, recent decisions with reasoning, current drawdown, agent-disagreement rate, eval-regression status.
- **Alerts**: Discord webhook on (a) any execution, (b) drawdown breach, (c) eval regression on a prompt change, (d) cost spike.

## Component 8 — Graduation Criteria

Real money is enabled only when ALL pass over a 3-month live paper window:

1. **Sharpe > 1.0** net of fees and slippage.
2. **Max drawdown < 15%**.
3. **≥ 30 closed trades** (statistical significance floor).
4. **Agent-eval golden scores stable** — no >10% regression over the period.
5. **Cost-per-decision below ceiling** — ballpark $0.10/decision; precise number locked at implementation time.
6. **Manual review** — read the last 20 trade transcripts. Do they make sense?

Initial real-money allocation: **$500**. Doubled only after another 30 live real-money trades pass the same gate.

## Component 9 — Tech Stack

| Layer | Tool | Rationale |
|---|---|---|
| Language | Python 3.12 | Domain default |
| Package manager | `uv` | Fast, modern, lockfile-based |
| Data | `pandas`, `pandas-ta`, `ccxt`, `duckdb` | Standard quant stack |
| LLM SDK | `anthropic` (direct, with prompt caching) | Primitives > framework for learning |
| Schemas | `pydantic` v2 | Structured outputs, contracts |
| Orchestration | Custom mini graph runner (~150 LOC) | Pedagogical |
| Backtest | Custom, built around `Broker` interface | Reproducibility, no mode forks |
| Observability | Langfuse (self-hosted) + Parquet logs | Traces + warehouse |
| Dashboard | Streamlit | Fastest to ship |
| Alerts | Discord webhook | Free, instant |
| Deployment | Docker Compose → Fly.io / Hetzner VPS for live | Cheap, simple |
| Tests | `pytest`, `hypothesis` (property tests on backtester) | ML-grade rigor |

## Repository layout (v1)

```
vibe-trading/
├── docs/superpowers/specs/            # this spec + future ones
├── src/vibe_trading/
│   ├── data/                          # ccxt fetcher, DuckDB store
│   ├── features/                      # indicators, patterns
│   ├── agents/                        # 4 agents + graph runner
│   ├── brokers/                       # Broker interface, Paper, Coinbase
│   ├── decisions/                     # Decision schema, audit log
│   ├── eval/                          # backtest, golden sets, metrics
│   ├── obs/                           # Langfuse, dashboard, alerts
│   └── runtime/                       # scheduler, entry points
├── tests/                             # pytest + hypothesis
├── notebooks/                         # eval analysis, ad-hoc research
├── docker-compose.yml                 # Langfuse + app
├── pyproject.toml
└── README.md
```

## Risks and open questions

- **LLM non-determinism**: even with `temperature=0`, reasoning output drifts. Mitigation: log everything, treat eval suite as the source of truth, never deploy a prompt change without eval pass.
- **Lookahead bias in backtest**: easy to leak future info accidentally (e.g., a divergence calculated on a window that includes the candle being decided on). Mitigation: backtest harness re-runs the *exact* pipeline; feature pipeline takes a `decision_time` cutoff parameter; property tests assert no future bars are read.
- **Cost runaway**: 4 agents × 6 decisions/day × 2 symbols ≈ 48 LLM calls/day. With caching this is cheap, but a misconfigured retry loop could blow it up. Mitigation: hard daily $-cap kill-switch, traced cost dashboard.
- **Survivorship bias**: BTC/ETH have survived; backtests on them flatter strategies. Acknowledged as a known limitation of v1; expanding the universe is a future-work item.
- **Slippage assumptions**: paper-broker slippage is a guess until calibrated against live paper fills. Mitigation: instrument live paper fills, recalibrate, re-backtest.
- **Graduation criteria are estimates** — we'll likely tighten them after seeing real eval distributions.

## Future extensions (post-v1)

- ML-based chart-pattern detectors (small CNNs/classifiers trained on labeled patterns) replacing or augmenting rule-based detectors in Component 2.
- News + social sentiment agent.
- Universe expansion to top-10 by market cap.
- Multi-timeframe agents (1h tactical layer under the 4h decision layer).
- Reinforcement learning over the agent's policy parameters.
- Real-money graduation (gated by Component 8 criteria).
