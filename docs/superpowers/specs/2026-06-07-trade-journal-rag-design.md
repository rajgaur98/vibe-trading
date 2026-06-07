# Design Spec — Trade-Journal RAG (precedent retrieval for the Head Trader)

> **Initiative:** turn the append-only decision journal into a retrieval-augmented memory.
> Before the Head Trader decides, retrieve the most *similar past setups and their outcomes*
> and inject them as precedent. This is the rubric's Tier-2 **RAG** done in a way that fits a
> numeric trading bot: embed a textual "setup card" per decision, semantic-rank past
> decisions, and ground the trader's reasoning in its own track record.

## Problem

The Head Trader decides each entry from the current analyst report + a static scorecard
(`{"accuracy": 0.55}`). It has **no memory of its own history** — it can't see that "the last
4 times I went long into resistance while overbought, 3 hit the stop." The append-only audit
log (`audit.py`, Parquet) and the `trades` table (now linked to decisions via `decision_id`)
hold exactly that history, but nothing feeds it back into the decision.

## Solution

A **trade-journal RAG layer**. At decision time we build a compact text **setup card**
(analyst thesis + regime labels), embed it (Gemini via `litellm.embedding`), and persist the
embedding keyed by `decision_id`. Before the trader decides, we embed the *current* setup,
**cosine-rank past decisions in-memory** (the journal is small — no vector DB), take the top-k,
attach each one's **outcome**, and inject them into the trader's prompt as precedent. The
`DecisionPipeline` orchestrates it (it already owns analyst→trader); the retriever is an
**injected dependency** so it is eval-safe and degrades to empty on cold-start.

Outcomes are dual-source:
- **Closed-trade decisions** → real outcome from `trades` (win/loss, realized PnL %, exit).
- **FLAT / risk-rejected decisions** → a **counterfactual forward return** over a fixed horizon
  (default 24h = next 6×4h candles, from the DuckDB candle cache): "a {action} would have ±X%."

**Out of scope:** news/narrative RAG (a separate, larger ingestion project); pgvector / any
external vector DB (in-memory cosine is sufficient at this scale); routing embedding calls
through the LLM cost sink (tiny; deferred); making the eval include precedents (the eval uses
an empty retriever — see *Eval-safety*).

## Components

### 1. `src/vibe_trading/journal.py` [NEW]

```python
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/text-embedding-004")
COUNTERFACTUAL_HORIZON_CANDLES = int(os.getenv("JOURNAL_COUNTERFACTUAL_HORIZON_CANDLES", "6"))  # ×4h = 24h
PRECEDENT_K = int(os.getenv("JOURNAL_PRECEDENT_K", "4"))
```

- **`build_setup_card(analyst_report, snapshot) -> str`** — deterministic text blob the embedding
  represents. Includes the analyst `thesis`, `market_bias`, `volume_confirmation`,
  `confluence_score`, and the snapshot regime labels (`rsi_regime`, `macd_regime`, `adx_regime`,
  `obv_trend`, `support_proximity`, `resistance_proximity`, `candlestick_pattern`,
  `funding_rate`, `open_interest_trend`). Pure; no I/O.

- **`embed(text) -> Optional[list[float]]`** — `litellm.embedding(model=EMBEDDING_MODEL, input=[text])`
  → the vector. Returns `None` on any error (never raises).

- **`Precedent`** (dataclass) — `symbol, action, when, similarity, kind ("closed"|"counterfactual"),
  outcome_pct, outcome_label` (e.g. `"hit TP +2.3%"` / `"skipped; a long would have +1.1% / 24h"`).

- **`persist_embedding(pg, decision_id, symbol, timestamp, action, entry_price, setup_text, embedding)`**
  — INSERT into `decision_embeddings` (no-op if embedding is None or already present). Caller
  supplies the pooled `pg` connection. Never raises.

- **`PrecedentRetriever`** — constructed with a `PostgresDatabase` factory + a `Database` (DuckDB)
  factory + `k`/`horizon`. Methods:
  - **`retrieve_for(setup_text) -> RetrievalResult(embedding, precedents)`** — embeds `setup_text`,
    then `retrieve(embedding)`; returns both so the caller can persist the embedding. On embed
    failure → `RetrievalResult(None, [])`.
  - **`retrieve(embedding) -> list[Precedent]`** — load candidate rows from `decision_embeddings`
    **whose `timestamp` is older than the counterfactual horizon** (so a counterfactual is
    actually known), cosine-rank (numpy) against `embedding`, take top-`k`, attach each outcome
    (below), return. Returns `[]` on any error or empty journal.
  - **`_attach_outcome(row) -> Precedent`**: if a `trades` row exists for `decision_id` →
    **closed** outcome (result, `realized_pnl`, exit price → PnL %). Else → **counterfactual**:
    read the symbol's 4h `close` at `timestamp` and at `timestamp + horizon` from DuckDB candles,
    forward return % signed by the decision's `action` (a long profits on an up-move).

- **`NoOpRetriever`** — `retrieve_for(_) -> RetrievalResult(None, [])`. Injected by the eval and
  whenever `JOURNAL_RAG_ENABLED` is false, so the trader behaves exactly as today.

### 2. `src/vibe_trading/data/db.py` [MODIFY] — new table + migration

Postgres `_create_tables` gains (DuckDB does not need it — embeddings are transactional state):
```sql
CREATE TABLE IF NOT EXISTS decision_embeddings (
    decision_id   VARCHAR PRIMARY KEY,
    symbol        VARCHAR,
    timestamp     TIMESTAMP,
    action        VARCHAR,
    entry_price   DOUBLE PRECISION,
    setup_text    TEXT,
    embedding     DOUBLE PRECISION[]
)
```
Plus the idempotent migration list already in `_create_tables` is irrelevant here (new table via
`CREATE TABLE IF NOT EXISTS`). `embedding` is a native Postgres `float8[]`; psycopg2 adapts a
Python `list[float]` to/from it. The INSERT uses the DuckDB-dialect `?` placeholders that
`translate_query` rewrites to `%s` (no special-casing needed for a plain INSERT).

### 3. `src/vibe_trading/agents/trader.py` [MODIFY]

`HeadTrader.decide(symbol, analyst_report, scorecard, open_positions, current_price, precedents=None)`.
When `precedents` is non-empty, format a **"PRECEDENTS — similar past setups you took and how
they resolved"** block and append it to the user prompt (one line per precedent: symbol, action,
date, similarity, outcome). When `None`/empty, the prompt is byte-identical to today's. The
trader's structured-output contract (`HeadTraderOutput`) is unchanged.

### 4. `src/vibe_trading/runtime/decision_pipeline.py` [MODIFY]

`DecisionPipeline.__init__` gains an injected `retriever` (defaults to `NoOpRetriever`). In
`run_symbol`, after the analyst + snapshot stages and before the trader:
```python
setup_text = build_setup_card(analyst_report, snapshot)
retrieval = self.retriever.retrieve_for(setup_text)        # embedding + precedents
proposal = self.trader.decide(symbol, analyst_report, self.scorecard,
                              open_positions, current_price=exec_price,
                              precedents=retrieval.precedents)
```
`DecisionResult` gains `setup_text` and `setup_embedding` so the scheduler can persist the
embedding alongside the decision_log/audit writes. Reads (embed + retrieve) live in the pipeline
— consistent with it already calling the analyst/trader; the only **write** stays in the scheduler.

### 5. `src/vibe_trading/runtime/scheduler.py` [MODIFY]

- Construct the retriever once in `__init__`: a real `PrecedentRetriever` when
  `JOURNAL_RAG_ENABLED` is true (default), else a `NoOpRetriever`. Gating is purely this flag —
  it works in PAPER and LIVE_TESTNET alike (both populate decision_log/trades). The eval never
  reaches this code (it constructs `HeadTrader` directly), so eval stays precedent-free. Pass the
  retriever into `DecisionPipeline`.
- After the decision_log INSERT, **persist the setup embedding**:
  `journal.persist_embedding(self.pg_db, decision_id, symbol, proposal["timestamp"], action,
  result.snapshot["close"], result.setup_text, result.setup_embedding)` — embed-on-write, keyed
  by `decision_id`, so the row is available as a future precedent once its outcome lands.

### 6. Config (`.env.example`)
```
JOURNAL_RAG_ENABLED=true                       # retrieve past-setup precedents for the trader
EMBEDDING_MODEL=gemini/text-embedding-004      # litellm-format embedding model
JOURNAL_PRECEDENT_K=4                           # how many precedents to inject
JOURNAL_COUNTERFACTUAL_HORIZON_CANDLES=6        # 4h candles ahead for FLAT/rejected counterfactuals (=24h)
```

## Data flow (one symbol, live)
```
analyst.analyze ─▶ build_setup_card(thesis + regime labels)
                ─▶ retriever.retrieve_for(card): embed(card) ─▶ cosine-rank past decisions
                       (older than horizon) ─▶ top-k ─▶ attach outcome
                         · closed      → trades(decision_id) → realized PnL %
                         · FLAT/reject → forward return over horizon (DuckDB candles)
                ─▶ trader.decide(analyst, precedents) ─▶ "PRECEDENTS" block in prompt ─▶ proposal
scheduler ─▶ decision_log INSERT ─▶ journal.persist_embedding(decision_id, …, embedding)
```

## Error handling
| Failure | Behavior |
|---|---|
| `embed()` errors | `retrieve_for` → `(None, [])`; no precedents, no embedding persisted; logged |
| retrieval / cosine / outcome errors | `retrieve` → `[]`; tick continues |
| empty / cold-start journal | `[]` precedents; trader prompt identical to today |
| `persist_embedding` error | logged, swallowed (a decision is still logged/audited normally) |
| counterfactual candle gap | that precedent is dropped (skipped), not zero-filled |

**Invariant:** the journal RAG layer is strictly *additive* — any failure degrades to "no
precedents," which is exactly today's behavior. It can never block or corrupt a tick.

## Eval-safety & re-baseline
The retriever is **injected**. The eval harness (`run_case`) constructs `HeadTrader` directly and
passes **no precedents** (and the live retriever is never wired in eval), so eval scoring stays
deterministic and the committed `0.79` baseline is **unaffected**. This is a deliberate
test-isolation: the eval measures base analyst+trader reasoning; precedents are a live-only
augmentation. Making the eval reflect precedents would require a seeded fixture journal + a
re-baseline — **explicitly deferred** (YAGNI). `JOURNAL_RAG_ENABLED=false` fully disables the
feature in live too.

## Testing (no live LLM / no network in unit tests)
`tests/test_journal.py`:
1. `build_setup_card` — deterministic; includes thesis + each regime label; stable for equal inputs.
2. cosine ranking — given a query vector + a fixed set of candidate vectors, returns the correct
   top-k in the right order (pure; embeddings injected, no `litellm`).
3. `_attach_outcome` closed — a `trades` fixture row → correct PnL % + win/loss label.
4. `_attach_outcome` counterfactual — a candle fixture → correct signed forward return for long vs short.
5. recency filter — a decision newer than the horizon is excluded from candidates.
6. graceful-empty — empty journal → `[]`; `embed()` returning `None` → `retrieve_for` → `(None, [])`.
7. `NoOpRetriever.retrieve_for` → `(None, [])`.

`tests/test_trader_precedents.py` (or extend trader tests):
8. `decide(..., precedents=[...])` injects the precedent lines into the prompt; `precedents=None`
   leaves the prompt unchanged (mock the LLM call; assert on the messages passed).

`tests/test_decision_pipeline.py` (extend):
9. the pipeline calls `retriever.retrieve_for` with the setup card and threads `precedents` into
   `trader.decide`; `DecisionResult` carries `setup_text`/`setup_embedding` (mock retriever).

`tests/test_scheduler.py` (extend):
10. `journal.persist_embedding` is invoked with the decision's id + embedding after the decision
    log write (mock `journal`/db).

`tests/test_db.py` (extend): `decision_embeddings` table is created with the expected columns.

**Manual/live verification:** with `JOURNAL_RAG_ENABLED=true` and some closed demo trades, confirm
a tick logs the embedding, retrieval returns precedents once the journal has aged past the
horizon, and the trader's Langfuse trace shows the precedent block in its prompt.

## Backwards compatibility
- Strictly additive: `precedents` defaults to `None`; `DecisionPipeline.retriever` defaults to
  `NoOpRetriever`; PAPER/eval behavior is unchanged.
- New Postgres table only; no change to existing tables. No new hard dependency (`litellm` already
  present; `numpy` already present via the indicator stack).
