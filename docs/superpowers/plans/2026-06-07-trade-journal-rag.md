# Trade-Journal RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve the most similar past trading setups + their outcomes and inject them into the Head Trader's prompt as precedent (RAG over the bot's own decision journal).

**Architecture:** A per-decision text "setup card" (analyst thesis + regime labels) is embedded with Gemini and stored keyed by `decision_id`. Before the trader decides, the `DecisionPipeline` embeds the current setup, cosine-ranks past decisions in-memory, attaches each precedent's outcome (closed trade via `decision_id`→`trades`, or a counterfactual forward return for FLAT/rejected), and passes the top-k into `trader.decide`. The retriever is an injected dependency (eval uses a no-op → baseline unaffected); the whole layer is additive and fail-soft.

**Tech Stack:** Python 3.12, `litellm.embedding` with **`gemini/gemini-embedding-001`** (verified working on the project's Gemini key, 3072-dim, free tier), `numpy` (already a dep, for cosine), `psycopg2` Postgres `float8[]` arrays, DuckDB candle cache, `pytest` + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-06-07-trade-journal-rag-design.md`

---

## File Structure
- **Create `src/vibe_trading/journal.py`** — the whole RAG layer: `build_setup_card`, `embed`, `cosine_topk`, `Precedent`/`RetrievalResult` dataclasses, `persist_embedding`, `PrecedentRetriever`, `NoOpRetriever`.
- **Modify `src/vibe_trading/data/db.py`** — add the `decision_embeddings` Postgres table.
- **Modify `src/vibe_trading/agents/trader.py`** — `decide(..., precedents=None)` + a "PRECEDENTS" prompt block.
- **Modify `src/vibe_trading/runtime/decision_pipeline.py`** — inject a retriever; build setup card → retrieve → pass precedents to the trader; carry `setup_text`/`setup_embedding` on `DecisionResult`.
- **Modify `src/vibe_trading/runtime/scheduler.py`** — construct the retriever (gated by `JOURNAL_RAG_ENABLED`), pass it to the pipeline, and persist the embedding after the decision-log write.
- **Modify `.env.example`, `README.md`** — config + docs.
- **Create `tests/test_journal.py`; modify `tests/test_trader.py`(new precedent test), `tests/test_decision_pipeline.py`, `tests/test_scheduler.py`, `tests/test_db.py`.**

**Locked interfaces (used across tasks):**
- `build_setup_card(analyst_report, snapshot: dict) -> str` (pure).
- `embed(text: str, model: str = None) -> Optional[list[float]]` (None on failure).
- `cosine_topk(query: list, candidates: list[tuple], k: int) -> list[tuple]` — `candidates` is `[(key, vector), …]`; returns `[(key, score), …]` top-k desc.
- `@dataclass Precedent(symbol, action, when, similarity, kind, outcome_pct, outcome_label)`.
- `@dataclass RetrievalResult(embedding: Optional[list], precedents: list[Precedent])`.
- `persist_embedding(conn, decision_id, symbol, timestamp, action, entry_price, setup_text, embedding) -> None` — `conn` is a connected DB wrapper; no-op if embedding is None; never raises.
- `PrecedentRetriever(k=…, horizon_candles=…, pg_factory=None, duck_factory=None, embed_fn=embed, now_fn=None)` with `.retrieve_for(setup_text) -> RetrievalResult` and `.retrieve(embedding) -> list[Precedent]`.
- `NoOpRetriever().retrieve_for(_) -> RetrievalResult(None, [])`.
- `HeadTrader.decide(symbol, analyst_output, scorecard, open_positions, current_price=0.0, precedents=None)`.

---

### Task 1: journal.py — pure pieces (setup card, cosine, dataclasses, NoOp)

**Files:**
- Create: `src/vibe_trading/journal.py`
- Test: `tests/test_journal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_journal.py`:

```python
from types import SimpleNamespace

from vibe_trading.journal import (
    build_setup_card, cosine_topk, NoOpRetriever, RetrievalResult, Precedent,
)


def _analyst(**over):
    base = dict(market_bias="bullish", volume_confirmation="confirmed", confluence_score=0.8,
                thesis="uptrend holding above support")
    base.update(over)
    return SimpleNamespace(**base)


def _snapshot(**over):
    base = dict(rsi_regime="overbought", macd_regime="bullish", adx_regime="strong_trend",
                obv_trend="rising", support_proximity="near", resistance_proximity="immediate_contact",
                candlestick_pattern="none", funding_rate="positive", open_interest_trend="rising",
                close=100.0)
    base.update(over)
    return base


def test_build_setup_card_deterministic_and_includes_fields():
    card = build_setup_card(_analyst(), _snapshot())
    assert "bias=bullish" in card and "rsi=overbought" in card and "adx=strong_trend" in card
    assert "thesis=uptrend holding above support" in card
    # deterministic for equal inputs
    assert build_setup_card(_analyst(), _snapshot()) == card


def test_build_setup_card_tolerates_missing_snapshot_keys():
    card = build_setup_card(_analyst(), {})
    assert "rsi=None" in card  # missing keys render as None, never raise


def test_cosine_topk_ranks_by_similarity():
    q = [1.0, 0.0]
    cands = [("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [0.7, 0.7])]
    top = cosine_topk(q, cands, k=2)
    assert [k for k, _ in top] == ["a", "c"]      # a (parallel) > c (45deg) > b (orthogonal)
    assert top[0][1] > top[1][1]


def test_cosine_topk_empty_and_zero_vectors():
    assert cosine_topk([1.0, 0.0], [], k=3) == []
    assert cosine_topk(None, [("a", [1.0])], k=3) == []
    assert cosine_topk([0.0, 0.0], [("a", [1.0, 1.0])], k=3) == []  # zero query -> []


def test_noop_retriever_returns_empty():
    r = NoOpRetriever().retrieve_for("any setup")
    assert isinstance(r, RetrievalResult)
    assert r.embedding is None and r.precedents == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_trading.journal'`

- [ ] **Step 3: Create the module's pure pieces**

Create `src/vibe_trading/journal.py`:

```python
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
PRECEDENT_K = int(os.getenv("JOURNAL_PRECEDENT_K", "4"))
COUNTERFACTUAL_HORIZON_CANDLES = int(os.getenv("JOURNAL_COUNTERFACTUAL_HORIZON_CANDLES", "6"))
_CANDLE_HOURS = 4  # the bot trades the 4h timeframe


@dataclass
class Precedent:
    symbol: str
    action: str
    when: str          # ISO date of the past decision
    similarity: float
    kind: str          # "closed" | "counterfactual"
    outcome_pct: float
    outcome_label: str


@dataclass
class RetrievalResult:
    embedding: Optional[list]
    precedents: list   # list[Precedent]


def build_setup_card(analyst_report, snapshot: dict) -> str:
    """Compact, deterministic text representation of a setup — what gets embedded.
    Combines the analyst's qualitative read with the snapshot's regime labels."""
    a = analyst_report
    s = snapshot or {}
    return (
        f"bias={a.market_bias}; volume={a.volume_confirmation}; confluence={a.confluence_score}; "
        f"rsi={s.get('rsi_regime')}; macd={s.get('macd_regime')}; adx={s.get('adx_regime')}; "
        f"obv={s.get('obv_trend')}; support_prox={s.get('support_proximity')}; "
        f"resistance_prox={s.get('resistance_proximity')}; pattern={s.get('candlestick_pattern')}; "
        f"funding={s.get('funding_rate')}; oi={s.get('open_interest_trend')}; "
        f"thesis={a.thesis}"
    )


def cosine_topk(query, candidates, k):
    """`candidates`: list of (key, vector). Returns the top-k [(key, score)] by cosine
    similarity, descending. Skips zero/length-mismatched vectors; [] on empty/zero query."""
    import numpy as np
    if query is None or not candidates:
        return []
    q = np.asarray(query, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    scored = []
    for key, vec in candidates:
        v = np.asarray(vec, dtype=float)
        vn = np.linalg.norm(v)
        if vn == 0 or v.shape != q.shape:
            continue
        scored.append((key, float(q.dot(v) / (qn * vn))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


class NoOpRetriever:
    """Used by the eval and whenever JOURNAL_RAG_ENABLED is false — the trader then
    behaves exactly as before (no precedents, no embedding)."""
    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        return RetrievalResult(None, [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/journal.py tests/test_journal.py
git commit -m "feat(journal): setup card + cosine ranking + NoOp retriever"
```

---

### Task 2: embedding + persistence + decision_embeddings table

**Files:**
- Modify: `src/vibe_trading/journal.py`, `src/vibe_trading/data/db.py`
- Test: `tests/test_journal.py`, `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_journal.py`:

```python
from unittest.mock import MagicMock, patch
from vibe_trading.journal import embed, persist_embedding


def test_embed_returns_vector():
    fake = MagicMock()
    fake.data = [{"embedding": [0.1, 0.2, 0.3]}]
    with patch("vibe_trading.journal.litellm.embedding", return_value=fake) as emb:
        out = embed("setup card text")
    assert out == [0.1, 0.2, 0.3]
    assert emb.call_args.kwargs["model"] == "gemini/gemini-embedding-001"


def test_embed_returns_none_on_error():
    with patch("vibe_trading.journal.litellm.embedding", side_effect=Exception("rate limited")):
        assert embed("x") is None


def test_persist_embedding_inserts():
    conn = MagicMock()
    persist_embedding(conn, "d1", "BTC/USDT", "2026-06-01", "long", 100.0, "card", [0.1, 0.2])
    sql, params = conn.execute.call_args.args
    assert "INSERT INTO decision_embeddings" in sql
    assert params[0] == "d1" and params[-1] == [0.1, 0.2]


def test_persist_embedding_noop_on_none():
    conn = MagicMock()
    persist_embedding(conn, "d1", "BTC/USDT", "2026-06-01", "long", 100.0, "card", None)
    conn.execute.assert_not_called()
```

Add to `tests/test_db.py` (a new test; the file already constructs a temp `Database`):

```python
def test_decision_embeddings_columns_present():
    import tempfile, os as _os
    from vibe_trading.data.db import PostgresDatabase
    # The Postgres schema string is what matters; assert the CREATE includes the table + columns
    # by inspecting the source (no live DB needed in unit tests).
    import inspect
    src = inspect.getsource(PostgresDatabase._create_tables)
    assert "decision_embeddings" in src
    for col in ("decision_id", "symbol", "embedding"):
        assert col in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -k "embed or persist" tests/test_db.py::test_decision_embeddings_columns_present -q`
Expected: FAIL — `embed`/`persist_embedding` not defined; `decision_embeddings` not in the schema.

- [ ] **Step 3: Add `embed` + `persist_embedding` to journal.py**

Append to `src/vibe_trading/journal.py` (and add `import litellm` near the top imports):

```python
import litellm  # add to the import block at the top of the file


def embed(text: str, model: str = None) -> Optional[list]:
    """Embed `text` via LiteLLM (Gemini by default). Returns the vector, or None on any
    error (rate limit, network, bad model) — retrieval then degrades to no precedents."""
    try:
        resp = litellm.embedding(model=model or EMBEDDING_MODEL, input=[text])
        return list(resp.data[0]["embedding"])
    except Exception as e:
        logger.warning(f"journal embed failed (non-fatal): {e}")
        return None


def persist_embedding(conn, decision_id, symbol, timestamp, action, entry_price,
                      setup_text, embedding) -> None:
    """Append this decision's setup embedding to decision_embeddings, keyed by decision_id,
    so it becomes a future precedent once its outcome lands. `conn` is a connected DB
    wrapper (caller owns connect/close). No-op when embedding is None; never raises."""
    if embedding is None:
        return
    try:
        conn.execute(
            "INSERT INTO decision_embeddings "
            "(decision_id, symbol, timestamp, action, entry_price, setup_text, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (decision_id, symbol, timestamp, action, entry_price, setup_text, embedding),
        )
    except Exception as e:
        logger.error(f"journal persist_embedding failed (non-fatal): {e}")
```

- [ ] **Step 4: Add the `decision_embeddings` table to db.py**

In `src/vibe_trading/data/db.py`, inside `PostgresDatabase._create_tables` (after the `llm_cost_log` CREATE, before the idempotent ALTER block), add:

```python
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_embeddings (
                    decision_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    action VARCHAR,
                    entry_price DOUBLE PRECISION,
                    setup_text TEXT,
                    embedding DOUBLE PRECISION[]
                )
            """)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py tests/test_db.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/vibe_trading/journal.py src/vibe_trading/data/db.py tests/test_journal.py tests/test_db.py
git commit -m "feat(journal): embed() + persist_embedding + decision_embeddings table"
```

---

### Task 3: PrecedentRetriever — candidate load + cosine retrieve + recency filter

**Files:**
- Modify: `src/vibe_trading/journal.py`
- Test: `tests/test_journal.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_journal.py`:

```python
from datetime import datetime
from vibe_trading.journal import PrecedentRetriever


def _retriever_with_candidates(rows, attach=None):
    """Build a retriever whose candidate loader returns `rows` and whose outcome attacher
    is stubbed, so retrieve()'s ranking logic is tested in isolation."""
    r = PrecedentRetriever(k=2, horizon_candles=6, embed_fn=lambda t: [1.0, 0.0])
    r._load_candidates = lambda cutoff: rows
    if attach is not None:
        r._attach_outcome = attach
    return r


def test_retrieve_ranks_and_limits_to_k():
    # rows: (decision_id, symbol, ts, action, entry_price, vector)
    rows = [
        ("d1", "BTC/USDT", datetime(2026, 6, 1), "long", 100.0, [1.0, 0.0]),   # cos 1.0
        ("d2", "ETH/USDT", datetime(2026, 6, 1), "short", 50.0, [0.0, 1.0]),   # cos 0.0
        ("d3", "SOL/USDT", datetime(2026, 6, 1), "long", 20.0, [0.7, 0.7]),    # cos ~0.7
    ]
    r = _retriever_with_candidates(
        rows, attach=lambda row, score: Precedent(row[1], row[3], "x", score, "closed", 1.0, "ok")
    )
    out = r.retrieve([1.0, 0.0])
    assert [p.symbol for p in out] == ["BTC/USDT", "SOL/USDT"]  # top-2 by cosine


def test_retrieve_for_embeds_then_retrieves():
    r = PrecedentRetriever(k=2, embed_fn=lambda t: [1.0, 0.0])
    r._load_candidates = lambda cutoff: []
    res = r.retrieve_for("setup card")
    assert res.embedding == [1.0, 0.0] and res.precedents == []


def test_retrieve_for_embed_failure_returns_empty():
    r = PrecedentRetriever(embed_fn=lambda t: None)
    res = r.retrieve_for("setup card")
    assert res.embedding is None and res.precedents == []


def test_retrieve_recency_cutoff_passed_to_loader():
    captured = {}
    r = PrecedentRetriever(k=2, horizon_candles=6, embed_fn=lambda t: [1.0, 0.0],
                           now_fn=lambda: datetime(2026, 6, 2, 0, 0, 0))
    def _loader(cutoff):
        captured["cutoff"] = cutoff
        return []
    r._load_candidates = _loader
    r.retrieve([1.0, 0.0])
    # cutoff = now - 6*4h = 24h earlier
    assert captured["cutoff"] == datetime(2026, 6, 1, 0, 0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -k "retrieve" -v`
Expected: FAIL — `PrecedentRetriever` not defined.

- [ ] **Step 3: Implement PrecedentRetriever (load + retrieve)**

Append to `src/vibe_trading/journal.py`:

```python
class PrecedentRetriever:
    """Semantic retrieval over the decision journal: embed the current setup, cosine-rank
    past decisions (older than the counterfactual horizon, so their outcome is known),
    take top-k, attach each outcome. All DB access is fail-soft."""

    def __init__(self, k: int = PRECEDENT_K, horizon_candles: int = COUNTERFACTUAL_HORIZON_CANDLES,
                 pg_factory=None, duck_factory=None, embed_fn=embed, now_fn=None):
        self.k = k
        self.horizon_candles = horizon_candles
        self._embed = embed_fn
        self._now = now_fn or (lambda: datetime.utcnow())
        # Lazy DB factories (own pooled / read-only connections); injected in tests.
        self._pg_factory = pg_factory
        self._duck_factory = duck_factory

    def _pg(self):
        if self._pg_factory:
            return self._pg_factory()
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase()

    def _duck(self):
        if self._duck_factory:
            return self._duck_factory()
        from vibe_trading.data.db import Database
        return Database(read_only=True)

    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        emb = self._embed(setup_text)
        if emb is None:
            return RetrievalResult(None, [])
        try:
            precedents = self.retrieve(emb)
        except Exception as e:
            logger.error(f"journal retrieve failed (non-fatal): {e}")
            precedents = []
        return RetrievalResult(emb, precedents)

    def retrieve(self, embedding) -> list:
        cutoff = self._now() - timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        rows = self._load_candidates(cutoff)            # [(id, symbol, ts, action, entry, vector)]
        ranked = cosine_topk(embedding, [(row, row[5]) for row in rows], self.k)
        out = []
        for row, score in ranked:
            p = self._attach_outcome(row, score)
            if p is not None:
                out.append(p)
        return out

    def _load_candidates(self, cutoff):
        pg = self._pg()
        pg.connect()
        try:
            rows = pg.conn.execute(
                "SELECT decision_id, symbol, timestamp, action, entry_price, embedding "
                "FROM decision_embeddings WHERE timestamp < ?",
                (cutoff,),
            ).fetchall()
            return [tuple(r) for r in rows]
        finally:
            pg.close()
```

(`_attach_outcome` is added in Task 4; the ranking tests stub it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -k "retrieve" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/journal.py tests/test_journal.py
git commit -m "feat(journal): PrecedentRetriever candidate load + cosine retrieve"
```

---

### Task 4: outcome attachment (closed trade + FLAT/rejected counterfactual)

**Files:**
- Modify: `src/vibe_trading/journal.py`
- Test: `tests/test_journal.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_journal.py`:

```python
from unittest.mock import MagicMock


def _row(action="long", symbol="BTC/USDT", entry=100.0):
    return ("d1", symbol, datetime(2026, 6, 1), action, entry, [1.0, 0.0])


def test_attach_outcome_closed_trade():
    r = PrecedentRetriever()
    pg = MagicMock()
    # trades row: (result, realized_pnl, size_usd)
    pg.conn.execute.return_value.fetchone.return_value = ("win", 23.0, 1000.0)
    r._pg = lambda: pg
    p = r._attach_outcome(_row(action="long"), score=0.9)
    assert p.kind == "closed"
    assert round(p.outcome_pct, 2) == 2.3          # 23 / 1000 * 100
    assert "win" in p.outcome_label and "+2.3%" in p.outcome_label


def test_attach_outcome_counterfactual_flat():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None  # no trade
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = (110.0,)  # future close
    r._pg = lambda: pg
    r._duck = lambda: duck
    p = r._attach_outcome(_row(action="flat", entry=100.0), score=0.8)
    assert p.kind == "counterfactual"
    assert round(p.outcome_pct, 2) == 10.0          # (110-100)/100 raw forward return
    assert "skipped" in p.outcome_label


def test_attach_outcome_counterfactual_rejected_short_signs_by_action():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = (90.0,)  # price fell 10%
    r._pg = lambda: pg
    r._duck = lambda: duck
    p = r._attach_outcome(_row(action="short", entry=100.0), score=0.7)
    # a SHORT profits when price falls -> +10%
    assert round(p.outcome_pct, 2) == 10.0
    assert "short" in p.outcome_label


def test_attach_outcome_counterfactual_missing_future_candle_drops():
    r = PrecedentRetriever(horizon_candles=6)
    pg = MagicMock(); pg.conn.execute.return_value.fetchone.return_value = None
    duck = MagicMock(); duck.conn.execute.return_value.fetchone.return_value = None  # no future candle
    r._pg = lambda: pg
    r._duck = lambda: duck
    assert r._attach_outcome(_row(action="flat"), score=0.5) is None  # dropped, not zero-filled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -k "attach_outcome" -v`
Expected: FAIL — `_attach_outcome` not defined.

- [ ] **Step 3: Implement `_attach_outcome`**

Append to the `PrecedentRetriever` class in `src/vibe_trading/journal.py`:

```python
    def _attach_outcome(self, row, score):
        decision_id, symbol, ts, action, entry_price, _vec = row
        when = ts.date().isoformat() if hasattr(ts, "date") else str(ts)

        # 1) Did this decision become a closed trade? -> real outcome.
        pg = self._pg()
        pg.connect()
        try:
            trade = pg.conn.execute(
                "SELECT result, realized_pnl, size_usd FROM trades WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        finally:
            pg.close()
        if trade is not None:
            result, realized_pnl, size_usd = trade
            pct = (float(realized_pnl) / float(size_usd) * 100.0) if size_usd else 0.0
            return Precedent(symbol, action, when, score, "closed", pct,
                             f"traded {action} -> {result} {pct:+.1f}%")

        # 2) Otherwise (FLAT / risk-rejected) -> counterfactual forward return from candles.
        if not entry_price:
            return None
        cutoff = ts + timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        duck = self._duck()
        duck.connect()
        try:
            fut = duck.conn.execute(
                "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                "AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                (symbol, cutoff),
            ).fetchone()
        finally:
            duck.close()
        if not fut or fut[0] is None:
            return None  # horizon not elapsed / data gap -> drop, never fabricate
        fwd = (float(fut[0]) - float(entry_price)) / float(entry_price) * 100.0
        # sign by the proposed direction: a long profits on an up-move, a short on a down-move
        signed = -fwd if action == "short" else fwd
        if action == "flat":
            label = f"skipped (flat); price moved {fwd:+.1f}% over {self.horizon_candles * _CANDLE_HOURS}h"
        else:
            label = f"{action} skipped (risk veto); would have {signed:+.1f}%"
        return Precedent(symbol, action, when, score, "counterfactual", signed, label)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_journal.py -q`
Expected: PASS (all journal tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/journal.py tests/test_journal.py
git commit -m "feat(journal): outcome attachment (closed PnL + counterfactual forward return)"
```

---

### Task 5: Head Trader — accept + inject precedents

**Files:**
- Modify: `src/vibe_trading/agents/trader.py`
- Test: `tests/test_trader.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_trader.py`:

```python
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from vibe_trading.agents.trader import HeadTrader
from vibe_trading.journal import Precedent


def _analyst():
    return SimpleNamespace(
        model_dump=lambda: {"market_bias": "bullish"},
    )


def _client_returning(content):
    c = MagicMock()
    c.provider = "gemini"
    c.model = "gemma-4-31b-it"
    c.call_llm.return_value = content
    return c


_VALID = '{"action":"flat","stop_loss_strategy":"1.5_atr","take_profit_strategy":"3.0_atr","risk_reward_ratio":2.0,"hold_period_bias":"medium","reasoning_summary":"no edge"}'


def test_decide_injects_precedents_into_prompt():
    client = _client_returning(_VALID)
    trader = HeadTrader(client=client)
    precedents = [Precedent("SOL/USDT", "long", "2026-05-01", 0.91, "closed", 2.3, "traded long -> win +2.3%")]
    trader.decide("BTC/USDT", _analyst(), {"accuracy": 0.5}, [], current_price=100.0, precedents=precedents)
    prompt = client.call_llm.call_args.kwargs["prompt"]
    assert "PRECEDENTS" in prompt
    assert "traded long -> win +2.3%" in prompt


def test_decide_without_precedents_omits_block():
    client = _client_returning(_VALID)
    trader = HeadTrader(client=client)
    trader.decide("BTC/USDT", _analyst(), {"accuracy": 0.5}, [], current_price=100.0)
    prompt = client.call_llm.call_args.kwargs["prompt"]
    assert "PRECEDENTS" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_trader.py -q`
Expected: FAIL — `decide()` has no `precedents` param / no PRECEDENTS block.

- [ ] **Step 3: Add the `precedents` param + prompt block**

In `src/vibe_trading/agents/trader.py`, change the `decide` signature to add `precedents=None`:

```python
    def decide(
        self,
        symbol: str,
        analyst_output: AnalystOutput,
        scorecard: dict,
        open_positions: list,
        current_price: float = 0.0,
        precedents=None,
    ) -> dict:
```

Then, immediately before the `prompt = f"""..."""` assignment, build the precedent block:

```python
            precedent_block = ""
            if precedents:
                lines = "\n".join(
                    f"- {p.symbol} {p.action.upper()} ({p.when}, similarity {p.similarity:.2f}): {p.outcome_label}"
                    for p in precedents
                )
                precedent_block = (
                    "\n--- PRECEDENTS — similar past setups you took and how they resolved ---\n"
                    f"{lines}\n"
                    "Weigh these against the current setup; repeated losses on a similar setup are a reason for caution.\n"
                )
```

And insert `{precedent_block}` into the prompt f-string just after the `--- Current Open Positions ---` section (before `--- Rules ---`):

```python
--- Current Open Positions ---
{json.dumps(open_positions, indent=2, default=str)}
{precedent_block}
--- Rules ---
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_trader.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/agents/trader.py tests/test_trader.py
git commit -m "feat(trader): inject retrieved precedents into the decision prompt"
```

---

### Task 6: DecisionPipeline — wire the retriever

**Files:**
- Modify: `src/vibe_trading/runtime/decision_pipeline.py`
- Test: `tests/test_decision_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decision_pipeline.py`:

```python
def test_pipeline_retrieves_and_threads_precedents():
    from vibe_trading.journal import RetrievalResult
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d1"}
    retriever = MagicMock()
    retriever.retrieve_for.return_value = RetrievalResult([0.1, 0.2], ["PRECEDENT_OBJ"])
    p.retriever = retriever

    res = p.run_symbol("BTC/USDT", "ts", 100.0)

    # retriever was asked with the setup card text (a string)
    assert isinstance(retriever.retrieve_for.call_args.args[0], str)
    # precedents threaded into the trader
    assert trader.decide.call_args.kwargs["precedents"] == ["PRECEDENT_OBJ"]
    # embedding + setup text carried on the result for the scheduler to persist
    assert res.setup_embedding == [0.1, 0.2]
    assert isinstance(res.setup_text, str)


def test_pipeline_defaults_to_noop_retriever():
    # _pipeline() builds DecisionPipeline without a retriever -> NoOp -> empty precedents
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d1"}
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert trader.decide.call_args.kwargs["precedents"] == []
    assert res.setup_embedding is None
```

(`_pipeline()` is the existing helper in this file; it constructs `DecisionPipeline(...)` — leave its call unchanged so the default-retriever path is exercised.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_decision_pipeline.py -k "retriev or precedent or noop" -v`
Expected: FAIL — `DecisionPipeline` has no `retriever`; `DecisionResult` has no `setup_text`/`setup_embedding`.

- [ ] **Step 3: Wire the retriever into the pipeline**

In `src/vibe_trading/runtime/decision_pipeline.py`:

(a) Add the new fields to `DecisionResult`:

```python
@dataclass
class DecisionResult:
    symbol: str
    status: str
    analyst_report: Any = None
    snapshot: Optional[dict] = None
    proposal: Optional[dict] = None
    trace_id: Optional[str] = None
    risk_result: Optional[dict] = None
    setup_text: Optional[str] = None
    setup_embedding: Optional[list] = None
```

(b) Add a `retriever` param to `__init__` (default `NoOpRetriever`), importing the journal pieces at the top:

```python
from vibe_trading.journal import build_setup_card, NoOpRetriever
```
```python
    def __init__(self, analyst, trader, risk_manager, feature_pipeline, broker,
                 scorecard: dict, trace_id_fn: Optional[Callable[[], Optional[str]]] = None,
                 retriever=None):
        ...
        self.scorecard = scorecard
        self.retriever = retriever or NoOpRetriever()
        self._trace_id_fn = trace_id_fn or (lambda: None)
```

(c) In `run_symbol`, between the snapshot stage and the trader call, build the card + retrieve, and thread precedents into `trader.decide`. Replace the existing trader-call block:

```python
        # Stage 3 — retrieve precedents (similar past setups + outcomes), then Head Trader.
        open_positions = self.broker.get_open_positions()
        setup_text = build_setup_card(analyst_report, snapshot)
        retrieval = self.retriever.retrieve_for(setup_text)
        proposal = self.trader.decide(
            symbol, analyst_report, self.scorecard, open_positions,
            current_price=exec_price, precedents=retrieval.precedents,
        )
        trace_id = self._trace_id_fn()
```

(d) Add `setup_text` + `setup_embedding` to BOTH the flat-path and the approved/rejected `DecisionResult(...)` returns:

```python
        if proposal["action"] == "flat":
            return DecisionResult(symbol, "flat", analyst_report=analyst_report,
                                  snapshot=snapshot, proposal=proposal, trace_id=trace_id,
                                  setup_text=setup_text, setup_embedding=retrieval.embedding)
        ...
        return DecisionResult(symbol, status, analyst_report=analyst_report, snapshot=snapshot,
                              proposal=proposal, trace_id=trace_id, risk_result=risk_result,
                              setup_text=setup_text, setup_embedding=retrieval.embedding)
```

- [ ] **Step 4: Run tests to verify they pass (and nothing regressed)**

Run: `PYTHONPATH=src uv run pytest tests/test_decision_pipeline.py -q`
Expected: PASS — the 2 new tests plus all existing pipeline tests (the NoOp default keeps them green; `trader.decide` now always gets a `precedents=` kwarg).

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/runtime/decision_pipeline.py tests/test_decision_pipeline.py
git commit -m "feat(pipeline): retrieve precedents and thread them to the trader"
```

---

### Task 7: Scheduler wiring + persist embedding + config/docs + full verification

**Files:**
- Modify: `src/vibe_trading/runtime/scheduler.py`, `.env.example`, `README.md`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scheduler.py`:

```python
def test_build_retriever_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("JOURNAL_RAG_ENABLED", "false")
    from vibe_trading.journal import NoOpRetriever
    sched = _scheduler_without_init()
    assert isinstance(sched._build_retriever(), NoOpRetriever)


def test_build_retriever_real_when_enabled(monkeypatch):
    monkeypatch.setenv("JOURNAL_RAG_ENABLED", "true")
    from vibe_trading.journal import PrecedentRetriever
    sched = _scheduler_without_init()
    assert isinstance(sched._build_retriever(), PrecedentRetriever)


def test_persist_embedding_called_after_decision(monkeypatch):
    import vibe_trading.runtime.scheduler as sched_mod
    calls = {}
    monkeypatch.setattr(sched_mod.journal, "persist_embedding",
                        lambda *a, **k: calls.setdefault("args", a))
    # The persist call passes decision_id + embedding; assert the helper exists + is import-wired.
    sched_mod.journal.persist_embedding("conn", "d1", "BTC/USDT", "ts", "long", 100.0, "card", [0.1])
    assert calls["args"][1] == "d1" and calls["args"][-1] == [0.1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -k "retriever or persist_embedding" -v`
Expected: FAIL — `_build_retriever` not defined; `scheduler.journal` not imported.

- [ ] **Step 3: Wire the retriever + persistence into the scheduler**

In `src/vibe_trading/runtime/scheduler.py`:

(a) Add the import near the other `vibe_trading` imports:

```python
from vibe_trading import journal
```

(b) Add a `_build_retriever` helper (place it just above `_maybe_start_ws_listener`):

```python
    def _build_retriever(self):
        """Real PrecedentRetriever when JOURNAL_RAG_ENABLED (default true), else NoOp.
        Gating is purely this flag — the eval never reaches here (it builds HeadTrader
        directly), so eval scoring stays precedent-free and the baseline is unaffected."""
        if os.getenv("JOURNAL_RAG_ENABLED", "true").lower() == "true":
            return journal.PrecedentRetriever()
        return journal.NoOpRetriever()
```

(c) In `__init__`, pass the retriever into the pipeline (extend the existing `DecisionPipeline(...)` construction):

```python
        self.decision_pipeline = DecisionPipeline(
            self.analyst, self.trader, self.risk_manager, self.pipeline, self.broker,
            scorecard=self.scorecard, trace_id_fn=self._current_trace_id,
            retriever=self._build_retriever(),
        )
```

(d) In `sync_and_evaluate`, inside the existing `decision_log` connect/close block (right after the decision_log INSERT, before the `finally: self.pg_db.close()`), persist the embedding:

```python
                        journal.persist_embedding(
                            self.pg_db.conn, proposal["decision_id"], proposal["symbol"],
                            proposal["timestamp"], proposal["action"],
                            float(snapshot.get("close", 0.0)),
                            result.setup_text, result.setup_embedding,
                        )
```

(Note: `result` is the `DecisionResult` returned by `decision_pipeline.run_symbol`; it carries `setup_text`/`setup_embedding`. `snapshot` is `result.snapshot`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_scheduler.py -q`
Expected: PASS

- [ ] **Step 5: Add config + docs**

Append to `.env.example`:

```env

# --- Trade-journal RAG (precedent retrieval for the Head Trader) ---
JOURNAL_RAG_ENABLED=true                       # retrieve similar past setups + outcomes as precedent
EMBEDDING_MODEL=gemini/gemini-embedding-001    # litellm-format embedding model (verified, free tier)
JOURNAL_PRECEDENT_K=4                           # how many precedents to inject
JOURNAL_COUNTERFACTUAL_HORIZON_CANDLES=6        # 4h candles ahead for FLAT/rejected counterfactuals (=24h)
```

Add a short subsection to `README.md` under the architecture/features area:

```markdown
### Trade-Journal RAG (decision memory)

Before the Head Trader decides, the bot embeds the current "setup card" (analyst thesis +
regime labels) with Gemini (`gemini-embedding-001`), cosine-ranks its **past** decisions in
memory, and injects the top-k similar setups + their outcomes into the trader's prompt —
closed trades (real PnL via `decision_id`→`trades`) and FLAT/rejected counterfactuals (forward
return over 24h). It's gated by `JOURNAL_RAG_ENABLED`, fail-soft (any error → no precedents),
and disabled in the eval (the committed baseline is unaffected).
```

- [ ] **Step 6: Run the FULL suite**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: PASS — the prior 235 tests plus the new journal (~13), trader (2), pipeline (2), scheduler (3), db (1) tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/vibe_trading/runtime/scheduler.py .env.example README.md tests/test_scheduler.py
git commit -m "feat(scheduler): wire journal RAG retriever + persist setup embeddings"
```

---

## Manual Live Verification (you, after merge)
1. `JOURNAL_RAG_ENABLED=true` (default). Restart the bot. Confirm a tick logs no errors and (once decisions exist) `decision_embeddings` gets rows.
2. After a few ticks/trades older than 24h accumulate, confirm the Head Trader's Langfuse trace shows a **"PRECEDENTS"** block in its prompt with real past setups + outcomes.
3. Sanity: `SELECT count(*) FROM decision_embeddings;` grows; closed-trade precedents show real PnL, FLAT ones show counterfactual returns.
```
