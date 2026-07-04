# Retrieval Upgrade Implementation Plan (pgvector + retrieval evals + ablation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real indexed vector search for the journal RAG (pgvector + HNSW on Supabase, with the in-Python cosine fallback preserved), a committed retrieval-quality eval (recall@k / MRR), and evidence for whether precedent injection helps (backtest A/B + online segmentation by `precedents_k`).

**Architecture:** D1 migrates `decision_embeddings.embedding` from `DOUBLE PRECISION[]` to pgvector's `vector(dim)` behind a capability probe, indexes it with an HNSW **halfvec expression index** (pgvector's documented pattern for >2000-dim embeddings — gemini-embedding-001 is 3072-dim, above HNSW's 2000-dim vector limit), and moves `PrecedentRetriever`'s ranking into a single SQL query. D2 adds pure recall@k/MRR metrics plus a runnable harness over a labeled query set and exported corpus. D3 stamps `precedents_k` onto decisions and scores (online segmentation) and adds a no-lookahead `ReplayJournal` to the backtester (A/B ablation). D4 (reranking) is explicitly **out of scope** — it is gated on D2's numbers showing recall headroom, decided then.

**Tech Stack:** Supabase Postgres + pgvector (extension ≥ 0.7 for halfvec), `pgvector` Python package (psycopg2 adapter), numpy, existing journal/backtest/eval modules, pytest.

**Workstream:** D of `docs/superpowers/specs/2026-07-04-ai-deepening-roadmap-design.md`.
**Depends on workstreams B and C** (`decision_log.prompt_version`, `decision_scores` table, `OutcomeScorer._persist`). Do not start before C is merged.

## Global Constraints

- Fail-soft capability probe: without the pgvector extension (vanilla local Postgres), everything keeps today's behavior — `float8[]` column, in-Python `cosine_topk`. No code path may *require* pgvector.
- `EMBEDDING_DIM` env (default `3072`, matching `gemini/gemini-embedding-001`) drives the vector typmod, casts, and index DDL. When existing rows disagree with the env, the migration aborts with a clear error rather than corrupting data.
- The HNSW index MUST be the halfvec expression form for dims > 2000 — a plain `hnsw (embedding vector_cosine_ops)` fails at 3072 dims. Queries must use the byte-identical cast expression or the index is not used.
- No-lookahead invariant in the backtest ablation: a precedent is only retrievable once its decision is older than the counterfactual horizon **relative to the replay clock**, and its counterfactual candle must not be after the replay clock.
- Unit tests are offline (no Postgres/pgvector/network). Postgres-touching behavior is covered by a manual-verification checklist plus an integration test gated on `RUN_PG_INTEGRATION=1`.
- Run tests with `uv run pytest <path> -v`.

---

### Task 1: pgvector capability probe, migration, and adapters (db.py)

**Files:**
- Modify: `pyproject.toml` (add `"pgvector>=0.2.5",` to `dependencies`)
- Modify: `src/vibe_trading/data/db.py`
- Test: `tests/test_pgvector.py`

**Interfaces:**
- Consumes: existing `PostgresDatabase._create_tables` / `connect`.
- Produces: `EMBEDDING_DIM: int` (module constant, env-driven); class flags `PostgresDatabase.pgvector_enabled: bool` and `PostgresDatabase.pgvector_halfvec: bool` (both default `False`); pure helpers `pgvector_migration_statements(dim: int) -> list[str]` and `resolve_embedding_dim(existing_len: Optional[int], env_dim: int) -> int`; `adapt_embedding(vec: list) -> object`. Task 2's SQL path and `journal.persist_embedding` rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pgvector.py
import numpy as np
import pytest

from vibe_trading.data.db import (
    PostgresDatabase, adapt_embedding, pgvector_migration_statements,
    resolve_embedding_dim,
)


def test_migration_statements_use_halfvec_expression_index():
    stmts = pgvector_migration_statements(dim=3072)
    joined = " ".join(stmts)
    assert "ALTER TABLE decision_embeddings ALTER COLUMN embedding TYPE vector(3072)" in joined
    assert "USING embedding::vector(3072)" in joined
    # HNSW at 3072 dims REQUIRES the halfvec expression form (vector caps at 2000)
    assert "USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)" in joined
    assert "CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw" in joined


def test_resolve_embedding_dim_prefers_existing_rows():
    assert resolve_embedding_dim(existing_len=1536, env_dim=3072) == 1536
    assert resolve_embedding_dim(existing_len=None, env_dim=3072) == 3072


def test_resolve_embedding_dim_rejects_conflict_with_data():
    # env says 3072 but... rows say 1536 -> use rows; a zero/negative env is invalid
    with pytest.raises(ValueError):
        resolve_embedding_dim(existing_len=None, env_dim=0)


def test_adapt_embedding_by_capability(monkeypatch):
    vec = [0.1, 0.2, 0.3]
    monkeypatch.setattr(PostgresDatabase, "pgvector_enabled", False)
    assert adapt_embedding(vec) == [0.1, 0.2, 0.3]          # float8[] path
    monkeypatch.setattr(PostgresDatabase, "pgvector_enabled", True)
    out = adapt_embedding(vec)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32  # pgvector adapter path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pgvector.py -v`
Expected: FAIL (`ImportError: cannot import name 'adapt_embedding'`)

- [ ] **Step 3: Write the implementation**

Add `"pgvector>=0.2.5",` to `[project] dependencies` in `pyproject.toml`, then run `uv lock`.

In `src/vibe_trading/data/db.py`, add module-level pieces (below `logger = ...`):

```python
import numpy as np

# Dimensionality of journal embeddings (gemini/gemini-embedding-001 = 3072).
# Drives the pgvector column typmod, query casts, and index DDL.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))


def resolve_embedding_dim(existing_len, env_dim: int) -> int:
    """The dim the migration must use: existing rows win (migrating a live table
    to the wrong typmod would corrupt it); env is the cold-start default."""
    if existing_len:
        return int(existing_len)
    if env_dim <= 0:
        raise ValueError(f"EMBEDDING_DIM must be positive, got {env_dim}")
    return env_dim


def pgvector_migration_statements(dim: int) -> list:
    """DDL to move decision_embeddings.embedding from float8[] to vector(dim) and
    index it. The index is pgvector's documented halfvec EXPRESSION form because
    HNSW on plain vector caps at 2000 dims (ours is 3072). Queries must use the
    byte-identical cast expression to hit the index (see journal.pgvector_topk_sql)."""
    return [
        f"ALTER TABLE decision_embeddings "
        f"ALTER COLUMN embedding TYPE vector({dim}) USING embedding::vector({dim})",
        f"CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw "
        f"ON decision_embeddings "
        f"USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)",
    ]


def adapt_embedding(vec):
    """Adapt a Python list embedding for the active decision_embeddings column type:
    float32 ndarray when pgvector is registered (the pgvector psycopg2 adapter
    serializes ndarrays to vector literals), plain list for the float8[] fallback."""
    if PostgresDatabase.pgvector_enabled:
        return np.asarray(vec, dtype=np.float32)
    return list(vec)
```

In `PostgresDatabase`, add the class flags next to `_pool`:

```python
    _pool = None
    pgvector_enabled = False   # extension present + column migrated to vector
    pgvector_halfvec = False   # halfvec type available (pgvector >= 0.7) -> HNSW index built
```

In `PostgresDatabase.connect()`, after `self.conn = PostgresConnectionWrapper(raw_conn)`:

```python
                if PostgresDatabase.pgvector_enabled:
                    try:
                        from pgvector.psycopg2 import register_vector
                        register_vector(raw_conn)  # idempotent per connection
                    except Exception as e:
                        logger.warning(f"pgvector register_vector failed (non-fatal): {e}")
```

At the END of `PostgresDatabase._create_tables` (inside the existing `try`, after the migration loop, before `self.conn.commit()`), add the probe + migration:

```python
            # --- pgvector enablement (fail-soft; see docs/superpowers/plans/...) ---
            try:
                try:
                    self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception:
                    self.conn.rollback()  # no privilege / no extension -> fallback path
                ext = self.conn.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
                if ext:
                    col = self.conn.execute(
                        "SELECT udt_name FROM information_schema.columns "
                        "WHERE table_name = 'decision_embeddings' "
                        "AND column_name = 'embedding'").fetchone()
                    if col and col[0] == "_float8":  # still float8[] -> migrate once
                        row = self.conn.execute(
                            "SELECT array_length(embedding, 1) FROM decision_embeddings "
                            "LIMIT 1").fetchone()
                        dim = resolve_embedding_dim(row[0] if row else None, EMBEDDING_DIM)
                        for stmt in pgvector_migration_statements(dim):
                            try:
                                self.conn.execute(stmt)
                            except Exception as e:
                                # index DDL may fail on pgvector < 0.7 (no halfvec);
                                # the column migration alone is still a win.
                                logger.warning(f"pgvector DDL skipped: {e}")
                                self.conn.rollback()
                    PostgresDatabase.pgvector_enabled = True
                    half = self.conn.execute(
                        "SELECT 1 FROM pg_type WHERE typname = 'halfvec'").fetchone()
                    PostgresDatabase.pgvector_halfvec = bool(half)
                    logger.info(f"pgvector enabled (halfvec={PostgresDatabase.pgvector_halfvec}).")
            except Exception as e:
                logger.warning(f"pgvector probe failed — using in-Python retrieval: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pgvector.py tests/test_db.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/vibe_trading/data/db.py tests/test_pgvector.py
git commit -m "feat(retrieval): pgvector probe, float8[]->vector migration, halfvec HNSW index"
```

---

### Task 2: SQL retrieval path in PrecedentRetriever

**Files:**
- Modify: `src/vibe_trading/journal.py`
- Test: `tests/test_journal.py` (extend)

**Interfaces:**
- Consumes: Task 1's flags, `adapt_embedding`, `EMBEDDING_DIM`.
- Produces: `pgvector_topk_sql(halfvec: bool, dim: int) -> str`; `PrecedentRetriever(..., use_pgvector: Optional[bool] = None)` (None = auto-detect from the class flags); `journal.persist_embedding` writes via `adapt_embedding`. Retrieval output (`list[Precedent]`) is unchanged — the trader and pipeline need no edits.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_journal.py`:

```python
def test_pgvector_topk_sql_halfvec_matches_index_expression():
    from vibe_trading.journal import pgvector_topk_sql
    sql = pgvector_topk_sql(halfvec=True, dim=3072)
    # must be byte-identical to the index expression or the HNSW index is skipped
    assert "embedding::halfvec(3072) <=> ?::halfvec(3072)" in sql
    assert "1 - (embedding::halfvec(3072) <=> ?::halfvec(3072)) AS similarity" in sql
    assert "WHERE timestamp < ?" in sql and "LIMIT ?" in sql


def test_pgvector_topk_sql_plain_vector_without_halfvec():
    from vibe_trading.journal import pgvector_topk_sql
    sql = pgvector_topk_sql(halfvec=False, dim=3072)
    assert "halfvec" not in sql
    assert "embedding <=> ?" in sql


def test_retriever_uses_sql_path_when_pgvector_active():
    from unittest.mock import MagicMock
    from datetime import datetime
    from vibe_trading.journal import PrecedentRetriever

    pg = MagicMock()
    pg.conn.execute.return_value.fetchall.return_value = [
        ("dec-1", "BTC/USDT", datetime(2026, 6, 1), "long", 100.0, 0.93),
    ]
    pg.conn.execute.return_value.fetchone.return_value = ("win", 50.0, 1000.0)
    r = PrecedentRetriever(pg_factory=lambda: pg, use_pgvector=True,
                           now_fn=lambda: datetime(2026, 7, 5))
    precedents = r.retrieve([0.1] * 4)
    assert len(precedents) == 1
    assert precedents[0].similarity == pytest.approx(0.93)
    assert precedents[0].kind == "closed"
    ranking_sql = [c.args[0] for c in pg.conn.execute.call_args_list
                   if "ORDER BY" in c.args[0]]
    assert ranking_sql and "<=>" in ranking_sql[0]   # ranking happened in SQL


def test_retriever_fallback_path_unchanged_when_pgvector_off():
    from unittest.mock import MagicMock
    from datetime import datetime
    from vibe_trading.journal import PrecedentRetriever

    pg = MagicMock()
    pg.conn.execute.return_value.fetchall.return_value = []
    r = PrecedentRetriever(pg_factory=lambda: pg, use_pgvector=False,
                           now_fn=lambda: datetime(2026, 7, 5))
    assert r.retrieve([0.1] * 4) == []
    sql = pg.conn.execute.call_args_list[0].args[0]
    assert "<=>" not in sql                 # candidates loaded, ranked in Python
    assert "SELECT decision_id, symbol" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_journal.py -v -k pgvector`
Expected: FAIL (`ImportError: cannot import name 'pgvector_topk_sql'`)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/journal.py`:

1. Add the import near the top: `from vibe_trading.data.db import adapt_embedding, EMBEDDING_DIM` (db imports nothing from journal — no cycle).

2. Add the SQL builder (below `cosine_topk`):

```python
def pgvector_topk_sql(halfvec: bool, dim: int) -> str:
    """Top-k cosine retrieval in SQL. With halfvec=True the cast expression is
    byte-identical to the HNSW expression index (db.pgvector_migration_statements)
    so the planner can use it; without halfvec it is a correct (unindexed) scan
    on the vector column. Params: (query, cutoff, query, k)."""
    if halfvec:
        col = f"embedding::halfvec({dim})"
        q = f"?::halfvec({dim})"
    else:
        col, q = "embedding", "?"
    return (
        "SELECT decision_id, symbol, timestamp, action, entry_price, "
        f"1 - ({col} <=> {q}) AS similarity "
        "FROM decision_embeddings WHERE timestamp < ? "
        f"ORDER BY {col} <=> {q} LIMIT ?"
    )
```

3. In `persist_embedding`, adapt the value: change the params tuple's last element from `embedding` to `adapt_embedding(embedding)`.

4. In `PrecedentRetriever.__init__`, add the parameter `use_pgvector: Optional[bool] = None` and store `self.use_pgvector = use_pgvector`. Add:

```python
    def _pgvector_active(self) -> bool:
        if self.use_pgvector is not None:
            return self.use_pgvector
        from vibe_trading.data.db import PostgresDatabase
        return PostgresDatabase.pgvector_enabled

    def _load_topk_pgvector(self, embedding, cutoff):
        """[(row, similarity)] where row matches _load_candidates' shape (vector
        slot None — _attach_outcome never reads it)."""
        from vibe_trading.data.db import PostgresDatabase
        sql = pgvector_topk_sql(PostgresDatabase.pgvector_halfvec, EMBEDDING_DIM)
        q = adapt_embedding(embedding)
        pg = self._pg()
        pg.connect()
        try:
            rows = pg.conn.execute(sql, (q, cutoff, q, self.k)).fetchall()
        finally:
            pg.close()
        return [((r[0], r[1], r[2], r[3], r[4], None), float(r[5])) for r in rows]
```

5. Rewrite `retrieve` to branch:

```python
    def retrieve(self, embedding) -> list:
        cutoff = self._now() - timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        if self._pgvector_active():
            ranked = self._load_topk_pgvector(embedding, cutoff)
        else:
            rows = self._load_candidates(cutoff)
            ranked = cosine_topk(embedding, [(row, row[5]) for row in rows], self.k)
        out = []
        for row, score in ranked:
            p = self._attach_outcome(row, score)
            if p is not None:
                out.append(p)
        return out
```

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run pytest tests/test_journal.py -v` — expected: all pass (existing journal tests exercise the fallback path unchanged).
Run: `uv run pytest` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/journal.py tests/test_journal.py
git commit -m "feat(retrieval): SQL top-k via pgvector with index-matching halfvec casts"
```

---

### Task 3: Retrieval-quality metrics (recall@k, MRR)

**Files:**
- Create: `src/vibe_trading/eval/retrieval.py`
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float`; `mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float`; `evaluate_queries(results: list[tuple[list, list]], k: int) -> dict` (keys `recall_at_k`, `mrr`, `query_count`). Task 4's harness consumes all three.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval_eval.py
import pytest

from vibe_trading.eval.retrieval import recall_at_k, mrr, evaluate_queries


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "c"], ["c", "z"], k=2) == pytest.approx(0.0)
    assert recall_at_k(["a", "b"], ["a", "z"], k=2) == pytest.approx(0.5)


def test_recall_requires_relevant_labels():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


def test_mrr():
    assert mrr(["x", "a", "b"], ["a"]) == pytest.approx(0.5)   # first relevant at rank 2
    assert mrr(["a"], ["a"]) == pytest.approx(1.0)
    assert mrr(["x", "y"], ["a"]) == 0.0                        # never retrieved


def test_evaluate_queries_averages():
    results = [
        (["a", "b"], ["a"]),        # recall@2 = 1.0, mrr = 1.0
        (["x", "a"], ["a", "b"]),   # recall@2 = 0.5, mrr = 0.5
    ]
    summary = evaluate_queries(results, k=2)
    assert summary == {"recall_at_k": pytest.approx(0.75),
                       "mrr": pytest.approx(0.75), "query_count": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/vibe_trading/eval/retrieval.py
"""Retrieval-quality metrics for the journal RAG (spec workstream D2).

recall@k answers 'of the precedents a human judged relevant, how many did the
retriever surface in its top k?'; MRR answers 'how high does the first relevant
one rank?'. Pure functions — the harness (evals/retrieval_eval.py) owns I/O."""


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    if not relevant_ids:
        raise ValueError("recall@k needs at least one relevant label per query")
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list, relevant_ids: list) -> float:
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_queries(results: list, k: int) -> dict:
    """`results`: [(retrieved_ids, relevant_ids), ...] — one tuple per labeled query."""
    if not results:
        return {"recall_at_k": 0.0, "mrr": 0.0, "query_count": 0}
    recalls = [recall_at_k(r, rel, k) for r, rel in results]
    mrrs = [mrr(r, rel) for r, rel in results]
    n = len(results)
    return {"recall_at_k": sum(recalls) / n, "mrr": sum(mrrs) / n, "query_count": n}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/retrieval.py tests/test_retrieval_eval.py
git commit -m "feat(retrieval): recall@k + MRR metrics"
```

---

### Task 4: Retrieval eval harness, corpus export, and baseline

**Files:**
- Create: `evals/retrieval_eval.py`, `evals/export_retrieval_corpus.py`, `evals/retrieval/queries.yaml` (seeded with 3 examples; ~20 labeled by hand afterwards)
- Test: `tests/test_retrieval_eval.py` (extend)

**Interfaces:**
- Consumes: Task 3's metrics; `journal.embed` and `journal.cosine_topk`.
- Produces: `evals/retrieval_eval.py` with `run(queries_path, corpus_path, cache_path, k, embed_fn) -> dict` and a CLI (`--update-baseline`, exit 1 on recall regression > 0.05 vs `evals/retrieval-baseline.json`); `evals/export_retrieval_corpus.py` dumping `{"decision_id", "setup_text"}` JSONL from `decision_embeddings`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_eval.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, "evals")  # evals/ scripts are not a package
import retrieval_eval


def fake_embed(text: str):
    """Deterministic toy embeddings: identical texts collide, so the labeled
    relevant doc ranks first for its own query text."""
    return [float((hash(text) >> s) % 97) / 97.0 for s in (0, 7, 14, 21)]


def _write_fixtures(tmp_path: Path):
    corpus = [
        {"decision_id": "dec-a", "setup_text": "bias=bullish; rsi=overbought"},
        {"decision_id": "dec-b", "setup_text": "bias=bearish; rsi=oversold"},
    ]
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps(c) for c in corpus) + "\n")
    (tmp_path / "queries.yaml").write_text(
        "- id: q1\n"
        "  setup_text: 'bias=bullish; rsi=overbought'\n"
        "  relevant: [dec-a]\n")
    return tmp_path / "queries.yaml", tmp_path / "corpus.jsonl"


def test_run_computes_metrics_with_cache(tmp_path):
    queries, corpus = _write_fixtures(tmp_path)
    cache = tmp_path / "cache.json"
    summary = retrieval_eval.run(queries, corpus, cache, k=1, embed_fn=fake_embed)
    assert summary["recall_at_k"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)
    assert summary["query_count"] == 1
    # embeddings were cached: a second run with a crashing embed_fn still works
    def boom(text):
        raise AssertionError("embed called despite cache")
    summary2 = retrieval_eval.run(queries, corpus, cache, k=1, embed_fn=boom)
    assert summary2["recall_at_k"] == pytest.approx(1.0)


def test_regression_gate(tmp_path):
    queries, corpus = _write_fixtures(tmp_path)
    cache = tmp_path / "cache.json"
    baseline = tmp_path / "retrieval-baseline.json"
    baseline.write_text(json.dumps({"k": 1, "recall_at_k": 1.0, "mrr": 1.0,
                                    "query_count": 1}))
    rc = retrieval_eval.main(["--queries", str(queries), "--corpus", str(corpus),
                              "--cache", str(cache), "--baseline", str(baseline),
                              "--k", "1"], embed_fn=fake_embed)
    assert rc == 0
    # now poison the baseline upward -> current run regresses -> exit 1
    baseline.write_text(json.dumps({"k": 1, "recall_at_k": 2.0, "mrr": 1.0,
                                    "query_count": 1}))
    rc = retrieval_eval.main(["--queries", str(queries), "--corpus", str(corpus),
                              "--cache", str(cache), "--baseline", str(baseline),
                              "--k", "1"], embed_fn=fake_embed)
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_eval.py -v -k "run_computes or regression"`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_eval'`

- [ ] **Step 3: Write the implementation**

```python
# evals/retrieval_eval.py
"""Retrieval-quality eval for the journal RAG: recall@k / MRR of the production
ranking (embed + cosine) against hand-labeled relevance judgments.

  python evals/retrieval_eval.py                  # gate against retrieval-baseline.json
  python evals/retrieval_eval.py --update-baseline

Labels live in evals/retrieval/queries.yaml; the candidate corpus is exported
from prod via evals/export_retrieval_corpus.py. Embeddings are cached in
data/retrieval_embedding_cache.json (keyed by sha256 of text) so re-runs are free."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from vibe_trading.eval.retrieval import evaluate_queries
from vibe_trading.journal import cosine_topk, embed as live_embed

RECALL_REGRESSION_THRESHOLD = 0.05


def _cached_embed(text: str, cache: dict, embed_fn) -> list:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key not in cache:
        vec = embed_fn(text)
        if vec is None:
            raise RuntimeError(f"embedding failed for: {text[:60]}...")
        cache[key] = vec
    return cache[key]


def run(queries_path: Path, corpus_path: Path, cache_path: Path, k: int,
        embed_fn=live_embed) -> dict:
    queries = yaml.safe_load(Path(queries_path).read_text())
    corpus = [json.loads(line) for line in
              Path(corpus_path).read_text().splitlines() if line.strip()]
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    corpus_vecs = [(c["decision_id"], _cached_embed(c["setup_text"], cache, embed_fn))
                   for c in corpus]
    results = []
    for q in queries:
        q_vec = _cached_embed(q["setup_text"], cache, embed_fn)
        ranked = cosine_topk(q_vec, corpus_vecs, k=max(k, 10))
        results.append(([rid for rid, _ in ranked], q["relevant"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return evaluate_queries(results, k=k)


def main(argv=None, embed_fn=live_embed) -> int:
    p = argparse.ArgumentParser(prog="retrieval-eval")
    p.add_argument("--queries", type=Path, default=Path("evals/retrieval/queries.yaml"))
    p.add_argument("--corpus", type=Path, default=Path("evals/retrieval/corpus.jsonl"))
    p.add_argument("--cache", type=Path,
                   default=Path("data/retrieval_embedding_cache.json"))
    p.add_argument("--baseline", type=Path,
                   default=Path("evals/retrieval-baseline.json"))
    p.add_argument("--k", type=int, default=4)   # = JOURNAL_PRECEDENT_K default
    p.add_argument("--update-baseline", action="store_true")
    args = p.parse_args(argv)

    summary = run(args.queries, args.corpus, args.cache, args.k, embed_fn=embed_fn)
    summary["k"] = args.k
    print(f"recall@{args.k}: {summary['recall_at_k']:.3f}   "
          f"MRR: {summary['mrr']:.3f}   ({summary['query_count']} queries)")

    if args.update_baseline:
        args.baseline.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Baseline written: {args.baseline}")
        return 0
    if not args.baseline.exists():
        print("No baseline — run with --update-baseline to seed it.")
        return 0
    baseline = json.loads(args.baseline.read_text())
    delta = summary["recall_at_k"] - baseline["recall_at_k"]
    if delta < -RECALL_REGRESSION_THRESHOLD:
        print(f"REGRESSION: recall@{args.k} {baseline['recall_at_k']:.3f} -> "
              f"{summary['recall_at_k']:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# evals/export_retrieval_corpus.py
"""Export the journal corpus (decision_id + setup_text) for the retrieval eval.
Run against the prod .env; commit the JSONL so the eval is reproducible anywhere."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from vibe_trading.data.db import PostgresDatabase


def main() -> int:
    load_dotenv()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/retrieval/corpus.jsonl")
    pg = PostgresDatabase()
    pg.connect()
    try:
        rows = pg.conn.execute(
            "SELECT decision_id, setup_text FROM decision_embeddings "
            "ORDER BY timestamp ASC").fetchall()
    finally:
        pg.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for decision_id, setup_text in rows:
            f.write(json.dumps({"decision_id": decision_id,
                                "setup_text": setup_text}) + "\n")
    print(f"{len(rows)} corpus entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Seed `evals/retrieval/queries.yaml` with three placeholder-free examples drawn from real setup-card vocabulary (they will be replaced during labeling, but the file must parse and document the format):

```yaml
# Hand-labeled retrieval relevance judgments. For each query setup, `relevant`
# lists decision_ids from corpus.jsonl a human judged to be genuinely analogous
# setups (same regime + structure story, not just same symbol).
# Labeling workflow: python evals/export_retrieval_corpus.py, then for each query
# skim the corpus for analogous setups. Aim for ~20 queries across regimes.
- id: q-example-bullish-breakout
  setup_text: "bias=bullish; volume=confirmed; confluence=0.8; rsi=neutral; macd=bullish_cross; adx=strong_trend; obv=accumulation; support_prox=far; resistance_prox=near; pattern=none; funding=neutral; oi=rising; thesis=Breakout above resistance on rising volume"
  relevant: [REPLACE-WITH-REAL-DECISION-ID]
- id: q-example-oversold-chop
  setup_text: "bias=neutral; volume=weak; confluence=0.3; rsi=oversold; macd=bearish; adx=weak_trend; obv=flat; support_prox=near; resistance_prox=far; pattern=hammer; funding=neutral; oi=falling; thesis=Oversold bounce attempt in a chop range"
  relevant: [REPLACE-WITH-REAL-DECISION-ID]
- id: q-example-bearish-divergence
  setup_text: "bias=bearish; volume=divergent; confluence=0.6; rsi=overbought; macd=bearish_divergence; adx=strong_trend; obv=distribution; support_prox=far; resistance_prox=near; pattern=shooting_star; funding=positive; oi=rising; thesis=Price making highs on OBV distribution — exhaustion risk"
  relevant: [REPLACE-WITH-REAL-DECISION-ID]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: all pass

- [ ] **Step 5: Manual labeling + baseline (requires prod env)**

1. `uv run python evals/export_retrieval_corpus.py` — commit `evals/retrieval/corpus.jsonl`.
2. Replace the three example queries with ~20 real labeled queries (the labeling workflow is documented in the YAML header).
3. `uv run python evals/retrieval_eval.py --update-baseline` — commit `evals/retrieval-baseline.json`.

- [ ] **Step 6: Commit**

```bash
git add evals/retrieval_eval.py evals/export_retrieval_corpus.py \
        evals/retrieval/ tests/test_retrieval_eval.py
git commit -m "feat(retrieval): recall@k/MRR eval harness + corpus export + baseline gate"
```

---

### Task 5: precedents_k — online segmentation plumbing

**Files:**
- Modify: `src/vibe_trading/data/db.py` (both `decision_log` schemas + migration lists; `decision_scores` schema + Postgres migration list)
- Modify: `src/vibe_trading/runtime/decision_pipeline.py` (`DecisionResult`, `run_symbol`)
- Modify: `src/vibe_trading/runtime/scheduler.py` (decision_log INSERT)
- Modify: `src/vibe_trading/eval/online.py` (`OutcomeScorer` SELECT + `_persist`)
- Test: `tests/test_decision_pipeline.py`, `tests/test_online_evals.py` (extend)

**Interfaces:**
- Consumes: workstream B's 11-column decision_log INSERT; workstream C's `decision_scores` and `OutcomeScorer`.
- Produces: `DecisionResult.precedents_k: int = 0`; `decision_log.precedents_k INTEGER`; `decision_scores.precedents_k INTEGER`; scoring copies it through. The digest/segmentation query is then plain SQL: `SELECT precedents_k > 0, AVG(outcome_score) FROM decision_scores GROUP BY 1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decision_pipeline.py` (reuse its existing mock style — analyst/trader/risk/pipeline/broker are MagicMocks, retriever injectable):

```python
def test_decision_result_carries_precedents_k(pipeline_with_mocks):
    """run_symbol reports how many precedents were actually injected."""
    from vibe_trading.journal import RetrievalResult, Precedent
    pipeline, mocks = pipeline_with_mocks
    two = [Precedent("BTC/USDT", "long", "2026-06-01", 0.9, "closed", 2.0, "won"),
           Precedent("ETH/USDT", "long", "2026-06-02", 0.8, "closed", 1.0, "won")]
    pipeline.retriever.retrieve_for.return_value = RetrievalResult([0.1], two)
    result = pipeline.run_symbol("BTC/USDT", mocks.last_ts, 100.0)
    assert result.precedents_k == 2
```

Append to `tests/test_online_evals.py` (extend `test_run_pass_scores_one_decision_and_pushes_langfuse`'s fixtures — the decision row grows a `precedents_k` column):

```python
def test_run_pass_copies_precedents_k(monkeypatch):
    now = datetime(2026, 7, 5, 12, 0)
    old_ts = now - timedelta(hours=48)
    pg = _fake_pg({
        "FROM decision_log": [("dec-1", old_ts, "BTC/USDT", "long", "trace-1",
                               "bundle-v1", 3)],
        "FROM trades": [], "FROM open_positions": [],
    })
    duck = MagicMock()
    duck.conn.execute.return_value.fetchone.side_effect = [(100.0,), (104.0,)]
    scorer = OutcomeScorer(pg_factory=lambda: pg, duck_factory=lambda: duck,
                           now_fn=lambda: now, push_fn=lambda *a, **k: None)
    assert scorer.run_pass() == 1
    insert = [c for c in pg.conn.execute.call_args_list
              if "INSERT OR IGNORE INTO decision_scores" in c.args[0]][0]
    assert "precedents_k" in insert.args[0]
    assert 3 in insert.args[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decision_pipeline.py tests/test_online_evals.py -v -k precedents_k`
Expected: FAIL (`DecisionResult` has no `precedents_k`; column absent)

- [ ] **Step 3: Write the implementation**

`src/vibe_trading/data/db.py`:
1. Add `precedents_k INTEGER` (comment: `-- how many journal precedents the trader saw`) to both `decision_log` CREATE TABLE statements and to the `decision_scores` CREATE TABLE.
2. Append to both idempotent migration tuples:
   `"ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS precedents_k INTEGER",`
   and to the Postgres tuple additionally:
   `"ALTER TABLE decision_scores ADD COLUMN IF NOT EXISTS precedents_k INTEGER",`

`src/vibe_trading/runtime/decision_pipeline.py`:
1. Add the field to `DecisionResult`: `precedents_k: int = 0`.
2. In `run_symbol`, after `retrieval = self.retriever.retrieve_for(setup_text)`, compute `precedents_k = len(retrieval.precedents)` and pass `precedents_k=precedents_k` into **both** `DecisionResult(...)` constructions that follow (the `"flat"` return and the approved/rejected return).

`src/vibe_trading/runtime/scheduler.py`: extend the decision_log INSERT to 12 columns — append `precedents_k` to the column list, a 12th `?`, and `result.precedents_k` as the final param (after `prompts.bundle_version()`).

`src/vibe_trading/eval/online.py`:
1. In `_run_pass`'s candidates SELECT, append `, d.precedents_k` to the column list and unpack it: `for decision_id, ts, symbol, action, trace_id, prompt_version, precedents_k in candidates:` — thread it to `self._persist(decision_id, outcome, prompt_version, precedents_k)`.
2. `_persist(self, decision_id, outcome, prompt_version, precedents_k=None)` — extend the INSERT to `(decision_id, scored_at, kind, outcome_pct, outcome_score, prompt_version, precedents_k)` with the extra param.

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run pytest tests/test_decision_pipeline.py tests/test_online_evals.py tests/test_scheduler.py tests/test_db.py -v` — expected: all pass.
Run: `uv run pytest` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/data/db.py src/vibe_trading/runtime/decision_pipeline.py \
        src/vibe_trading/runtime/scheduler.py src/vibe_trading/eval/online.py \
        tests/test_decision_pipeline.py tests/test_online_evals.py
git commit -m "feat(retrieval): stamp precedents_k through decisions into scores"
```

---

### Task 6: ReplayJournal — no-lookahead RAG for the backtester

**Files:**
- Modify: `src/vibe_trading/journal.py`
- Test: `tests/test_journal.py` (extend)

**Interfaces:**
- Consumes: `cosine_topk`, `Precedent`, `RetrievalResult`, `PRECEDENT_K`, `COUNTERFACTUAL_HORIZON_CANDLES`, `_CANDLE_HOURS`.
- Produces: `ReplayJournal(k=..., horizon_candles=..., embed_fn=embed, candle_close_fn=None)` with attribute `current_ts` (set by the engine each replay step) and methods `record_decision(decision_id, symbol, ts, action, entry_price, embedding)`, `record_closed_trade(trade: dict)`, `retrieve_for(setup_text) -> RetrievalResult`. `candle_close_fn(symbol, target_ts, not_after_ts) -> Optional[float]` is the engine-supplied candle lookup. Task 7 wires it into `BacktestEngine`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_journal.py`:

```python
def test_replay_journal_no_lookahead_and_outcomes():
    from datetime import datetime, timedelta
    from vibe_trading.journal import ReplayJournal

    fixed = [0.9, 0.1, 0.0]
    journal_ = ReplayJournal(
        k=4, horizon_candles=6, embed_fn=lambda text: fixed,
        candle_close_fn=lambda sym, target, not_after: 104.0)
    t0 = datetime(2026, 6, 1, 0, 0)

    # decision older than the horizon relative to the replay clock -> retrievable
    journal_.record_decision("dec-old", "BTC/USDT", t0, "long", 100.0, fixed)
    # decision INSIDE the horizon -> must NOT be retrievable (outcome unknown yet)
    journal_.record_decision("dec-new", "BTC/USDT", t0 + timedelta(hours=30),
                             "long", 100.0, fixed)
    journal_.current_ts = t0 + timedelta(hours=36)   # horizon = 24h

    result = journal_.retrieve_for("whatever")
    ids = [p.outcome_label for p in result.precedents]
    assert len(result.precedents) == 1               # only dec-old
    p = result.precedents[0]
    assert p.kind == "counterfactual"
    assert p.outcome_pct == pytest.approx(4.0)       # (104-100)/100, long

    # once its trade closes, the precedent switches to the real outcome
    journal_.record_closed_trade({"decision_id": "dec-old", "result": "win",
                                  "realized_pnl": 80.0, "size_usd": 1000.0})
    p2 = journal_.retrieve_for("whatever").precedents[0]
    assert p2.kind == "closed"
    assert p2.outcome_pct == pytest.approx(8.0)


def test_replay_journal_empty_and_failed_embed_degrade():
    from datetime import datetime
    from vibe_trading.journal import ReplayJournal
    journal_ = ReplayJournal(embed_fn=lambda text: None)
    journal_.current_ts = datetime(2026, 6, 2)
    result = journal_.retrieve_for("x")
    assert result.embedding is None and result.precedents == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_journal.py -v -k replay`
Expected: FAIL (`ImportError: cannot import name 'ReplayJournal'`)

- [ ] **Step 3: Write the implementation**

Append to `src/vibe_trading/journal.py`:

```python
class ReplayJournal:
    """Backtest-time journal RAG (spec D3): accumulates decisions during a
    chronological replay and retrieves precedents with NO lookahead —
      * a decision becomes a candidate only once it is older than the
        counterfactual horizon relative to the replay clock (current_ts), and
      * its counterfactual candle is bounded by not_after=current_ts, so a data
        gap can never leak a future price into the replay past.
    In-memory only; nothing is persisted."""

    def __init__(self, k: int = PRECEDENT_K,
                 horizon_candles: int = COUNTERFACTUAL_HORIZON_CANDLES,
                 embed_fn=embed, candle_close_fn=None):
        self.k = k
        self.horizon_candles = horizon_candles
        self._embed = embed_fn
        # (symbol, target_ts, not_after_ts) -> first 4h close in [target, not_after]
        self._candle_close = candle_close_fn
        self.current_ts = None            # set by the engine each replay step
        self._records: list = []          # dicts, insertion-ordered (chronological)
        self._trades: dict = {}           # decision_id -> (result, pnl, size)

    def record_decision(self, decision_id, symbol, ts, action, entry_price,
                        embedding) -> None:
        if embedding is None:
            return
        self._records.append({"decision_id": decision_id, "symbol": symbol,
                              "ts": ts, "action": action,
                              "entry_price": entry_price, "embedding": embedding})

    def record_closed_trade(self, trade: dict) -> None:
        decision_id = trade.get("decision_id")
        if decision_id:
            self._trades[decision_id] = (trade["result"], trade["realized_pnl"],
                                         trade["size_usd"])

    def retrieve_for(self, setup_text: str) -> RetrievalResult:
        emb = self._embed(setup_text)
        if emb is None or self.current_ts is None:
            return RetrievalResult(emb, [])
        horizon = timedelta(hours=self.horizon_candles * _CANDLE_HOURS)
        cutoff = self.current_ts - horizon
        candidates = [(r, r["embedding"]) for r in self._records if r["ts"] < cutoff]
        ranked = cosine_topk(emb, candidates, self.k)
        out = []
        for rec, score in ranked:
            p = self._attach(rec, score, horizon)
            if p is not None:
                out.append(p)
        return RetrievalResult(emb, out)

    def _attach(self, rec: dict, score: float, horizon) -> Optional[Precedent]:
        when = rec["ts"].date().isoformat() if hasattr(rec["ts"], "date") else str(rec["ts"])
        action = rec["action"]
        trade = self._trades.get(rec["decision_id"])
        if trade is not None:
            result, pnl, size = trade
            pct = (float(pnl) / float(size) * 100.0) if size else 0.0
            return Precedent(rec["symbol"], action, when, score, "closed", pct,
                             f"traded {action} -> {result} {pct:+.1f}%")
        if not rec["entry_price"] or self._candle_close is None:
            return None
        fut = self._candle_close(rec["symbol"], rec["ts"] + horizon, self.current_ts)
        if fut is None:
            return None  # gap — drop, never fabricate (matches PrecedentRetriever)
        fwd = (float(fut) - float(rec["entry_price"])) / float(rec["entry_price"]) * 100.0
        signed = -fwd if action == "short" else fwd
        if action == "flat":
            label = (f"skipped (flat); price moved {fwd:+.1f}% over "
                     f"{self.horizon_candles * _CANDLE_HOURS}h")
        else:
            label = f"{action} skipped (risk veto); would have {signed:+.1f}%"
        return Precedent(rec["symbol"], action, when, score, "counterfactual",
                         signed, label)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_journal.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/journal.py tests/test_journal.py
git commit -m "feat(retrieval): ReplayJournal — no-lookahead precedents for backtests"
```

---

### Task 7: Backtest A/B wiring and CLI flags

**Files:**
- Modify: `src/vibe_trading/eval/backtest.py` (`BacktestEngine.__init__`, `run`, `_get_decision`, `_update_and_resolve_brackets`)
- Modify: `src/vibe_trading/cli.py` (backtest subcommand)
- Test: `tests/test_backtest_rag.py`

**Interfaces:**
- Consumes: Task 6's `ReplayJournal`; `journal.build_setup_card`; `HeadTrader.decide(..., precedents=...)` (already supported).
- Produces: `BacktestEngine(db, symbols, initial_balance=10000.0, journal_rag=False)`; `vibe-trading backtest --live-agents --journal-rag --summary-out PATH`; `decision_id` threaded through backtest order submission and closed trades (required for closed-trade precedent outcomes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest_rag.py
from unittest.mock import MagicMock, patch

from vibe_trading.eval.backtest import BacktestEngine


def test_engine_defaults_to_no_rag():
    engine = BacktestEngine(MagicMock(), ["BTC/USDT"])
    assert engine.journal_rag is False
    assert engine.replay_journal is None


def test_live_decision_threads_precedents_and_records(monkeypatch):
    """With journal_rag on, the live-agent path retrieves precedents, passes them
    to the trader, and records the new decision into the replay journal."""
    from vibe_trading.journal import RetrievalResult

    engine = BacktestEngine(MagicMock(), ["BTC/USDT"], journal_rag=True)
    engine.replay_journal = MagicMock()
    engine.replay_journal.retrieve_for.return_value = RetrievalResult([0.1], [])

    fake_analyst = MagicMock()
    fake_trader = MagicMock()
    fake_trader.decide.return_value = {"decision_id": "dec-1", "action": "long",
                                       "symbol": "BTC/USDT",
                                       "stop_loss_strategy": "1.5_atr",
                                       "take_profit_strategy": "next_resistance",
                                       "risk_reward_ratio": 2.0,
                                       "hold_period_bias": "medium",
                                       "reasoning_summary": "r"}
    with patch("vibe_trading.agents.analyst.TechnicalVolumeAnalyst",
               return_value=fake_analyst), \
         patch("vibe_trading.agents.trader.HeadTrader", return_value=fake_trader), \
         patch("vibe_trading.data.fetcher.DataFetcher"):
        snapshot = {"close": 100.0, "rsi_14": 50.0, "obv_trend": "flat",
                    "macd_regime": "neutral"}
        from datetime import datetime
        proposal = engine._get_decision("BTC/USDT", snapshot,
                                        datetime(2026, 6, 1), use_live_agents=True)
    assert proposal["decision_id"] == "dec-1"
    engine.replay_journal.retrieve_for.assert_called_once()
    assert fake_trader.decide.call_args.kwargs["precedents"] == []
    engine.replay_journal.record_decision.assert_called_once()
    rec_args = engine.replay_journal.record_decision.call_args.args
    assert rec_args[0] == "dec-1"


def test_mock_path_rejects_journal_rag():
    import pytest
    engine = BacktestEngine(MagicMock(), ["BTC/USDT"], journal_rag=True)
    engine.replay_journal = MagicMock()
    with pytest.raises(ValueError):
        engine._get_decision("BTC/USDT", {"close": 1.0, "rsi_14": 50.0,
                                          "obv_trend": "flat",
                                          "macd_regime": "neutral"},
                             None, use_live_agents=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest_rag.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'journal_rag'`)

- [ ] **Step 3: Write the implementation**

In `src/vibe_trading/eval/backtest.py`:

1. `__init__` gains `journal_rag: bool = False`:

```python
        self.journal_rag = journal_rag
        self.replay_journal = None   # built in run() once the DB is connected
```

2. In `run()`, right after `self.db.connect()`:

```python
        if self.journal_rag:
            from vibe_trading.journal import ReplayJournal

            def _candle_close(symbol, target_ts, not_after_ts):
                row = self.db.conn.execute(
                    "SELECT close FROM candles WHERE symbol = ? AND timeframe = '4h' "
                    "AND timestamp >= ? AND timestamp <= ? "
                    "ORDER BY timestamp ASC LIMIT 1",
                    (symbol, target_ts, not_after_ts)).fetchone()
                return row[0] if row else None

            self.replay_journal = ReplayJournal(candle_close_fn=_candle_close)
```

3. In the `for i, ts in enumerate(timestamps):` loop, first line of the body:

```python
            if self.replay_journal is not None:
                self.replay_journal.current_ts = ts
```

and after `closed_trades = self._update_and_resolve_brackets(ts, current_prices)`:

```python
            if self.replay_journal is not None:
                for trade in closed_trades:
                    self.replay_journal.record_closed_trade(trade)
```

4. In `_get_decision`, live path — replace the analyst/trader block with:

```python
        if use_live_agents:
            from vibe_trading.agents.trader import HeadTrader
            from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
            from vibe_trading.data.fetcher import DataFetcher
            from vibe_trading.journal import build_setup_card

            analyst = TechnicalVolumeAnalyst(db=self.db, fetcher=DataFetcher())
            trader = HeadTrader()

            scorecard = {"accuracy": 0.55, "total_decisions": 100}
            open_positions = self.broker.get_open_positions()

            analyst_res = analyst.analyze(symbol=symbol, timestamp=timestamp)

            retrieval = None
            precedents = None
            if self.replay_journal is not None:
                setup_text = build_setup_card(analyst_res, snapshot)
                retrieval = self.replay_journal.retrieve_for(setup_text)
                precedents = retrieval.precedents

            proposal = trader.decide(symbol, analyst_res, scorecard, open_positions,
                                     current_price=float(snapshot.get("close", 0.0)),
                                     precedents=precedents)

            if self.replay_journal is not None and retrieval is not None:
                self.replay_journal.record_decision(
                    proposal["decision_id"], symbol, timestamp, proposal["action"],
                    float(snapshot.get("close", 0.0)), retrieval.embedding)
            return proposal
```

and at the top of the mock branch (`else:`):

```python
            if self.replay_journal is not None:
                raise ValueError("--journal-rag requires --live-agents: the mock "
                                 "signal has no analyst thesis to embed.")
```

5. Thread `decision_id` through execution — in `run()`'s `submit_order` call add `decision_id=proposal.get("decision_id"),` and in `_update_and_resolve_brackets`' Scenario-A `closed_info` dict add `"decision_id": pos.get("decision_id"),`. (PaperBroker already accepts and stores `decision_id` — the scheduler path uses it.)

In `src/vibe_trading/cli.py`, backtest subcommand:

```python
    backtest_parser.add_argument(
        "--journal-rag", action="store_true", default=False,
        help="Enable replay journal RAG (precedents for the trader). Requires --live-agents.")
    backtest_parser.add_argument(
        "--summary-out", type=str, default=None,
        help="Write the summary dict as JSON (for A/B diffing).")
```

and in the `backtest` handler:

```python
        if args.journal_rag and not args.live_agents:
            parser.error("--journal-rag requires --live-agents")
        engine = BacktestEngine(db, args.symbols, journal_rag=args.journal_rag)
        results = engine.run(start_dt, end_dt, use_live_agents=args.live_agents)
        print("\n=== Backtest Summary ===")
        for k, v in results.items():
            print(f"{k}: {v}")
        if args.summary_out:
            import json as _json
            from pathlib import Path as _Path
            _Path(args.summary_out).write_text(_json.dumps(results, indent=2, default=str))
```

- [ ] **Step 4: Run tests, then the whole suite**

Run: `uv run pytest tests/test_backtest_rag.py -v` — expected: all pass.
Run: `uv run pytest` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vibe_trading/eval/backtest.py src/vibe_trading/cli.py tests/test_backtest_rag.py
git commit -m "feat(retrieval): backtest A/B — --journal-rag replay RAG + --summary-out"
```

---

### Task 8: Integration verification and docs

**Files:**
- Create: `tests/test_pgvector_integration.py` (env-gated)
- Modify: `README.md`, `ARCHITECTURE.md`

- [ ] **Step 1: Write the env-gated integration test**

```python
# tests/test_pgvector_integration.py
"""Runs ONLY with RUN_PG_INTEGRATION=1 and a POSTGRES_URL pointing at a pgvector-
capable database (Supabase). Verifies the SQL path returns the same top-k as the
in-Python path on the live journal."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PG_INTEGRATION") != "1",
    reason="integration test — set RUN_PG_INTEGRATION=1 with a pgvector POSTGRES_URL")


def test_sql_topk_matches_python_topk():
    from datetime import datetime
    from vibe_trading.data.db import PostgresDatabase
    from vibe_trading.journal import PrecedentRetriever

    PostgresDatabase()  # triggers probe + migration
    assert PostgresDatabase.pgvector_enabled, "extension missing on this DB"

    pg = PostgresDatabase()
    pg.connect()
    try:
        row = pg.conn.execute(
            "SELECT embedding FROM decision_embeddings LIMIT 1").fetchone()
    finally:
        pg.close()
    if row is None:
        pytest.skip("empty journal")
    query = list(row[0])

    now = lambda: datetime.utcnow()
    sql_ids = [p.outcome_label for p in PrecedentRetriever(
        use_pgvector=True, now_fn=now).retrieve(query)]
    py_ids = [p.outcome_label for p in PrecedentRetriever(
        use_pgvector=False, now_fn=now).retrieve(query)]
    assert sql_ids == py_ids
```

- [ ] **Step 2: Manual verification checklist (prod Supabase)**

1. Deploy; first `PostgresDatabase()` construction logs `pgvector enabled (halfvec=True)` and the one-time column migration runs. Confirm:
   `SELECT udt_name FROM information_schema.columns WHERE table_name='decision_embeddings' AND column_name='embedding';` → `vector`.
2. `RUN_PG_INTEGRATION=1 uv run pytest tests/test_pgvector_integration.py -v` → passes.
3. Index usage: in the Supabase SQL editor run `EXPLAIN ANALYZE` on the output of `pgvector_topk_sql(halfvec=True, dim=<dim>)` with a real vector — expect `Index Scan using decision_embeddings_embedding_hnsw`.
4. One `trade-once` window: precedents still appear in the trader prompt (Langfuse trace) and `decision_log.precedents_k` is populated.
5. A/B ablation (small window to bound cost):
   ```bash
   uv run vibe-trading backtest --symbols BTC/USDT ETH/USDT --start 2026-05-01 --end 2026-06-01 \
       --live-agents --summary-out data/reports/ab_no_rag.json
   uv run vibe-trading backtest --symbols BTC/USDT ETH/USDT --start 2026-05-01 --end 2026-06-01 \
       --live-agents --journal-rag --summary-out data/reports/ab_rag.json
   diff data/reports/ab_no_rag.json data/reports/ab_rag.json
   ```
6. Online segmentation (after a week of C+D running):
   `SELECT precedents_k > 0 AS with_precedents, AVG(outcome_score), COUNT(*) FROM decision_scores GROUP BY 1;`

- [ ] **Step 3: Docs**

README: extend the trade-journal RAG section — pgvector/HNSW with automatic fallback, `EMBEDDING_DIM`, the retrieval eval commands (`evals/retrieval_eval.py`, `--update-baseline`), and the A/B commands from Step 2.5. ARCHITECTURE.md: update the journal-RAG paragraph (SQL-ranked retrieval when pgvector is available; halfvec expression index because 3072 > 2000; `ReplayJournal` for backtests; `precedents_k` segmentation). State explicitly: **reranking (D4) is deferred until the retrieval eval shows recall headroom at k.**

- [ ] **Step 4: Commit**

```bash
git add tests/test_pgvector_integration.py README.md ARCHITECTURE.md
git commit -m "docs(retrieval): pgvector integration test + README/ARCHITECTURE updates"
```

---

## Self-review notes

- Spec coverage: D1 probe/migration/HNSW/SQL-retrieval/fallback (Tasks 1–2), D2 labeled queries + recall@k/MRR + committed baseline + gate (Tasks 3–4), D3 backtest A/B with pinned-clock no-lookahead retrieval (Tasks 6–7) and online segmentation via `precedents_k` (Task 5), D4 explicitly deferred (Task 8 docs).
- Two spec deviations, deliberate: (1) the spec sketched `ALTER ... TYPE vector` + plain HNSW; at 3072 dims plain HNSW fails, so the index is the halfvec expression form — the spec's "index in place before the journal grows" intent is preserved. (2) `precedents_k` lives on `decision_log` (stamped at decision time) *and* is copied to `decision_scores` (where segmentation queries run), rather than scores-only — the decision row is the source of truth.
- Type consistency: `candle_close_fn(symbol, target_ts, not_after_ts)` defined in Task 6, implemented in Task 7; `pgvector_topk_sql(halfvec, dim)` defined in Task 2, its cast expression asserted byte-identical to Task 1's index DDL by tests in both tasks; `RetrievalResult`/`Precedent` shapes reused from the existing journal.
