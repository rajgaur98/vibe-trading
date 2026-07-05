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
