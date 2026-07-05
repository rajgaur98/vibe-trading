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
