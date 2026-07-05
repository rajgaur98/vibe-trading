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
