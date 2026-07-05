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
