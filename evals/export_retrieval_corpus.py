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
