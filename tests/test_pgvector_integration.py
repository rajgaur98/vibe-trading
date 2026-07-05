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
