from unittest.mock import MagicMock

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


class FakeConnWrapper:
    """Stands in for PostgresConnectionWrapper around a fake psycopg2 connection.
    `.execute(sql, params=None)` dispatches a canned fetchone() result by SQL
    substring match (checked in the order registered); `.commit`/`.rollback`
    are recorded so tests can assert on transaction boundaries."""

    def __init__(self, responses):
        # list of (substring, result) checked in order; result is whatever
        # fetchone() should return for that statement.
        self._responses = list(responses)
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, sql, params=None):
        self.executed.append(sql)
        for substring, outcome in self._responses:
            if substring in sql:
                if isinstance(outcome, Exception):
                    raise outcome
                self._last_result = outcome
                return self
        self._last_result = None
        return self

    def fetchone(self):
        return self._last_result

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _new_db(fake_conn, monkeypatch):
    monkeypatch.setattr(PostgresDatabase, "pgvector_enabled", False)
    monkeypatch.setattr(PostgresDatabase, "pgvector_halfvec", False)
    db = PostgresDatabase.__new__(PostgresDatabase)
    db.conn = fake_conn
    return db


def test_enable_pgvector_commits_column_migration_before_index_failure(monkeypatch):
    # Extension present, column starts as _float8 (unmigrated), halfvec type
    # absent (pre-0.7 pgvector) so the HNSW index CREATE raises. Simulate the
    # udt_name re-query flipping to 'vector' once the ALTER has "run".
    udt_calls = {"n": 0}

    class Conn(FakeConnWrapper):
        def execute(self, sql, params=None):
            if "udt_name FROM information_schema.columns" in sql:
                udt_calls["n"] += 1
                # First read (pre-ALTER): still float8[]. Every read after the
                # ALTER statement has executed: migrated to vector.
                altered = any(
                    "ALTER COLUMN embedding TYPE vector" in s for s in self.executed
                )
                self.executed.append(sql)
                self._last_result = ("vector",) if altered else ("_float8",)
                return self
            return super().execute(sql, params)

    conn = Conn([
        ("CREATE EXTENSION", None),
        ("pg_extension WHERE extname", (1,)),
        ("array_length(embedding, 1)", (3072,)),
        ("CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw",
         RuntimeError("halfvec type does not exist")),
        ("pg_type WHERE typname = 'halfvec'", None),  # halfvec absent
    ])
    db = _new_db(conn, monkeypatch)

    db._enable_pgvector()

    alter_calls = [s for s in conn.executed if "ALTER COLUMN embedding TYPE vector" in s]
    index_calls = [s for s in conn.executed if "CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw" in s]
    assert len(alter_calls) == 1, "column ALTER should have been executed"
    assert len(index_calls) == 1, "index CREATE should have been attempted"
    # A commit happened after the ALTER (and before the failing index CREATE).
    assert conn.commits >= 1
    # The index failure was rolled back.
    assert conn.rollbacks >= 1
    # Real state wins: column reports 'vector' after the ALTER -> enabled True.
    assert PostgresDatabase.pgvector_enabled is True
    # halfvec type absent -> flag reflects that regardless of index outcome.
    assert PostgresDatabase.pgvector_halfvec is False


def test_enable_pgvector_column_survives_index_failure_even_when_migration_fails(monkeypatch):
    # If the ALTER itself fails (not just the index), pgvector_enabled must
    # stay False because the column is still float8[] -- no code path may
    # claim pgvector is enabled when the column wasn't actually migrated.
    conn = FakeConnWrapper([
        ("CREATE EXTENSION", None),
        ("pg_extension WHERE extname", (1,)),
        ("udt_name FROM information_schema.columns", ("_float8",)),
        ("array_length(embedding, 1)", (3072,)),
        ("ALTER COLUMN embedding TYPE vector", RuntimeError("permission denied")),
        ("pg_type WHERE typname = 'halfvec'", None),
    ])
    db = _new_db(conn, monkeypatch)

    db._enable_pgvector()

    assert conn.rollbacks >= 1
    assert PostgresDatabase.pgvector_enabled is False
    assert PostgresDatabase.pgvector_halfvec is False
    # Index CREATE must never be attempted once the column migration failed.
    assert not any("CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw" in s for s in conn.executed)


def test_enable_pgvector_no_extension_skips_migration(monkeypatch):
    conn = FakeConnWrapper([
        ("CREATE EXTENSION", RuntimeError("permission denied to create extension")),
        ("pg_extension WHERE extname", None),  # extension not present
    ])
    db = _new_db(conn, monkeypatch)

    db._enable_pgvector()

    assert not any("ALTER COLUMN embedding TYPE vector" in s for s in conn.executed)
    assert not any("CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw" in s for s in conn.executed)
    assert PostgresDatabase.pgvector_enabled is False
    assert PostgresDatabase.pgvector_halfvec is False
    assert conn.rollbacks >= 1  # the failed CREATE EXTENSION was rolled back


def test_enable_pgvector_column_already_vector_skips_alter(monkeypatch):
    conn = FakeConnWrapper([
        ("CREATE EXTENSION", None),
        ("pg_extension WHERE extname", (1,)),
        ("udt_name FROM information_schema.columns", ("vector",)),
        ("pg_type WHERE typname = 'halfvec'", (1,)),
    ])
    db = _new_db(conn, monkeypatch)

    db._enable_pgvector()

    assert not any("ALTER COLUMN embedding TYPE vector" in s for s in conn.executed)
    assert not any("CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw" in s for s in conn.executed)
    assert PostgresDatabase.pgvector_enabled is True
    assert PostgresDatabase.pgvector_halfvec is True


def test_enable_pgvector_never_raises_on_unexpected_error(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("connection reset")
    db = _new_db(conn, monkeypatch)

    db._enable_pgvector()  # must not raise

    assert PostgresDatabase.pgvector_enabled is False
    assert PostgresDatabase.pgvector_halfvec is False
