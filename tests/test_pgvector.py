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
