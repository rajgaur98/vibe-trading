"""Unit tests for state_sync. The boto3 S3 client is replaced via monkeypatching
state_sync._client, so no network/credentials are needed."""
import os
from unittest.mock import MagicMock

from vibe_trading.runtime import state_sync


def test_pull_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("STATE_SYNC_BUCKET", raising=False)
    called = MagicMock()
    monkeypatch.setattr(state_sync, "_client", called)
    assert state_sync.pull() is False
    called.assert_not_called()  # never even builds a client


def test_pull_downloads_db_when_present(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    client = MagicMock()
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    assert state_sync.pull() is True
    client.download_file.assert_called_once_with("vibe-state", "vibe_trading.db", str(db_path))


def test_pull_returns_false_and_swallows_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "vibe_trading.db"))
    client = MagicMock()
    client.download_file.side_effect = Exception("NoSuchKey")
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    assert state_sync.pull() is False  # absent/failed object => fresh bootstrap, no raise


def test_push_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("STATE_SYNC_BUCKET", raising=False)
    called = MagicMock()
    monkeypatch.setattr(state_sync, "_client", called)
    state_sync.push()
    called.assert_not_called()


def test_push_uploads_db_and_audit_parquet(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    db_path.write_text("duckdb")
    audit_dir = tmp_path / "audit" / "decisions"
    audit_dir.mkdir(parents=True)
    (audit_dir / "a.parquet").write_text("p1")
    (audit_dir / "b.parquet").write_text("p2")

    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    client = MagicMock()
    monkeypatch.setattr(state_sync, "_client", lambda: client)

    state_sync.push()

    uploaded = {c.args[2] for c in client.upload_file.call_args_list}  # set of object keys
    assert "vibe_trading.db" in uploaded
    assert "audit/decisions/a.parquet" in uploaded
    assert "audit/decisions/b.parquet" in uploaded


def test_push_swallows_errors(monkeypatch, tmp_path):
    db_path = tmp_path / "vibe_trading.db"
    db_path.write_text("x")
    monkeypatch.setenv("STATE_SYNC_BUCKET", "vibe-state")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    client = MagicMock()
    client.upload_file.side_effect = Exception("network down")
    monkeypatch.setattr(state_sync, "_client", lambda: client)
    state_sync.push()  # must not raise
