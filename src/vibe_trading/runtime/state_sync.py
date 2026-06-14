"""Object-storage cache for the DuckDB candle DB + Parquet audit corpus, so an
ephemeral scheduled `trade-once` container starts warm and never loses the
(non-reproducible) audit records. S3-compatible (Cloudflare R2 / S3 / B2 / GCS).

Entirely gated on STATE_SYNC_BUCKET: unset => every function is a no-op, so the
local `live`/`trade-once` paths and the test suite are unaffected. Never raises —
a sync failure must never break trading (decisions are durable in Postgres).
"""
import os
import glob
import logging

from vibe_trading.audit import DEFAULT_AUDIT_DIR

logger = logging.getLogger(__name__)

CANDLE_DB_KEY = "vibe_trading.db"          # object key for the candle DB
AUDIT_KEY_PREFIX = "audit/decisions/"      # object key prefix for audit parquet


def _bucket():
    return os.getenv("STATE_SYNC_BUCKET") or None


def _db_path():
    return os.getenv("DATABASE_PATH", "data/vibe_trading.db")


def _audit_decisions_dir():
    return os.path.join(os.getenv("AUDIT_DIR", DEFAULT_AUDIT_DIR), "decisions")


def _client():
    """Build an S3-compatible client from env (endpoint + creds)."""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("STATE_SYNC_ENDPOINT") or None,
        aws_access_key_id=os.getenv("STATE_SYNC_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("STATE_SYNC_SECRET_ACCESS_KEY") or None,
    )


def pull() -> bool:
    """Download the candle DB from the bucket to its local path. Returns True if a
    file was downloaded, False if sync is disabled or the object is absent/failed
    (the caller's bootstrap_if_needed will then fetch fresh candles)."""
    bucket = _bucket()
    if not bucket:
        return False
    db_path = _db_path()
    try:
        client = _client()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        client.download_file(bucket, CANDLE_DB_KEY, db_path)
        logger.info(f"state_sync: pulled {CANDLE_DB_KEY} -> {db_path}")
        return True
    except Exception as e:
        logger.warning(f"state_sync: pull skipped/failed ({e}); will bootstrap fresh candles")
        return False


def push() -> None:
    """Upload the candle DB and every audit Parquet back to the bucket. No-op when
    sync is disabled. Swallows all errors (decisions remain durable in Postgres)."""
    bucket = _bucket()
    if not bucket:
        return
    try:
        client = _client()
        db_path = _db_path()
        if os.path.exists(db_path):
            client.upload_file(db_path, bucket, CANDLE_DB_KEY)
        for path in glob.glob(os.path.join(_audit_decisions_dir(), "*.parquet")):
            client.upload_file(path, bucket, AUDIT_KEY_PREFIX + os.path.basename(path))
        logger.info("state_sync: pushed candle DB + audit parquet")
    except Exception as e:
        logger.warning(f"state_sync: push failed ({e}); decisions remain durable in Postgres")
