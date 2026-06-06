"""Append-only Parquet audit log for trading decisions.

Every decision the system makes (including FLAT) is written as ONE immutable
Parquet file under ``{audit_dir}/decisions/``. Files are never modified after
they are written — the log is strictly append-only — so the full corpus of
decisions can be queried later with DuckDB, e.g.::

    query_decisions("WHERE action='long' AND risk_reward_ratio >= 2.0")

Unlike ``decision_log.agent_transcripts`` (which historically stored the feature
snapshot, not the actual agent reasoning), the ``agent_transcripts`` field here
carries a JSON dump of the REAL analyst + trader outputs, and the deterministic
feature ``snapshot`` is kept in its own separate column.

Design notes:
- DuckDB is the only dependency (pyarrow is NOT required). We build a one-row
  in-memory relation from the record and ``COPY ... TO '<file>.parquet'``.
- Dict / list values are JSON-encoded to strings so every column is a flat
  scalar that Parquet stores cleanly and DuckDB can SELECT back.
- ``append_decision`` never raises into the caller — any failure is logged and
  swallowed, because the trading loop must not be brought down by an audit write.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = "data/audit"


def _json_default(value):
    """Fallback encoder for values json.dumps can't handle natively."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _flatten(record: dict) -> dict:
    """Return a copy of `record` with every value reduced to a Parquet-friendly
    scalar. dict/list values are JSON-encoded; datetimes are ISO strings;
    everything else is coerced to str unless it's already a (str/int/float/bool/None)."""
    flat: dict = {}
    for key, value in record.items():
        col = str(key)
        if isinstance(value, (dict, list)):
            flat[col] = json.dumps(value, default=_json_default, sort_keys=True)
        elif isinstance(value, datetime):
            flat[col] = value.isoformat()
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[col] = value
        else:
            # Decimal, UUID, pydantic, etc. — stringify so Parquet has a scalar.
            flat[col] = str(value)
    return flat


def _safe_fragment(value: str, fallback: str) -> str:
    """Sanitize a value for use inside a filename (keep it filesystem-safe)."""
    text = str(value) if value is not None else fallback
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)
    return safe or fallback


def append_decision(record: dict, audit_dir: str = DEFAULT_AUDIT_DIR) -> None:
    """Append ONE immutable Parquet file for `record` to ``{audit_dir}/decisions/``.

    The file is named ``{timestamp}__{decision_id}.parquet`` (a fresh uuid suffix
    guards against same-millisecond collisions), so existing files are never
    touched. Never raises — failures are logged and swallowed so the trading loop
    is unaffected.
    """
    try:
        decisions_dir = os.path.join(audit_dir, "decisions")
        os.makedirs(decisions_dir, exist_ok=True)

        flat = _flatten(record)

        ts = record.get("timestamp")
        if isinstance(ts, datetime):
            ts_frag = ts.strftime("%Y%m%dT%H%M%S%f")
        else:
            ts_frag = _safe_fragment(ts, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        id_frag = _safe_fragment(record.get("decision_id"), uuid.uuid4().hex)
        # uuid suffix prevents same-(ts,id) collisions from clobbering a prior file.
        filename = f"{ts_frag}__{id_frag}__{uuid.uuid4().hex[:8]}.parquet"
        out_path = os.path.join(decisions_dir, filename)

        # Build a one-row relation from the flattened record and COPY it out as Parquet.
        # Using a parameterized VALUES list keeps the SQL injection-safe and dialect-clean.
        columns = list(flat.keys())
        col_idents = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join("?" for _ in columns)
        values = [flat[c] for c in columns]

        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                f"CREATE TABLE rec AS SELECT * FROM (VALUES ({placeholders})) AS t({col_idents})",
                values,
            )
            # Escape single quotes in the path for the COPY literal.
            safe_path = out_path.replace("'", "''")
            con.execute(f"COPY rec TO '{safe_path}' (FORMAT PARQUET)")
        finally:
            con.close()
    except Exception as e:  # never propagate into the trading loop
        logger.error(f"audit.append_decision failed (decision not persisted to Parquet): {e}")


def query_decisions(where_sql: str = None, audit_dir: str = DEFAULT_AUDIT_DIR) -> list:
    """Query the audit log, returning a list of dicts (one per decision).

    `where_sql` is an optional full WHERE clause, e.g. ``"WHERE action='flat'"``.
    Returns ``[]`` when no decision files exist. Returns ``[]`` on query error.
    """
    decisions_dir = os.path.join(audit_dir, "decisions")
    if not os.path.isdir(decisions_dir):
        return []
    # Any parquet files present?
    if not any(f.endswith(".parquet") for f in os.listdir(decisions_dir)):
        return []

    glob = os.path.join(decisions_dir, "*.parquet").replace("'", "''")
    sql = f"SELECT * FROM read_parquet('{glob}')"
    if where_sql:
        sql += " " + where_sql

    try:
        rel = duckdb.sql(sql)
        cols = rel.columns
        return [dict(zip(cols, row)) for row in rel.fetchall()]
    except Exception as e:
        logger.error(f"audit.query_decisions failed: {e}")
        return []
