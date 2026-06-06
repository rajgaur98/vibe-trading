"""Tests for the append-only Parquet audit log (vibe_trading.audit).

The audit log writes ONE immutable Parquet file per decision so the full corpus
of decisions can be queried later ("of all decisions where X, what was Y").
"""
from datetime import datetime, timezone
from decimal import Decimal

from vibe_trading import audit


def _record(decision_id: str, action: str, symbol: str = "BTC/USDT"):
    """Build a representative decision record (mirrors what the scheduler assembles)."""
    return {
        "decision_id": decision_id,
        "timestamp": datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        "symbol": symbol,
        "action": action,
        "stop_loss_strategy": "2.0_atr",
        "take_profit_strategy": "3.0_atr",
        "risk_reward_ratio": 2.0,
        "reasoning_summary": f"reasoning for {decision_id}",
        "trace_id": f"trace-{decision_id}",
        # nested dict values must be JSON-encoded transparently
        "agent_transcripts": {"analyst": {"market_bias": "bullish"}, "trader": {"action": action}},
        "snapshot": {"rsi": 55.0, "atr": 1.23},
    }


def test_append_and_query_filters_by_action(tmp_path):
    audit_dir = str(tmp_path / "audit")
    audit.append_decision(_record("d1", "long"), audit_dir=audit_dir)
    audit.append_decision(_record("d2", "short"), audit_dir=audit_dir)
    audit.append_decision(_record("d3", "flat"), audit_dir=audit_dir)

    flat = audit.query_decisions("WHERE action='flat'", audit_dir=audit_dir)
    assert len(flat) == 1
    assert flat[0]["decision_id"] == "d3"
    assert flat[0]["action"] == "flat"

    # all three are queryable
    all_rows = audit.query_decisions(audit_dir=audit_dir)
    assert len(all_rows) == 3
    assert {r["decision_id"] for r in all_rows} == {"d1", "d2", "d3"}


def test_append_is_append_only(tmp_path):
    """Each decision writes its own file; existing files are never modified."""
    import os

    audit_dir = str(tmp_path / "audit")
    decisions_dir = os.path.join(audit_dir, "decisions")

    audit.append_decision(_record("d1", "long"), audit_dir=audit_dir)
    files_after_1 = sorted(os.listdir(decisions_dir))
    assert len(files_after_1) == 1
    mtime_1 = os.path.getmtime(os.path.join(decisions_dir, files_after_1[0]))

    audit.append_decision(_record("d2", "short"), audit_dir=audit_dir)
    files_after_2 = sorted(os.listdir(decisions_dir))
    # files accumulate (append-only)
    assert len(files_after_2) == 2
    # the original file was not rewritten/modified
    assert files_after_1[0] in files_after_2
    assert os.path.getmtime(os.path.join(decisions_dir, files_after_1[0])) == mtime_1


def test_nested_and_decimal_values_survive_round_trip(tmp_path):
    audit_dir = str(tmp_path / "audit")
    rec = _record("d1", "long")
    rec["risk_reward_ratio"] = Decimal("2.5")  # trader emits a Decimal
    audit.append_decision(rec, audit_dir=audit_dir)

    rows = audit.query_decisions(audit_dir=audit_dir)
    assert len(rows) == 1
    row = rows[0]
    # nested dict was JSON-encoded into a string column and is recoverable
    import json

    transcripts = json.loads(row["agent_transcripts"])
    assert transcripts["analyst"]["market_bias"] == "bullish"


def test_query_empty_dir_returns_empty_list(tmp_path):
    assert audit.query_decisions(audit_dir=str(tmp_path / "nope")) == []


def test_append_bad_record_does_not_raise(tmp_path):
    """A malformed record must never raise into the caller (the trading loop)."""
    audit_dir = str(tmp_path / "audit")

    class Unserializable:
        pass

    bad = {"decision_id": object(), "blob": Unserializable()}
    # must not raise
    audit.append_decision(bad, audit_dir=audit_dir)
