"""Unit tests for the DecisionPipeline graph-runner. All agent/broker/feature deps are
mocked, so these run with no network and no LLM calls — they verify the orchestration
graph (stage order, branching, what each stage receives), not the agents themselves."""
from unittest.mock import MagicMock

from vibe_trading.runtime.decision_pipeline import DecisionPipeline, DecisionResult


def _pipeline():
    analyst, trader, risk, fp, broker = (MagicMock() for _ in range(5))
    broker.get_open_positions.return_value = []
    broker.get_balance.return_value = 10000.0
    broker.peak_balance = 12000.0
    fp.run.return_value = {"close": 100.0}
    p = DecisionPipeline(analyst, trader, risk, fp, broker,
                         scorecard={"accuracy": 0.55}, trace_id_fn=lambda: "trace-1")
    return p, analyst, trader, risk, fp, broker


def test_analyst_runtimeerror_skips_symbol():
    p, analyst, trader, risk, fp, broker = _pipeline()
    analyst.analyze.side_effect = RuntimeError("tool loop exceeded")
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert res.status == "analyst_failed"
    trader.decide.assert_not_called()
    risk.evaluate_proposal.assert_not_called()


def test_empty_snapshot_skips_before_trader():
    p, analyst, trader, risk, fp, broker = _pipeline()
    fp.run.return_value = {}  # empty snapshot
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert res.status == "no_snapshot"
    assert res.analyst_report is analyst.analyze.return_value
    trader.decide.assert_not_called()


def test_flat_proposal_skips_risk():
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d1", "symbol": "BTC/USDT"}
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert res.status == "flat"
    assert res.proposal["action"] == "flat"
    assert res.trace_id == "trace-1"
    assert res.snapshot == {"close": 100.0}
    risk.evaluate_proposal.assert_not_called()


def test_approved_runs_risk_with_exec_price_and_broker_state():
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "long", "decision_id": "d2"}
    risk.evaluate_proposal.return_value = {"approved": True, "size_usd": 100.0}
    res = p.run_symbol("BTC/USDT", "ts", 250.0)
    assert res.status == "approved"
    assert res.risk_result["approved"] is True
    fp._get_candles.assert_called_once()
    _, kwargs = risk.evaluate_proposal.call_args
    assert kwargs["current_price"] == 250.0           # exec price (futures mark), not spot
    assert kwargs["account_balance"] == 10000.0        # lazy get_balance only on non-flat
    assert kwargs["peak_balance"] == 12000.0
    # trader saw the same exec price
    assert trader.decide.call_args.kwargs["current_price"] == 250.0


def test_pipeline_retrieves_and_threads_precedents():
    from vibe_trading.journal import RetrievalResult
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d1"}
    retriever = MagicMock()
    retriever.retrieve_for.return_value = RetrievalResult([0.1, 0.2], ["PRECEDENT_OBJ"])
    p.retriever = retriever

    res = p.run_symbol("BTC/USDT", "ts", 100.0)

    assert isinstance(retriever.retrieve_for.call_args.args[0], str)   # setup card text
    assert trader.decide.call_args.kwargs["precedents"] == ["PRECEDENT_OBJ"]
    assert res.setup_embedding == [0.1, 0.2]
    assert isinstance(res.setup_text, str)


def test_pipeline_defaults_to_noop_retriever():
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d1"}
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert trader.decide.call_args.kwargs["precedents"] == []
    assert res.setup_embedding is None


def test_rejected_status():
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "short", "decision_id": "d3"}
    risk.evaluate_proposal.return_value = {"approved": False, "reason": "below $10 min"}
    res = p.run_symbol("BTC/USDT", "ts", 100.0)
    assert res.status == "rejected"
    assert res.risk_result["reason"] == "below $10 min"


def test_flat_never_calls_get_balance():
    """Preserve the original laziness: balance is only fetched when risk runs (non-flat)."""
    p, analyst, trader, risk, fp, broker = _pipeline()
    trader.decide.return_value = {"action": "flat", "decision_id": "d4"}
    p.run_symbol("BTC/USDT", "ts", 100.0)
    broker.get_balance.assert_not_called()
