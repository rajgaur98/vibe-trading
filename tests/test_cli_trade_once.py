"""Unit test for the execute_trade_once orchestration: pull before evaluate, ping
on success, push always (finally). TradingScheduler/state_sync/monitoring/langfuse
are all monkeypatched, so nothing real runs."""
from unittest.mock import MagicMock

from vibe_trading import cli


def _patch(monkeypatch, evaluate_raises=False):
    calls = []

    def _evaluate():
        calls.append("evaluate")
        if evaluate_raises:
            raise RuntimeError("boom")

    scheduler = MagicMock()
    scheduler.sync_and_evaluate.side_effect = _evaluate
    monkeypatch.setattr(cli, "TradingScheduler", lambda symbols: scheduler)
    monkeypatch.setattr(cli.state_sync, "pull", lambda: calls.append("pull"))
    monkeypatch.setattr(cli.state_sync, "push", lambda: calls.append("push"))
    monkeypatch.setattr(cli.monitoring, "ping_healthcheck", lambda success=True: calls.append(f"ping:{success}"))
    monkeypatch.setattr(cli, "_flush_langfuse", lambda: calls.append("flush"))
    return calls


def test_execute_trade_once_happy_path_order(monkeypatch):
    calls = _patch(monkeypatch)
    cli.execute_trade_once([])
    assert calls[0] == "pull"            # warm state first
    assert "evaluate" in calls
    assert "ping:True" in calls          # pinged on success
    assert calls[-1] == "push"           # state pushed last (finally)


def test_execute_trade_once_pushes_even_on_failure(monkeypatch):
    calls = _patch(monkeypatch, evaluate_raises=True)
    try:
        cli.execute_trade_once([])
    except RuntimeError:
        pass
    assert "ping:True" not in calls      # NOT pinged => dead-man's-switch fires
    assert calls[-1] == "push"           # still pushed in finally
