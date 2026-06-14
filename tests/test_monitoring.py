"""Unit tests for monitoring.ping_healthcheck. requests.get is monkeypatched, so
no network is used."""
from unittest.mock import MagicMock

from vibe_trading.runtime import monitoring


def test_ping_noop_when_url_unset(monkeypatch):
    monkeypatch.delenv("HEALTHCHECK_PING_URL", raising=False)
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)
    get.assert_not_called()


def test_ping_success_gets_base_url(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc")
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)
    assert get.call_args.args[0] == "https://hc.example/abc"


def test_ping_failure_appends_fail(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc/")
    get = MagicMock()
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=False)
    assert get.call_args.args[0] == "https://hc.example/abc/fail"


def test_ping_swallows_network_errors(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_PING_URL", "https://hc.example/abc")
    get = MagicMock(side_effect=Exception("timeout"))
    monkeypatch.setattr(monitoring.requests, "get", get)
    monitoring.ping_healthcheck(success=True)  # must not raise
