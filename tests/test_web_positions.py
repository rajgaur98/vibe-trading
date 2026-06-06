"""Tests for the LIVE_TESTNET positions helper. We test the helper module directly
(no HTTP, no PostgresDatabase import), patching the cached broker so no ccxt/network
is touched."""
from unittest.mock import MagicMock

import vibe_trading.web.live_positions as lp


def test_live_testnet_positions_returns_exchange_positions(monkeypatch):
    fake_broker = MagicMock()
    fake_broker.get_open_positions.return_value = [{"symbol": "BTC/USDT", "side": "long"}]
    monkeypatch.setattr(lp, "_get_live_broker", lambda: fake_broker)
    assert lp.live_testnet_positions() == [{"symbol": "BTC/USDT", "side": "long"}]


def test_live_testnet_positions_returns_none_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("exchange unreachable")
    monkeypatch.setattr(lp, "_get_live_broker", _boom)
    assert lp.live_testnet_positions() is None  # None signals fallback to Postgres
