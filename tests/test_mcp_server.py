"""Tests for vibe_trading.mcp_server.

These tests are hermetic: no real database, network, or LLM is touched.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Import-time side-effect guard
# ---------------------------------------------------------------------------


def test_import_is_side_effect_free():
    """Importing mcp_server must NOT construct a Database/DataFetcher/ToolExecutor."""
    # We import inside the test so any construction attempt raises immediately
    # because the real Database.connect() would fail in CI (no DB path).
    # We assert _executor is still None after import.
    import vibe_trading.mcp_server as mod

    assert mod._executor is None, (
        "_executor should be None at import time — lazy construction violated"
    )


# ---------------------------------------------------------------------------
# 2. All 6 tools are registered on the FastMCP instance
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "get_candles",
    "get_indicators",
    "get_support_resistance",
    "get_candlestick_patterns",
    "get_derivatives",
    "get_market_sentiment",
}


def test_all_tools_registered_sync():
    """All 6 expected tools appear in the ToolManager (synchronous check)."""
    from vibe_trading.mcp_server import mcp

    registered = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert EXPECTED_TOOLS == registered, (
        f"Registered tools mismatch.\nExpected: {EXPECTED_TOOLS}\nGot: {registered}"
    )


def test_all_tools_registered_async():
    """All 6 expected tools appear via the async list_tools() API."""
    from vibe_trading.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    registered = {t.name for t in tools}
    assert EXPECTED_TOOLS == registered, (
        f"Registered tools mismatch (async).\nExpected: {EXPECTED_TOOLS}\nGot: {registered}"
    )


# ---------------------------------------------------------------------------
# 3. Tool wiring: monkeypatch _get_executor and verify delegation
# ---------------------------------------------------------------------------

SENTINEL = json.dumps({"test": "sentinel_value"})


def _make_mock_executor():
    mock_exec = MagicMock()
    mock_exec.execute.return_value = SENTINEL
    return mock_exec


def test_get_candles_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_candles("BTC/USDT", "4h", 10)

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with(
        "get_candles", {"symbol": "BTC/USDT", "timeframe": "4h", "limit": 10}
    )


def test_get_indicators_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_indicators("ETH/USDT", "1d")

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with(
        "get_indicators", {"symbol": "ETH/USDT", "timeframe": "1d"}
    )


def test_get_support_resistance_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_support_resistance("SOL/USDT")

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with(
        "get_support_resistance", {"symbol": "SOL/USDT"}
    )


def test_get_candlestick_patterns_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_candlestick_patterns("BNB/USDT")

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with(
        "get_candlestick_patterns", {"symbol": "BNB/USDT"}
    )


def test_get_derivatives_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_derivatives("BTC/USDT")

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with(
        "get_derivatives", {"symbol": "BTC/USDT"}
    )


def test_get_market_sentiment_delegates_to_executor():
    import vibe_trading.mcp_server as mod

    mock_exec = _make_mock_executor()
    with patch.object(mod, "_get_executor", return_value=mock_exec):
        result = mod.get_market_sentiment()

    assert result == SENTINEL
    mock_exec.execute.assert_called_once_with("get_market_sentiment", {})
