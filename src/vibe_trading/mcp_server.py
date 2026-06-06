"""MCP server for vibe-trading market-data tools.

Run with:
    python -m vibe_trading.mcp_server

This server exposes the analyst's 6 market-data tools over the MCP stdio transport
so any MCP client (e.g. Claude Code) can drive the project's market-data layer:

    - get_candles          : historical OHLCV candles for a symbol/timeframe
    - get_indicators       : RSI, MACD, ADX, OBV, SMA technical indicators
    - get_support_resistance: nearest support and resistance price levels
    - get_candlestick_patterns: active candlestick pattern detection
    - get_derivatives      : funding rate and open interest for a futures symbol
    - get_market_sentiment : crypto Fear & Greed Index

All tools return JSON strings. The underlying ToolExecutor is constructed lazily on
the first tool call so that importing this module is side-effect-free and does not
require a live database or network connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vibe-trading")

# ---------------------------------------------------------------------------
# Lazy executor construction
# ---------------------------------------------------------------------------

_executor = None


def _get_executor():
    """Return the shared ToolExecutor, constructing it on first call."""
    global _executor
    if _executor is None:
        from vibe_trading.agents.tools import ToolExecutor
        from vibe_trading.data.db import Database
        from vibe_trading.data.fetcher import DataFetcher

        db = Database()
        db.connect()
        fetcher = DataFetcher()
        _executor = ToolExecutor(db, fetcher)
    return _executor


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@mcp.tool()
def get_candles(symbol: str, timeframe: str = "4h", limit: int = 20) -> str:
    """Fetch historical OHLCV candles for a trading symbol.

    Returns up to `limit` most recent candles (default 20, max 50).

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        timeframe: Candle interval: '4h' or '1d'.
        limit: Number of candles to return (1-50, default 20).
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_candles", {"symbol": symbol, "timeframe": timeframe, "limit": limit})


@mcp.tool()
def get_indicators(symbol: str, timeframe: str = "4h") -> str:
    """Compute technical indicators for the latest candle.

    Computes RSI(14), MACD, ADX(14), OBV, SMA20/50/200 plus regime labels
    (overbought/oversold, strong/weak trend, accumulation/distribution).

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        timeframe: Candle interval: '4h' or '1d'.
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_indicators", {"symbol": symbol, "timeframe": timeframe})


@mcp.tool()
def get_support_resistance(symbol: str) -> str:
    """Detect support and resistance levels from 4h price action.

    Uses scipy peak detection to find nearest support/resistance prices,
    returning distance percentages and proximity labels (immediate_contact /
    very_close / near / far).

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_support_resistance", {"symbol": symbol})


@mcp.tool()
def get_candlestick_patterns(symbol: str) -> str:
    """Detect active candlestick patterns for the latest 4h candle.

    Detects engulfing, hammer, morning/evening star, and shooting star patterns.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_candlestick_patterns", {"symbol": symbol})


@mcp.tool()
def get_derivatives(symbol: str) -> str:
    """Fetch funding rate and open interest for a perpetual futures contract.

    Returns funding rate (categorized as neutral / long_crowding / short_crowding)
    and open interest from Binance Futures.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_derivatives", {"symbol": symbol})


@mcp.tool()
def get_market_sentiment() -> str:
    """Fetch the current crypto Fear & Greed Index.

    Returns the index value (0-100) with its classification label:
    Extreme Fear, Fear, Neutral, Greed, or Extreme Greed.
    Reflects broad market mood, not symbol-specific sentiment.
    """
    executor = _get_executor()
    executor.set_timestamp(datetime.now(timezone.utc))
    return executor.execute("get_market_sentiment", {})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server on stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
