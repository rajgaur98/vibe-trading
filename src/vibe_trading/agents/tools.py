import json
import logging
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

from vibe_trading.data.db import Database
from vibe_trading.data.fetcher import DataFetcher
from vibe_trading.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

ANALYST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_candles",
            "description": (
                "Fetch historical OHLCV candles for a trading symbol ending at the analysis "
                "timestamp. Returns up to `limit` most recent candles (default 20, max 50)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTC/USDT'."},
                    "timeframe": {
                        "type": "string",
                        "enum": ["4h", "1d"],
                        "description": "Candle interval: '4h' or '1d'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of candles to return. Default 20, clamped to max 50.",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20,
                    },
                },
                "required": ["symbol", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": (
                "Compute technical indicators (RSI(14), MACD, ADX(14), OBV, SMA20/50/200) plus "
                "regime labels (overbought/oversold, strong/weak trend, accumulation/distribution) "
                "for the latest candle of the symbol on the specified timeframe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTC/USDT'."},
                    "timeframe": {"type": "string", "enum": ["4h", "1d"], "description": "Candle interval: '4h' or '1d'."},
                },
                "required": ["symbol", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_support_resistance",
            "description": (
                "Detect support and resistance levels from 4h price action using scipy peak "
                "detection. Returns nearest support/resistance prices plus distance and proximity "
                "labels (immediate_contact / very_close / near / far)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Trading pair, e.g. 'BTC/USDT'."}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candlestick_patterns",
            "description": (
                "Detect active candlestick patterns (engulfing, hammer, morning/evening star, "
                "shooting star) for the latest 4h candle of the symbol."
            ),
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Trading pair, e.g. 'BTC/USDT'."}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_derivatives",
            "description": (
                "Fetch funding rate (categorized as neutral / long_crowding / short_crowding) and "
                "open interest for the symbol's perpetual futures contract on Binance Futures."
            ),
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Trading pair, e.g. 'BTC/USDT'."}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": (
                "Fetch the current crypto Fear & Greed Index (0-100) with its classification "
                "label (Extreme Fear, Fear, Neutral, Greed, Extreme Greed). Reflects broad "
                "market mood, not symbol-specific."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ToolExecutor:
    """Dispatches LLM tool calls to Python handlers with try/except wrapping.

    Holds references to a `Database`, a `DataFetcher`, and an internal `FeaturePipeline`
    so handlers can reuse the existing indicator / S/R / candlestick logic without
    duplicating it. `current_timestamp` is set by the analyst before each tool-loop
    so that backtest replays do not query future candles (look-ahead bias).
    """

    def __init__(self, db: Database, fetcher: DataFetcher):
        self.db = db
        self.fetcher = fetcher
        self.pipeline = FeaturePipeline(db)
        self.current_timestamp: Optional[datetime] = None
        self._dispatch = {
            "get_candles": self._get_candles,
            "get_indicators": self._get_indicators,
            "get_support_resistance": self._get_support_resistance,
            "get_candlestick_patterns": self._get_candlestick_patterns,
            "get_derivatives": self._get_derivatives,
            "get_market_sentiment": self._get_market_sentiment,
        }

    def set_timestamp(self, ts: Optional[datetime]) -> None:
        """Pin the upper bound for candle queries (used by backtest replays)."""
        self.current_timestamp = ts

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool by name. Returns a JSON string (result or error)."""
        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    # ----- Handler stubs (implemented in later tasks) ---------------------

    def _get_candles(self, symbol: str, timeframe: str, limit: int = 20) -> list:
        raise NotImplementedError

    def _get_indicators(self, symbol: str, timeframe: str = "4h") -> dict:
        raise NotImplementedError

    def _get_support_resistance(self, symbol: str) -> dict:
        raise NotImplementedError

    def _get_candlestick_patterns(self, symbol: str) -> dict:
        raise NotImplementedError

    def _get_derivatives(self, symbol: str) -> dict:
        raise NotImplementedError

    def _get_market_sentiment(self) -> dict:
        raise NotImplementedError
