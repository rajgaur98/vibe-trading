from pydantic import BaseModel, Field
from typing import Literal
import json
import os
from vibe_trading.agents.client import GeminiClient

class AnalystOutput(BaseModel):
    market_bias: Literal["bullish", "bearish", "neutral"] = Field(
        description="The overall market trend direction identified from price, indicators, and structure."
    )
    volume_confirmation: Literal["confirmed", "divergent", "weak"] = Field(
        description="How volume behavior relates to the current price movement (e.g., rising volume on rally = confirmed)."
    )
    thesis: str = Field(
        description="A detailed paragraph summarizing the confluence of indicators, candlestick signals, and S/R levels."
    )
    nearest_support: float = Field(description="The closest valid support level identified from fractal pivots.")
    nearest_resistance: float = Field(description="The closest valid resistance level identified from fractal pivots.")
    confluence_score: float = Field(
        description="A value between 0.0 and 1.0 indicating the ratio of indicators supporting the overall bias."
    )

class TechnicalVolumeAnalyst:
    def __init__(self, client: GeminiClient = None):
        self.client = client or GeminiClient()
        self.model = os.getenv("GEMINI_ANALYST_MODEL", "gemini-3.5-flash")
        
        self.system_instruction = """
You are an elite Crypto Technical and Volume Analyst specializing in swing trading. 
Your objective is to evaluate a market snapshot and produce a structured technical thesis.

Analyze the following inputs:
1. Trend Structure (Moving averages stacking, ADX strength).
2. Momentum (RSI values and MACD hist expansions).
3. Volume Confirmation (OBV trends and volume spike confirmations).
4. Price Patterns & Candlesticks (Engulfing candles, morning/evening stars near support/resistance).
5. Derivatives Health (Funding rate limits, open interest trends).

Remember the classic Murphy principles:
- Volume must confirm the price trend (rising volume on breakouts, falling volume on pullbacks).
- Divergences between price and momentum (RSI/MACD) indicate an impending trend exhaustion.
- Chart patterns and candlesticks are only valid when they occur at major support/resistance levels.

Provide your output strictly adhering to the requested Pydantic JSON schema.
"""

    def analyze(self, snapshot: dict) -> AnalystOutput:
        """Runs the analyst agent over the market snapshot."""
        prompt = f"""
Analyze the following Market Snapshot for {snapshot['symbol']}:
{json.dumps(snapshot, indent=2, default=str)}

Evaluate all parameters, check for price-volume confirmation or divergence, and output the analysis.
"""
        raw_output = self.client.call_gemini(
            model_name=self.model,
            system_instruction=self.system_instruction,
            prompt=prompt,
            response_schema=AnalystOutput
        )
        
        # Parse the structured response
        data = json.loads(raw_output)
        return AnalystOutput(**data)
