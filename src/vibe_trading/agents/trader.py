from pydantic import BaseModel, Field
from typing import Literal
import json
import os
from uuid import uuid4
from datetime import datetime, timezone
from decimal import Decimal
from langfuse import observe, propagate_attributes
from vibe_trading.agents.client import LLMClient, validate_structured
from vibe_trading.agents.analyst import AnalystOutput


class HeadTraderOutput(BaseModel):
    action: Literal["long", "short", "flat", "close"] = Field(
        description="The action to take: 'long' to enter buy, 'short' to enter sell, 'close' to exit active position, or 'flat' to do nothing."
    )
    stop_loss_strategy: Literal["1.5_atr", "2.0_atr", "swing_low", "tight_atr"] = Field(
        description="The qualitative risk model strategy to determine the stop loss price boundary."
    )
    take_profit_strategy: Literal["3.0_atr", "4.0_atr", "next_resistance", "risk_reward_multiplier"] = Field(
        description="The qualitative profit capture strategy."
    )
    risk_reward_ratio: float = Field(
        description="The target risk-to-reward ratio (e.g., 2.0 means profit target is 2x larger than stop-loss)."
    )
    hold_period_bias: Literal["short", "medium", "long"] = Field(
        description="Expected holding period: short (1-2 days), medium (3-7 days), long (weeks)."
    )
    reasoning_summary: str = Field(
        description="A clear, concise summary of the rationale behind this final decision."
    )

class HeadTrader:
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()
        provider = self.client.provider
        self.model = os.getenv(f"{provider.upper()}_TRADER_MODEL") or self.client.model

        
        self.system_instruction = """
You are the Head Trader of a systematic crypto SWING-trading hedge fund.
Your job is to synthesize technical analysis, volume analysis, and historical performance metrics to make a final, highly disciplined trading decision.

You will receive:
1. An Analyst report containing bias, volume confirmation, and structural S/R zones (nearest_support, nearest_resistance).
2. The current market price.
3. The historical accuracy scorecard for the analyst.
4. The current portfolio positions.

Your core directives:
- Keep risk parameters strict. Do not chase trades if the analyst thesis is weak or has divergence.
- Resolve conflicts: if price bias is bullish but volume is weak/divergent, lean toward 'flat'.
- Do NOT compute raw stop/take-profit prices; SELECT the qualitative strategy using the rules below.

=== HOUSE METHODOLOGY (apply exactly) ===
Compute proximity from the current price and the analyst's S/R levels. A level is
"near" when it is within 2% of the current price.

STOP-LOSS STRATEGY (stop_loss_strategy):
- LONG entries:
  - If nearest_support is near (within 2% BELOW price) -> "swing_low" (anchor the stop just under structure).
  - Otherwise -> "1.5_atr".
- SHORT entries:
  - If nearest_resistance is near (within 2% ABOVE price) -> "tight_atr" (tight invalidation just above structure).
  - Otherwise -> "1.5_atr".

TAKE-PROFIT STRATEGY (take_profit_strategy):
- LONG entries -> "next_resistance" (target the structural level above).
- SHORT entries -> "3.0_atr" (measured move; there is no structural long target on a short).

RISK/REWARD (risk_reward_ratio):
- Target 2.0 (a 2:1 reward-to-risk). Use ~2.0 unless structure forces otherwise; never below 1.5.

HOLD PERIOD (hold_period_bias):
- This is a swing fund: default "medium" (3-7 days). Use "short" only for explicit
  counter-trend reversal scalps; "long" only for high-confluence trend continuation.

When action is "flat", the stop/take-profit/hold fields are not acted upon — still emit
schema-valid placeholder values, but spend your reasoning on WHY no edge exists.

Provide your output strictly matching the Pydantic JSON schema.
"""

    @observe()
    def decide(
        self,
        symbol: str,
        analyst_output: AnalystOutput,
        scorecard: dict,
        open_positions: list,
        current_price: float = 0.0,
    ) -> dict:
        """Invokes the Head Trader agent to make a trade decision.

        `current_price` is the live mark used to judge proximity to the analyst's S/R
        levels (drives the methodology's stop-loss selection). Defaults to 0.0 for
        backward compatibility; production/eval call sites pass the real price.
        """
        with propagate_attributes(
            trace_name=f"HeadTrader-decide-{symbol}",
            tags=[symbol],
            metadata={"symbol": symbol}
        ):
            prompt = f"""Make a trading decision for {symbol}.

--- Current Market Price ---
{current_price}

--- Analyst Output ---
{json.dumps(analyst_output.model_dump(), indent=2, default=str)}

--- Historical Analyst Accuracy Scorecard ---
{json.dumps(scorecard, indent=2, default=str)}

--- Current Open Positions ---
{json.dumps(open_positions, indent=2, default=str)}

--- Rules ---
- Apply the House Methodology for stop-loss, take-profit, risk/reward, and hold-period selection.
- Judge S/R proximity against the Current Market Price above (a level is "near" within 2%).
- Chart patterns and candlesticks are only valid when they occur at major support/resistance levels.
- Always output a valid schema.
"""
            def _call_single(extra: str = "") -> str:
                return self.client.call_llm(
                    model_name=self.model,
                    system_instruction=self.system_instruction,
                    prompt=prompt + extra,
                    response_schema=HeadTraderOutput,
                )

            raw_output = _call_single()

            # Validate into HeadTraderOutput with ONE corrective retry; records the
            # schema-compliance outcome onto the cost event. Raises SchemaValidationError
            # (never a bare KeyError) if both attempts fail. Read fields off the
            # validated model rather than building a dict by key.
            decision = validate_structured(
                self.client, HeadTraderOutput, raw_output, _call_single
            )

            # Hydrate the final proposal dictionary with system fields (UUID, timestamp).
            proposal = {
                "decision_id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                "action": decision.action,
                "stop_loss_strategy": decision.stop_loss_strategy,
                "take_profit_strategy": decision.take_profit_strategy,
                "risk_reward_ratio": Decimal(str(decision.risk_reward_ratio)),
                "hold_period_bias": decision.hold_period_bias,
                "reasoning_summary": decision.reasoning_summary,
            }

            return proposal
