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
from vibe_trading.agents import prompts


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

        self.system_instruction = prompts.TRADER_SYSTEM.text

    @observe()
    def decide(
        self,
        symbol: str,
        analyst_output: AnalystOutput,
        scorecard: dict,
        open_positions: list,
        current_price: float = 0.0,
        precedents=None,
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
            precedent_block = ""
            if precedents:
                lines = "\n".join(
                    f"- {p.symbol} {p.action.upper()} ({p.when}, similarity {p.similarity:.2f}): {p.outcome_label}"
                    for p in precedents
                )
                precedent_block = (
                    "\n--- PRECEDENTS — similar past setups you took and how they resolved ---\n"
                    f"{lines}\n"
                    "Weigh these against the current setup; repeated losses on a similar setup are a reason for caution.\n"
                )

            prompt = f"""Make a trading decision for {symbol}.

--- Current Market Price ---
{current_price}

--- Analyst Output ---
{json.dumps(analyst_output.model_dump(), indent=2, default=str)}

--- Historical Analyst Accuracy Scorecard ---
{json.dumps(scorecard, indent=2, default=str)}

--- Current Open Positions ---
{json.dumps(open_positions, indent=2, default=str)}
{precedent_block}
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
