import math
from typing import Any, Optional

from pydantic import BaseModel


# Tolerance constants. `ok` -> score 1.0; `zero` -> score 0.0; linear in between.
# For nearest_support / nearest_resistance, thresholds are PERCENTAGES of expected.
# For confluence_score / risk_reward_ratio, thresholds are ABSOLUTE distances.
# When expected == 0.0, percentage thresholds collapse to absolute distance.
NUMERIC_TOLERANCES: dict[str, dict[str, float]] = {
    "nearest_support":    {"ok": 0.02, "zero": 0.05},
    "nearest_resistance": {"ok": 0.02, "zero": 0.05},
    "confluence_score":   {"ok": 0.15, "zero": 0.30},
    "risk_reward_ratio":  {"ok": 0.5,  "zero": 1.5},
}

PERCENTAGE_FIELDS = {"nearest_support", "nearest_resistance"}


class FieldScore(BaseModel):
    field: str
    passed: bool
    score: float
    note: str = ""


def score_categorical(field: str, actual: Any, expected: Any) -> FieldScore:
    """Exact-match scoring for enum-typed fields."""
    if actual == expected:
        return FieldScore(field=field, passed=True, score=1.0)
    return FieldScore(
        field=field,
        passed=False,
        score=0.0,
        note=f"expected '{expected}', got '{actual}'",
    )


def score_numeric_tolerance(field: str, actual: Optional[float], expected: float) -> FieldScore:
    """Tolerance-based numeric scoring with linear degradation.

    See NUMERIC_TOLERANCES + PERCENTAGE_FIELDS for the per-field thresholds.
    """
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return FieldScore(field=field, passed=False, score=0.0, note="missing value")

    thresholds = NUMERIC_TOLERANCES[field]
    ok = thresholds["ok"]
    zero = thresholds["zero"]

    if field in PERCENTAGE_FIELDS and expected != 0.0:
        delta = abs(actual - expected) / abs(expected)
    else:
        delta = abs(actual - expected)

    if delta <= ok:
        return FieldScore(field=field, passed=True, score=1.0)
    if delta >= zero:
        return FieldScore(field=field, passed=False, score=0.0,
                          note=f"delta {delta:.4f} beyond zero threshold {zero}")

    score = 1.0 - (delta - ok) / (zero - ok)
    return FieldScore(field=field, passed=False, score=score,
                      note=f"delta {delta:.4f} between thresholds [{ok}, {zero}]")


import os
from typing import Callable

from vibe_trading.agents.client import LLMClient
from vibe_trading.eval.runner import Rubric


class CriterionEvaluation(BaseModel):
    criterion: str
    passed: bool
    justification: str


class JudgeOutput(BaseModel):
    must_mention_results: list[CriterionEvaluation]
    must_not_mention_results: list[CriterionEvaluation]


_JUDGE_SYSTEM = """
You are a meticulous, code-review-style evaluator. You will receive a piece of agent-generated
text and a rubric of must-mention and must-not-mention criteria.

For each must-mention criterion: mark passed=true only if the criterion is clearly present in the
text (not just hinted at). Otherwise passed=false.

For each must-not-mention criterion: mark passed=true if the criterion is clearly absent from the
text. If the text clearly violates it, passed=false.

Output strictly matches the JudgeOutput JSON schema. Provide a one-sentence justification per
criterion.
""".strip()

def build_judge() -> Callable[[str, Rubric], JudgeOutput]:
    """Returns a closure that, given (actual_text, rubric), calls an LLM judge and parses its output.

    Judge model is resolved per call from the `EVAL_JUDGE_MODEL` env var. When unset,
    falls back to the LLMClient's configured model — guarantees the judge always
    targets the active provider (so switching `LLM_PROVIDER` from gemini to groq
    doesn't leave the judge pointing at a model the new provider doesn't host).
    """
    client = LLMClient()

    def judge(actual_text: str, rubric: Rubric) -> JudgeOutput:
        model_name = os.getenv("EVAL_JUDGE_MODEL") or client.model
        prompt = "TEXT:\n" + actual_text + "\n\nRUBRIC:\nMUST MENTION:\n"
        prompt += "\n".join(f"- {c}" for c in rubric.must_mention) if rubric.must_mention else "(none)"
        prompt += "\n\nMUST NOT MENTION:\n"
        prompt += "\n".join(f"- {c}" for c in rubric.must_not_mention) if rubric.must_not_mention else "(none)"

        raw = client.call_llm(
            model_name=model_name,
            system_instruction=_JUDGE_SYSTEM,
            prompt=prompt,
            response_schema=JudgeOutput,
        )
        return JudgeOutput.model_validate_json(raw)

    return judge


def score_rubric(
    field: str,
    actual_text: str,
    rubric: Rubric,
    judge: Callable[[str, Rubric], JudgeOutput],
) -> FieldScore:
    """Score free-text against a rubric using the injected LLM judge.

    Score = passed_criteria / total_criteria across both must_mention and must_not_mention lists.
    Empty rubric -> score 1.0 (vacuous truth).
    Judge exception -> score 0.5 with judge_error note (suite continues).
    """
    total = len(rubric.must_mention) + len(rubric.must_not_mention)
    if total == 0:
        return FieldScore(field=field, passed=True, score=1.0, note="empty rubric")

    try:
        result = judge(actual_text, rubric)
    except Exception as e:
        return FieldScore(field=field, passed=False, score=0.5, note=f"judge_error: {e}")

    passed_count = (
        sum(1 for r in result.must_mention_results if r.passed)
        + sum(1 for r in result.must_not_mention_results if r.passed)
    )
    score = passed_count / total
    return FieldScore(field=field, passed=(score == 1.0), score=score,
                      note=f"{passed_count}/{total} criteria passed")


from vibe_trading.eval.runner import CaseResult, EvalCase


class CaseScore(BaseModel):
    case_id: str
    schema_ok: bool
    field_scores: list[FieldScore]
    analyst_score: float
    trader_score: float
    total_score: float
    error: Optional[str] = None


ANALYST_FIELDS = {"market_bias", "volume_confirmation", "nearest_support",
                  "nearest_resistance", "confluence_score", "thesis"}
TRADER_FIELDS = {"action", "stop_loss_strategy", "take_profit_strategy",
                 "risk_reward_ratio", "hold_period_bias", "reasoning_summary"}


def score_case(
    result: CaseResult,
    case: EvalCase,
    judge: Callable[[str, Rubric], JudgeOutput],
) -> CaseScore:
    """Aggregate per-field scores into a case-level CaseScore.

    Short-circuits to total_score=0.0 if snapshot or schema validation failed for either agent.
    """
    if not result.snapshot_ok or not result.analyst_schema_ok or not result.trader_schema_ok:
        return CaseScore(
            case_id=result.case_id, schema_ok=False, field_scores=[],
            analyst_score=0.0, trader_score=0.0, total_score=0.0,
            error=result.error,
        )

    field_scores: list[FieldScore] = [
        # Analyst — always scored (independent of the trade decision)
        score_categorical("market_bias", result.analyst_output["market_bias"], case.analyst_label.market_bias),
        score_categorical("volume_confirmation", result.analyst_output["volume_confirmation"], case.analyst_label.volume_confirmation),
        score_numeric_tolerance("nearest_support", result.analyst_output["nearest_support"], case.analyst_label.nearest_support),
        score_numeric_tolerance("nearest_resistance", result.analyst_output["nearest_resistance"], case.analyst_label.nearest_resistance),
        score_numeric_tolerance("confluence_score", result.analyst_output["confluence_score"], case.analyst_label.confluence_score),
        score_rubric("thesis", result.analyst_output.get("thesis", ""), case.analyst_label.thesis_rubric, judge),
        # Trader — the directional call and its rationale are always meaningful
        score_categorical("action", result.trader_output["action"], case.trader_label.action),
        score_rubric("reasoning_summary", result.trader_output.get("reasoning_summary", ""), case.trader_label.reasoning_rubric, judge),
    ]

    # Entry-strategy fields only carry meaning when the label calls for entering a
    # position. On a 'flat' label there is no position to size/stop/exit, so these
    # fields have no ground truth — scoring them against placeholder labels would
    # just inject noise. Score them only for directional (long/short) labels.
    if case.trader_label.action != "flat":
        field_scores.extend([
            score_categorical("stop_loss_strategy", result.trader_output["stop_loss_strategy"], case.trader_label.stop_loss_strategy),
            score_categorical("take_profit_strategy", result.trader_output["take_profit_strategy"], case.trader_label.take_profit_strategy),
            score_categorical("hold_period_bias", result.trader_output["hold_period_bias"], case.trader_label.hold_period_bias),
            score_numeric_tolerance("risk_reward_ratio", float(result.trader_output["risk_reward_ratio"]), case.trader_label.risk_reward_ratio),
        ])

    analyst_field_scores = [fs for fs in field_scores if fs.field in ANALYST_FIELDS]
    trader_field_scores = [fs for fs in field_scores if fs.field in TRADER_FIELDS]

    analyst_score = sum(fs.score for fs in analyst_field_scores) / len(analyst_field_scores)
    trader_score = sum(fs.score for fs in trader_field_scores) / len(trader_field_scores)
    total = sum(fs.score for fs in field_scores) / len(field_scores)

    return CaseScore(
        case_id=result.case_id, schema_ok=True, field_scores=field_scores,
        analyst_score=analyst_score, trader_score=trader_score, total_score=total,
        error=None,
    )
