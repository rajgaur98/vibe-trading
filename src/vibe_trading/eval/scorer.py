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

_DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"


def build_judge() -> Callable[[str, Rubric], JudgeOutput]:
    """Returns a closure that, given (actual_text, rubric), calls an LLM judge and parses its output.

    Judge model is resolved per call from the EVAL_JUDGE_MODEL env var, defaulting to gemini-3.1-flash-lite.
    """
    client = LLMClient()

    def judge(actual_text: str, rubric: Rubric) -> JudgeOutput:
        model_name = os.getenv("EVAL_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
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
