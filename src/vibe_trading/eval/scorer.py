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
