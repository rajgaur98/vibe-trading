"""Online evaluation — score PRODUCTION decisions once their outcome is knowable.

Three layers (spec workstream C), all strictly best-effort:
  C1  deterministic outcome scoring  (this module's OutcomeScorer)
  C2  sampled generic-rubric LLM judge (capped per day)
  C3  weekly drift digest to Discord

A 'correct' flat is scored as 'no strong move happened' — a proxy for 'no edge
existed', not ground truth (a flat that dodged a crash scores poorly here). The
caveat is deliberate and documented; deterministic beats clever.
"""
import logging

logger = logging.getLogger(__name__)

# Directional decisions: +DIRECTIONAL_BAND_PCT% move in the decision's favor -> 1.0,
# same move against -> 0.0, linear in between, 0.5 for no move.
DIRECTIONAL_BAND_PCT = 2.0
# Flat decisions: |move| <= FLAT_OK_PCT -> 1.0, >= FLAT_ZERO_PCT -> 0.0, linear between.
FLAT_OK_PCT = 1.0
FLAT_ZERO_PCT = 5.0


def directional_score(signed_pct: float) -> float:
    """Map a signed %-move (positive = the decision's direction was profitable)
    onto [0, 1]."""
    return max(0.0, min(1.0, 0.5 + signed_pct / (2.0 * DIRECTIONAL_BAND_PCT)))


def flat_score(move_pct: float) -> float:
    """Score a flat decision by how quiet the market actually was."""
    m = abs(move_pct)
    if m <= FLAT_OK_PCT:
        return 1.0
    if m >= FLAT_ZERO_PCT:
        return 0.0
    return 1.0 - (m - FLAT_OK_PCT) / (FLAT_ZERO_PCT - FLAT_OK_PCT)
