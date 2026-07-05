import pytest

from vibe_trading.eval.online import directional_score, flat_score


def test_directional_score_maps_band_to_unit_interval():
    assert directional_score(0.0) == pytest.approx(0.5)     # no move -> coin flip
    assert directional_score(2.0) == pytest.approx(1.0)     # +band in favor -> 1.0
    assert directional_score(-2.0) == pytest.approx(0.0)    # band against -> 0.0
    assert directional_score(1.0) == pytest.approx(0.75)    # linear in between
    assert directional_score(50.0) == 1.0                   # clamped
    assert directional_score(-50.0) == 0.0                  # clamped


def test_flat_score_rewards_small_moves():
    assert flat_score(0.0) == 1.0
    assert flat_score(1.0) == 1.0        # within FLAT_OK_PCT
    assert flat_score(-1.0) == 1.0       # sign-agnostic (uses |move|)
    assert flat_score(5.0) == 0.0        # at/after FLAT_ZERO_PCT
    assert flat_score(3.0) == pytest.approx(0.5)  # halfway on the linear ramp
