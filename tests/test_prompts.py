import hashlib
import json
from pathlib import Path

from vibe_trading.agents import prompts
from vibe_trading.agents.prompts import PromptSpec, check_pins, pins_payload


def spec(name="p1", version="v1", text="hello") -> PromptSpec:
    return PromptSpec(name=name, version=version, text=text)


def test_sha_is_first_12_hex_of_sha256():
    s = spec(text="hello")
    assert s.sha == hashlib.sha256(b"hello").hexdigest()[:12]
    assert len(s.sha) == 12


def test_stamp_format():
    s = spec(name="analyst_system", version="v3", text="hello")
    assert s.stamp == f"analyst_system:v3@{s.sha}"


def test_pins_payload_round_trip():
    s = spec()
    pins = pins_payload([s])
    assert pins == {"p1": {"version": "v1", "sha": s.sha}}
    assert check_pins([s], pins) == []


def test_text_edit_without_bump_is_a_violation():
    pinned = pins_payload([spec(text="original")])
    violations = check_pins([spec(text="EDITED")], pinned)
    assert len(violations) == 1
    assert "bump" in violations[0]  # tells the dev to bump the version


def test_version_bump_requires_pin_regen():
    pinned = pins_payload([spec(version="v1")])
    violations = check_pins([spec(version="v2")], pinned)
    assert len(violations) == 1
    assert "regenerate" in violations[0]  # deliberate: regen pins after a bump


def test_unpinned_and_orphaned_prompts_are_violations():
    pinned = pins_payload([spec(name="old_prompt")])
    violations = check_pins([spec(name="new_prompt")], pinned)
    assert len(violations) == 2  # new_prompt unpinned + old_prompt orphaned


PINS_PATH = Path("tests/fixtures/prompt_pins.json")


def test_registry_contains_the_four_system_prompts():
    assert set(prompts.REGISTRY) == {
        "analyst_system", "trader_system", "judge_system", "online_judge_system"}
    for s in prompts.REGISTRY.values():
        assert s.text.strip()  # non-empty, real content


def test_registered_texts_are_the_ones_agents_use():
    from vibe_trading.agents.analyst import TechnicalVolumeAnalyst
    from vibe_trading.agents.trader import HeadTrader
    from vibe_trading.eval import scorer
    import os
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    assert TechnicalVolumeAnalyst(db=None, fetcher=None).system_instruction \
        == prompts.ANALYST_SYSTEM.text
    assert HeadTrader().system_instruction == prompts.TRADER_SYSTEM.text
    assert scorer._JUDGE_SYSTEM == prompts.JUDGE_SYSTEM.text


def test_versions_map_and_bundle_version():
    vm = prompts.versions_map()
    assert vm["analyst_system"] == prompts.ANALYST_SYSTEM.stamp
    # bundle covers ONLY the decision-path prompts (analyst + trader), ';'-joined in
    # name order — judge prompts must not perturb the decision-level stamp
    assert prompts.bundle_version() == ";".join(
        [prompts.ANALYST_SYSTEM.stamp, prompts.TRADER_SYSTEM.stamp])


def test_pins_are_current():
    """THE enforcement test: editing any prompt text without bumping its version
    (or bumping without regenerating pins) fails here with instructions."""
    pins = json.loads(PINS_PATH.read_text())
    violations = prompts.check_pins(list(prompts.REGISTRY.values()), pins)
    assert violations == [], "\n".join(violations)


def test_trader_stamps_its_calls(monkeypatch):
    import os
    from unittest.mock import MagicMock
    from vibe_trading.agents.trader import HeadTrader
    from vibe_trading.agents.analyst import AnalystOutput

    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"action": "flat", "stop_loss_strategy": "1.5_atr", '
        '"take_profit_strategy": "3.0_atr", "risk_reward_ratio": 2.0, '
        '"hold_period_bias": "medium", "reasoning_summary": "r"}'
    )
    trader = HeadTrader(client=client)
    analyst_output = AnalystOutput(
        market_bias="neutral", volume_confirmation="weak", thesis="t",
        nearest_support=1.0, nearest_resistance=2.0, confluence_score=0.5)
    trader.decide("BTC/USDT", analyst_output, {}, [], current_price=1.5)
    assert client.call_llm.call_args.kwargs["prompt_version"] \
        == prompts.TRADER_SYSTEM.stamp


def test_analyst_snapshot_path_stamps_its_calls(monkeypatch):
    import os
    from unittest.mock import MagicMock
    from vibe_trading.agents.analyst import TechnicalVolumeAnalyst

    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    client.provider = "gemini"
    client.model = "m"
    client.call_llm.return_value = (
        '{"market_bias": "neutral", "volume_confirmation": "weak", "thesis": "t", '
        '"nearest_support": 1.0, "nearest_resistance": 2.0, "confluence_score": 0.5}'
    )
    analyst = TechnicalVolumeAnalyst(client=client, db=None, fetcher=None)
    analyst.analyze("BTC/USDT", snapshot={"close": 1.5})
    assert client.call_llm.call_args.kwargs["prompt_version"] \
        == prompts.ANALYST_SYSTEM.stamp
