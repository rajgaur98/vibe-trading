import hashlib

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
