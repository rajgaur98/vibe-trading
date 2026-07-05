"""The single home for LLM prompt text, with hand-bumped versions and computed
content hashes so every call/decision/eval can be traced to the exact prompt that
produced it — and so a prompt cannot be edited without a deliberate version bump
(enforced by tests/test_prompts.py against tests/fixtures/prompt_pins.json).

Workflow for changing a prompt:
  1. Edit the text AND bump the spec's `version` (v1 -> v2).
  2. Regenerate pins:  python -m vibe_trading.agents.prompts --write-pins tests/fixtures/prompt_pins.json
  3. Run the eval regression gate; re-seed the baseline only via --update-baseline.
"""
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    name: str          # registry key, e.g. "analyst_system"
    version: str       # hand-bumped on any semantic change, e.g. "v1"
    text: str

    @property
    def sha(self) -> str:
        """First 12 hex chars of sha256(text) — catches unbumped edits."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    @property
    def stamp(self) -> str:
        """The traceability string recorded everywhere: 'name:vN@sha12'."""
        return f"{self.name}:{self.version}@{self.sha}"


def pins_payload(specs: list) -> dict:
    """The committed pins fixture content for the given specs."""
    return {s.name: {"version": s.version, "sha": s.sha} for s in specs}


def check_pins(specs: list, pins: dict) -> list:
    """Compare the live registry against the committed pins. Returns human-readable
    violations; empty list means clean. Distinguishes 'you edited text without
    bumping' from 'you bumped — now regenerate pins deliberately'."""
    violations = []
    by_name = {s.name: s for s in specs}
    for s in specs:
        pinned = pins.get(s.name)
        if pinned is None:
            violations.append(
                f"{s.name}: not pinned — regenerate pins (--write-pins).")
        elif s.version != pinned["version"]:
            violations.append(
                f"{s.name}: version {pinned['version']} -> {s.version} — "
                f"regenerate pins to acknowledge the bump (--write-pins).")
        elif s.sha != pinned["sha"]:
            violations.append(
                f"{s.name}: text changed but version is still {s.version} — "
                f"bump the version, then regenerate pins.")
    for name in pins:
        if name not in by_name:
            violations.append(
                f"{name}: pinned but no longer registered — regenerate pins.")
    return violations
