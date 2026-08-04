#!/usr/bin/env python3
"""Regression checks for the CODEX2 efficiency and coverage contracts."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERSONA = ROOT.parent / "OmniVoice-clean-0.2.1" / "persona_project"
SCRIPTS = PERSONA / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import produce_anime_scene as producer


def main() -> None:
    config = json.loads((PERSONA / "production_config.json").read_text(encoding="utf-8"))
    assert config["scheduler"]["persistent_models"]
    assert config["scheduler"]["batch_size"] == 2
    assert config["vn"]["retry_takes"] == 1
    assert producer.words("the 13th Arcana") == ["the", "13th", "arcana"]
    assert producer.words(producer.alignment_text("the 13th Arcana")) == [
        "the", "thirteenth", "arcana"
    ]
    _, confirmed, _ = producer.source_language_confirmation(
        "You sound like a little kid.",
        "Du klingst wie ein kleines Kind.",
        "You sound like a little kid.",
    )
    assert confirmed
    _, common_confirmed, _ = producer.source_language_confirmation(
        "The seal is not meant to hold back Nyx.",
        "Dieses Siegel soll Nyx nicht zurückhalten.",
        "Dieses Siegel soll Nyx nicht zurückhalten.",
    )
    assert not common_confirmed
    assert producer.classify_failure({"error": "targets length is too long for CTC"}) == "DETERMINISTIC_TEXT_OR_WINDOW"
    tree = ast.parse((SCRIPTS / "produce_anime_scene.py").read_text(encoding="utf-8"))
    source = (SCRIPTS / "produce_anime_scene.py").read_text(encoding="utf-8")
    scheduler = (ROOT / "run_global_fmv_rounds.py").read_text(encoding="utf-8")
    assert "batch_unit_ids" in source
    assert "review_masked_ids" in source
    assert "GenerationRuntime" in scheduler
    assert "generation_work_exists" in scheduler
    assert "retryable_ids" in scheduler
    assert "only_rounds={round_index}" in scheduler
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "cheap_candidate_gate"
        for node in ast.walk(tree)
    )
    print("efficiency_policy_regression: PASS")


if __name__ == "__main__":
    main()
