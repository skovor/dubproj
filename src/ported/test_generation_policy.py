#!/usr/bin/env python3
"""Fast regression checks for P3R generation/retry policy (no GPU/model)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PERSONA = ROOT.parent / "OmniVoice-clean-0.2.1" / "persona_project"


def load_rounds():
    path = ROOT / "generate_p3r_cinematic_assets_v021_rounds.py"
    spec = importlib.util.spec_from_file_location("p3r_rounds_policy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempts(flags: list[bool]) -> list[dict]:
    return [{"qa": {"pass": value}} for value in flags]


def check_schedule() -> None:
    module = load_rounds()
    cases = [
        ([], "COLLECTING_INITIAL_1"),
        ([True], "PASS"),
        ([False], "COLLECTING_FIRST_4"),
        ([False, True], "COLLECTING_FIRST_4"),
        ([False, False, True, False, False], "PASS"),
        ([False] * 5, "NEEDS_SECOND_4"),
        ([False] * 5 + [True], "NEEDS_SECOND_4"),
        ([False] * 8 + [True], "PASS"),
        ([False] * 9, "BLOCKED_AFTER_MAX9"),
    ]
    for flags, expected in cases:
        actual = module.scheduling_state(attempts(flags))
        assert actual == expected, (flags, actual, expected)
    cohort_cases = [
        ([], 1),
        ([False], 5),
        ([False, True], 5),
        ([False] * 5, 9),
        ([False] * 5 + [True], 9),
        ([False] * 8, 9),
    ]
    for flags, expected in cohort_cases:
        actual = module.committed_cohort_target(attempts(flags))
        assert actual == expected, (flags, actual, expected)


def check_profiles() -> None:
    config = json.loads(
        (PERSONA / "production_config.json").read_text(encoding="utf-8")
    )
    assert config["vn"]["initial_takes"] == 1
    # VN retries are sequential: one additional candidate per failed line,
    # with max_rounds providing the hard attempt ceiling.
    assert config["vn"]["retry_takes"] == 1
    assert config["vn"]["max_rounds"] == 4
    assert config["anime"]["initial_takes"] == 4
    assert config["anime"]["retry_takes"] == 4


def check_release_contract() -> None:
    module = load_rounds()
    sr = 48000
    clean = np.zeros(sr, dtype=np.float32)
    clean[: sr - round(0.025 * sr)] = 0.2
    # Less than 35 ms of physical padding still passes because the last 20 ms
    # are quiet and the final sample is below -40 dB.
    assert module.base.release_metrics(clean, sr)["release_ok"]
    cut = np.full(sr, 0.2, dtype=np.float32)
    assert not module.base.release_metrics(cut, sr)["release_ok"]


def check_in_engine_disables_asr() -> None:
    path = PERSONA / "scripts" / "produce_in_engine_bank.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate"
    ]
    assert calls, "no in-engine evaluate calls found"
    for call in calls:
        keywords = {item.arg: item.value for item in call.keywords}
        value = keywords.get("use_asr")
        assert isinstance(value, ast.Constant) and value.value is False, ast.unparse(call)


def main() -> None:
    check_schedule()
    check_profiles()
    check_release_contract()
    check_in_engine_disables_asr()
    print("generation_policy_regression: PASS")


if __name__ == "__main__":
    main()
