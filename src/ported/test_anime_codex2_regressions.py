#!/usr/bin/env python3
"""Focused Codex2 regressions for Empalme B, tail QA and stale artifacts."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from anime_contract import contextual_final_word_gate, line_contract_hash


ROOT = Path(__file__).resolve().parent
MAP = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v3_codex2" / "100_030_M_map.json"
CONFIG = ROOT.parent / "OmniVoice-clean-0.2.1" / "persona_project" / "production_config.json"
PRODUCER = CONFIG.parent / "scripts" / "produce_anime_scene.py"


def _inputs() -> tuple[dict, dict, dict]:
    scene = json.loads(MAP.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    line = next(row for row in scene["lines"] if row["id"] == "100_030_M_L020")
    return scene, config, line


def test_l020_does_not_force_stale_splice_boundaries() -> None:
    _scene, _config, line = _inputs()
    assert line["preserve_leading_effort"] is True
    assert "effort_end_seconds" not in line
    assert "source_resume_seconds" not in line
    assert line["splice_trailing_guard_seconds"] >= 0.035


def test_contextual_tail_gate_is_not_asr_bypass() -> None:
    complete = {
        "quiet_tail_found": True,
        "hit_max_frames": False,
        "tail_release_ok": True,
    }
    clipped = {
        "quiet_tail_found": False,
        "hit_max_frames": True,
        "tail_release_ok": False,
    }
    assert contextual_final_word_gate(complete, 1.0, 24_000, 24_000)
    assert not contextual_final_word_gate(clipped, 1.0, 24_000, 24_000)
    assert not contextual_final_word_gate(complete, None, 24_000, 24_000)


def test_line_contract_changes_when_l020_map_changes() -> None:
    scene, config, line = _inputs()
    current = line_contract_hash(scene, line, MAP, config, PRODUCER, CONFIG.parent)
    altered = dict(line)
    altered["splice_crossfade_seconds"] = 0.036
    changed = line_contract_hash(
        scene, altered, MAP, config, PRODUCER, CONFIG.parent,
    )
    assert current != changed


def test_active_producer_contains_boundary_and_contextual_gates() -> None:
    tree = ast.parse(PRODUCER.read_text(encoding="utf-8"))
    source = PRODUCER.read_text(encoding="utf-8")
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "resolve_leading_boundaries" in functions
    assert "cut_short" in functions
    assert "contextual_final_word_gate" in source
    assert "accepted_line_contracts" in source
    assert "row.get(\"contract_hash\") in accepted_line_contracts" in source
    assert "all_required_dubs_mounted" in source
    assert "MISSING_CURRENT_CONTRACT_CANDIDATE" in source
    assert "reference: bool = False" in source
    assert "source_window(psm, line, sr, reference=True)" in source


def test_reference_edges_are_separate_from_delivery_edges() -> None:
    scene = json.loads(
        (
            ROOT / "P3R_ANIME_VISUAL_DUB_20260801"
            / "maps_delivery_aligned_v3_codex2"
            / "210_250_M_map.json"
        ).read_text(encoding="utf-8")
    )
    line = next(row for row in scene["lines"] if row["id"] == "210_250_M_L004")
    assert line["start"] == 18.28
    assert line["reference_start"] == 18.54
    assert line["reference_end"] > line["end"]
    assert line["reference_end"] >= line["speech_end"]


def test_codex2_experiment_and_duration_contract_are_explicit() -> None:
    _scene, config, _line = _inputs()
    assert config["anime"]["append_ellipsis_experiment"] is True
    assert config["anime"]["ellipsis_suffix"] == "..."
    assert config["contracts"]["max_tempo_deviation"] == 0.15
    assert config["contracts"]["duration_tolerance_seconds"] == 0.35
    source = PRODUCER.read_text(encoding="utf-8")
    assert "append_generation_suffix" in source
    assert "target_content_gate" in source
    assert "current_qa_hash" in source
    assert "contextual_final_word_ok" in source


def test_duration_first_gate_policy_keeps_quality_metrics_diagnostic() -> None:
    _scene, config, _line = _inputs()
    policy = config["qa"]["hard_gate_policy"]
    assert set(policy["hard"]) == {
        "not_empty", "source_language", "tail", "final_word",
        "splice_seam", "splice_boundary", "splice_speech_timing",
        "lufs", "clipping", "frames",
    }
    assert set(policy["diagnostic_only"]) == {
        "text", "onset", "span", "rate", "pause", "pitch_identity",
    }
    assert policy["text_ranking_enabled"] is False
    source = PRODUCER.read_text(encoding="utf-8")
    assert '"diagnostic_gates": diagnostic_gates' in source
    assert "100.0 * wer" not in source
    assert 'hard = {\n                "not_empty"' in source


def test_reconciled_inventory_recovers_known_english_leak_lines() -> None:
    report = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "ANIME_COVERAGE_RECONCILIATION_CODEX2.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert (
        payload["recovered_text_lines"]
        + payload.get("deduplicated_text_lines", 0)
        >= 72
    )
    assert payload.get("deduplicated_text_lines", 0) >= 7
    scene_100 = next(row for row in payload["scenes"] if row["scene"] == "100_030_M")
    assert any(
        item["id"] == "100_030_M_L012"
        and item["covered_by"] == "100_030_M_L014"
        for item in scene_100.get("deduplicated_lines", [])
    )
    by_scene = {row["scene"]: set(row["recovered_ids"]) for row in payload["scenes"]}
    assert "200_130_M_L020" in by_scene["200_130_M"]
    assert "220_030_M_L003" in by_scene["220_030_M"]
    assert "300_010_M_L002A" in by_scene["300_010_M"]


def test_subtitle_only_policy_demotes_audio_only_rows_without_blocking() -> None:
    root = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v3_codex2"
    blocked_ids = {
        "210_210_M_L008", "210_210_M_L011", "210_210_M_L012",
        "210_210_M_L013", "210_210_M_L014", "210_210_M_L016",
        "210_210_M_L017", "210_210_M_L020", "210_210_M_L021",
        "210_210_M_L028", "210_210_M_L029", "210_210_M_L030",
    }
    scene = json.loads((root / "210_210_M_map.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in scene["lines"]}
    for line_id in blocked_ids:
        row = rows[line_id]
        assert row["subtitle_authorized"] is False
        assert row["force_keep_original"] is True
        assert row["mapping_validation"] == "NO_VISIBLE_SUBTITLE_CARD"
        assert not row.get("generation_blocked")
        assert row["force_clone"] is False

    for path in root.glob("*_map.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not any(row.get("generation_blocked") for row in payload.get("lines", []))
    report = json.loads(
        (root.parent / "ANIME_COVERAGE_RECONCILIATION_CODEX2.json").read_text(encoding="utf-8")
    )
    assert report["subtitle_policy"]["name"] == "SUBTITLE_AUTHORIZED_ONLY"
    assert report["kept_unsubtitled_lines"] >= len(blocked_ids)


def main() -> None:
    for name, value in globals().copy().items():
        if name.startswith("test_") and callable(value):
            value()
    print("anime_codex2_regressions: PASS")


if __name__ == "__main__":
    main()
