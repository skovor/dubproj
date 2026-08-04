#!/usr/bin/env python3
"""Deterministic input-contract hashing for the active anime producer.

The hash intentionally covers the effective map, production configuration,
producer source, and referenced audio files.  It lets resumable generation and
candidate selection reject artifacts produced under an older contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _file_entry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "missing": True}
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _contract_hash(
    payload: dict[str, Any],
    map_path: Path,
    config: dict[str, Any],
    producer_path: Path,
    project_path: Path,
    files: list[dict[str, Any]],
) -> str:
    payload = {
        "schema": "codex2-anime-contract-v1",
        "payload": payload,
        "map_path": str(map_path.resolve()),
        "config": config,
        "producer": _file_entry(producer_path),
        "inputs": sorted(files, key=lambda item: item["path"]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_files(scene: dict[str, Any], map_path: Path, project_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    source_stem = scene.get("source_stem")
    if source_stem:
        source_path = _resolve(str(source_stem), map_path.parent)
        files.append(_file_entry(source_path))
        full_name = source_path.name.replace("_dialog_ch5", "_6ch")
        files.append(_file_entry(source_path.with_name(full_name)))
    for line in scene.get("lines", []):
        for segment in line.get("reference_segments", []) or []:
            if segment.get("path"):
                files.append(_file_entry(_resolve(str(segment["path"]), project_path)))
        for key in ("reference_audio", "reference_path"):
            if line.get(key):
                files.append(_file_entry(_resolve(str(line[key]), project_path)))
    return files


def scene_contract_hash(
    scene: dict[str, Any],
    map_path: Path,
    config: dict[str, Any],
    producer_path: Path,
    project_path: Path,
) -> str:
    """Return the hash for the exact scene inputs used by generation/QA."""
    effective_scene = {
        key: value for key, value in scene.items()
        if not key.startswith("_") and key not in {"contract_hash"}
    }
    return _contract_hash(
        {"scene": effective_scene},
        map_path,
        config,
        producer_path,
        project_path,
        _scene_files(scene, map_path, project_path),
    )


def line_contract_hash(
    scene: dict[str, Any],
    line: dict[str, Any],
    map_path: Path,
    config: dict[str, Any],
    producer_path: Path,
    project_path: Path,
) -> str:
    """Hash one line while retaining scene/source/config dependencies."""
    effective_scene = {
        key: value for key, value in scene.items()
        if not key.startswith("_") and key not in {"contract_hash", "lines"}
    }
    effective_line = {
        key: value for key, value in line.items()
        if not key.startswith("_") and key not in {"contract_hash"}
    }
    return _contract_hash(
        {"scene": effective_scene, "line": effective_line},
        map_path,
        config,
        producer_path,
        project_path,
        _scene_files({**scene, "lines": [line]}, map_path, project_path),
    )


def _line_payload(scene: dict[str, Any], line: dict[str, Any]) -> dict[str, Any]:
    """Stable semantic payload shared by the split stage contracts."""
    return {
        "scene": scene.get("scene"),
        "kind": scene.get("kind"),
        "line": {
            key: value for key, value in line.items()
            if not key.startswith("_")
            and key not in {"contract_hash", "expected_visual_ids"}
        },
    }


def generation_contract_hash(
    scene: dict[str, Any], line: dict[str, Any], map_path: Path,
    config: dict[str, Any], project_path: Path,
) -> str:
    """Hash only inputs that can change OmniVoice waveform generation."""
    payload = _line_payload(scene, line)
    profile = config.get("anime", config)
    generation_config = {
        key: profile.get(key) for key in (
            "model", "initial_takes", "retry_takes", "num_step",
            "guidance_scale", "position_temperature", "class_temperature",
            "t_shift", "postprocess_output", "pad_duration", "fade_duration",
            "append_ellipsis_experiment", "ellipsis_suffix",
        )
    }
    return _contract_hash(
        {"stage": "generation", **payload}, map_path, generation_config,
        map_path, project_path, _scene_files({**scene, "lines": [line]}, map_path, project_path),
    )


def processing_contract_hash(
    scene: dict[str, Any], line: dict[str, Any], map_path: Path,
    config: dict[str, Any], producer_path: Path, project_path: Path,
) -> str:
    """Hash the deterministic extraction/timing/splice processing contract."""
    processing_keys = {
        key: line.get(key) for key in (
            "start", "end", "synthesis_start", "preserve_leading_effort",
            "delivery_word_start", "delivery_word_count", "minimum_pause_after_word",
            "splice_crossfade_seconds", "splice_crossfade_curve",
            "splice_min_pause_seconds", "splice_boundary_tolerance_seconds",
            "effort_end_seconds", "source_resume_seconds",
            "allow_non_neutral_leading_interjection",
            "preserved_prefix_text", "preserved_source_intervals",
        ) if key in line
    }
    return _contract_hash(
        {"stage": "processing", "scene": scene.get("scene"), "line": processing_keys},
        map_path, config.get("contracts", {}), producer_path, project_path,
        _scene_files({**scene, "lines": [line]}, map_path, project_path),
    )


def qa_contract_hash(
    qa: dict[str, Any], producer_path: Path,
) -> str:
    """Hash thresholds/model policy without invalidating generation artifacts."""
    return _contract_hash(
        {"stage": "qa", "qa": qa}, producer_path, qa,
        producer_path, producer_path.parent, [],
    )


def mount_contract_hash(
    scene: dict[str, Any], config: dict[str, Any], producer_path: Path,
) -> str:
    """Hash the release assembly policy and coverage requirements."""
    return _contract_hash(
        {
            "stage": "mount",
            "scene": scene.get("scene"),
            "expected_visual_ids": scene.get("expected_visual_ids", []),
            "contracts": config.get("contracts", {}),
        }, producer_path, config.get("contracts", {}),
        producer_path, producer_path.parent, [],
    )


def contextual_final_word_gate(
    extraction_meta: dict[str, Any] | None,
    alignment_end_seconds: float | None,
    body_frames: int,
    sample_rate: int,
    alignment_tolerance_seconds: float = 0.005,
) -> bool:
    """Hard gate for an extracted contextual TTS body.

    ASR token presence is not enough: extraction must have found a quiet release
    and the final aligned word must fit inside the isolated body.
    """
    if not extraction_meta or not extraction_meta.get("tail_release_ok"):
        return False
    if alignment_end_seconds is None:
        return False
    return float(alignment_end_seconds) <= (
        body_frames / sample_rate + alignment_tolerance_seconds
    )
