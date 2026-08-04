"""Strict semantic validation for V2 scene manifests.

JSON Schema remains useful for editor tooling, but these checks are deliberately
kept in Python as they validate relationships (ownership, windows and refs)
that a shallow shape schema cannot prove.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import ContractError

TOPOLOGIES = {"LINE_SEPARATED", "IN_ENGINE_TIMELINE", "EMBEDDED_FMV"}
_SCENE_FIELDS = {"id", "topology", "lines", "source_stem", "dialogue_channel", "movie_identity_verified", "duration_seconds", "duration_samples", "extensions"}
_LINE_FIELDS = {"id", "speaker", "source_text", "target_text", "start", "end", "topology", "source_audio", "reference_audio", "reference_segments", "subtitle_authorized", "movie_identity_verified", "card_identity_verified", "card_timebase_verified", "force_keep_original", "preserve_reason", "synthesis_text_override", "delivery_text", "speech_start", "speech_end", "preserved_source_intervals", "source_resume", "extensions"}
_REF_FIELDS = {"path", "start", "end", "text", "channel", "start_sample", "end_sample"}


@dataclass(frozen=True)
class ManifestSummary:
    scene_count: int
    line_count: int
    topologies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"scene_count": self.scene_count, "line_count": self.line_count, "topologies": list(self.topologies)}


def _unknown(value: Mapping[str, Any], known: set[str], label: str) -> None:
    unknown = set(value) - known
    if unknown:
        raise ContractError(f"unknown {label} fields (use extensions): {sorted(unknown)}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    return float(value)


def _validate_reference_segments(line: Mapping[str, Any]) -> None:
    segments = line.get("reference_segments", [])
    if not isinstance(segments, list):
        raise ContractError(f"{line.get('id')}: reference_segments must be a list")
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ContractError(f"{line.get('id')}: reference segment must be an object")
        _unknown(segment, _REF_FIELDS, "reference segment")
        if not str(segment.get("path", "")).strip():
            raise ContractError(f"{line.get('id')}: reference segment path is empty")
        start = _number(segment.get("start", 0), "reference start")
        end = segment.get("end")
        if start < 0 or (end is not None and _number(end, "reference end") <= start):
            raise ContractError(f"{line.get('id')}: invalid reference segment range")
        if segment.get("channel") is not None and int(segment["channel"]) < 0:
            raise ContractError(f"{line.get('id')}: invalid reference channel")


def validate_scene_value(scene: Mapping[str, Any]) -> None:
    if not isinstance(scene, Mapping):
        raise ContractError("scene must be an object")
    _unknown(scene, _SCENE_FIELDS, "scene")
    scene_id = str(scene.get("id", "")).strip()
    if not scene_id:
        raise ContractError("scene id is required")
    topology = str(scene.get("topology", "LINE_SEPARATED"))
    if topology not in TOPOLOGIES:
        raise ContractError(f"unsupported topology: {topology}")
    lines = scene.get("lines")
    if not isinstance(lines, list):
        raise ContractError(f"{scene_id}: lines must be a list")
    if not isinstance(scene.get("dialogue_channel", 0), int) or int(scene.get("dialogue_channel", 0)) < 0:
        raise ContractError(f"{scene_id}: dialogue_channel must be a non-negative integer")
    duration = scene.get("duration_seconds")
    duration_samples = scene.get("duration_samples")
    if duration is not None and _number(duration, "duration_seconds") <= 0:
        raise ContractError(f"{scene_id}: duration_seconds must be positive")
    if duration_samples is not None and int(duration_samples) <= 0:
        raise ContractError(f"{scene_id}: duration_samples must be positive")
    seen: set[str] = set()
    windows: list[tuple[float, float, str]] = []
    for line in lines:
        if not isinstance(line, Mapping):
            raise ContractError(f"{scene_id}: every line must be an object")
        _unknown(line, _LINE_FIELDS, "line")
        line_id = str(line.get("id", "")).strip()
        if not line_id:
            raise ContractError(f"{scene_id}: line id is required")
        if line_id in seen:
            raise ContractError(f"{scene_id}: duplicate line id {line_id}")
        seen.add(line_id)
        for key in ("speaker", "source_text", "target_text"):
            if key not in line:
                raise ContractError(f"{line_id}: missing {key}")
        start, end = _number(line.get("start"), "start"), _number(line.get("end"), "end")
        if start < 0 or end <= start:
            raise ContractError(f"{line_id}: line window must be positive")
        if duration is not None and end > float(duration) + 1e-6:
            raise ContractError(f"{line_id}: line exceeds scene duration")
        if duration_samples is not None and end > int(duration_samples):
            raise ContractError(f"{line_id}: line exceeds scene sample duration")
        if topology == "EMBEDDED_FMV" and bool(line.get("subtitle_authorized")):
            required = ("movie_identity_verified", "card_identity_verified", "card_timebase_verified")
            if not bool(scene.get("movie_identity_verified")) and not bool(line.get("movie_identity_verified")):
                raise ContractError(f"{line_id}: movie identity evidence missing")
            if any(not bool(line.get(key)) for key in required[1:]):
                raise ContractError(f"{line_id}: card/timebase evidence missing")
        if line.get("speech_start") is not None and _number(line["speech_start"], "speech_start") < start:
            raise ContractError(f"{line_id}: speech_start is before window")
        if line.get("speech_end") is not None and _number(line["speech_end"], "speech_end") > end:
            raise ContractError(f"{line_id}: speech_end is after window")
        _validate_reference_segments(line)
        windows.append((start, end, line_id))
    if topology == "EMBEDDED_FMV":
        ordered = sorted(windows)
        allow_overlap = bool((scene.get("extensions") or {}).get("allow_overlap"))
        if not allow_overlap:
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1] - 1e-9:
                    raise ContractError(f"{scene_id}: undeclared overlapping windows {previous[2]} and {current[2]}")


def normalize_manifest(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and "scenes" in value:
        scenes = value["scenes"]
    elif isinstance(value, Mapping) and "id" in value and "lines" in value:
        scenes = [value]
    elif isinstance(value, list):
        scenes = value
    else:
        raise ContractError("manifest must be a scene object, a list, or an object with scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("manifest scenes must be a non-empty list")
    result = [dict(scene) for scene in scenes]
    for scene in result:
        validate_scene_value(scene)
    return result


def validate_manifest_value(value: Any) -> ManifestSummary:
    scenes = normalize_manifest(value)
    return ManifestSummary(len(scenes), sum(len(scene["lines"]) for scene in scenes), tuple(sorted({str(scene.get("topology", "LINE_SEPARATED")) for scene in scenes})))


__all__ = ["ManifestSummary", "TOPOLOGIES", "normalize_manifest", "validate_manifest_value", "validate_scene_value"]
