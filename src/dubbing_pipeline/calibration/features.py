"""Frozen feature extraction for target and final-anchor calibration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

FEATURE_SCHEMA_VERSION = "char-alignment-v2"
NORMALIZATION_VERSION = "alignment-text-normalization-v1"
TARGET_FEATURES = (
    "target_score", "native_char_coverage", "mean_char_score", "minimum_char_score", "p10_char_score",
    "delete_ratio", "substitute_ratio", "insert_ratio", "interpolated_ratio", "compression_ratio",
    "characters_per_second", "words_per_second", "duration", "performance_mode",
)
FINAL_ANCHOR_FEATURES = (
    "final_coverage", "final_minimum_score", "final_mean_score", "final_duration", "gap_to_active_speech_end_ms",
    "final_delete_count", "final_substitute_count", "insertions_inside_anchor", "final_interpolated",
)

@dataclass(frozen=True)
class FeatureRow:
    clip_id: str
    split: str
    split_group: str
    label: int
    features: dict[str, float]
    performance_mode: str = "NEUTRAL"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "validation", "hidden_test"}: raise ValueError("invalid split")
        if self.label not in (0, 1): raise ValueError("calibration labels must be 0 or 1")
        expected = set(TARGET_FEATURES) | set(FINAL_ANCHOR_FEATURES)
        unknown = set(self.features) - expected
        if unknown: raise ValueError(f"unknown calibration features: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "split": self.split, "split_group": self.split_group, "label": self.label, "features": dict(self.features), "performance_mode": self.performance_mode, "metadata": dict(self.metadata or {})}


def _float(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return default


def target_features(evidence: Mapping[str, Any], *, performance_mode: str = "NEUTRAL") -> dict[str, float]:
    """Read exactly the frozen target fields; missing values are a hard error."""
    char_count = max(1.0, _float(evidence.get("expected_characters"), 1.0))
    word_count = max(1.0, _float(evidence.get("expected_words"), 1.0))
    duration = max(1e-6, _float(evidence.get("duration"), 0.0))
    values = {
        "target_score": _float(evidence.get("raw_target_score", evidence.get("target_score"))),
        "native_char_coverage": _float(evidence.get("native_char_coverage")),
        "mean_char_score": _float(evidence.get("mean_char_score")),
        "minimum_char_score": _float(evidence.get("minimum_char_score")),
        "p10_char_score": _float(evidence.get("p10_char_score")),
        "delete_ratio": _float(evidence.get("delete_ratio", _float(evidence.get("delete_count")) / char_count)),
        "substitute_ratio": _float(evidence.get("substitute_ratio", _float(evidence.get("substitute_count")) / char_count)),
        "insert_ratio": _float(evidence.get("insert_ratio", _float(evidence.get("insert_count")) / char_count)),
        "interpolated_ratio": _float(evidence.get("interpolated_ratio", _float(evidence.get("interpolated_count")) / char_count)),
        "compression_ratio": _float(evidence.get("compression_ratio")),
        "characters_per_second": _float(evidence.get("characters_per_second", char_count / duration)),
        "words_per_second": _float(evidence.get("words_per_second", word_count / duration)),
        "duration": duration,
        "performance_mode": _performance_code(performance_mode),
    }
    _finite(values)
    return values


def final_anchor_features(evidence: Mapping[str, Any]) -> dict[str, float]:
    anchor = evidence.get("final_anchor_evidence") if isinstance(evidence.get("final_anchor_evidence"), Mapping) else evidence
    values = {
        "final_coverage": _float(anchor.get("coverage", anchor.get("final_coverage"))),
        "final_minimum_score": _float(anchor.get("minimum_score", anchor.get("final_minimum_score"))),
        "final_mean_score": _float(anchor.get("mean_score", anchor.get("final_mean_score"))),
        "final_duration": _float(anchor.get("duration_ms", anchor.get("final_duration"))) / (1000.0 if "duration_ms" in anchor else 1.0),
        "gap_to_active_speech_end_ms": _float(anchor.get("gap_to_active_speech_end_ms")),
        "final_delete_count": _float(anchor.get("deleted_characters", anchor.get("delete_count"))),
        "final_substitute_count": _float(anchor.get("substituted_characters", anchor.get("substitute_count"))),
        "insertions_inside_anchor": _float(anchor.get("insertions_inside_anchor", anchor.get("insert_count"))),
        "final_interpolated": 1.0 if bool(anchor.get("interpolated", False)) else 0.0,
    }
    _finite(values); return values


def _performance_code(value: str) -> float:
    return {"NEUTRAL": 0.0, "FAST": 1.0, "WHISPER": 2.0, "SHOUT": 3.0, "SCREAM_SPEECH": 4.0, "CRYING_SPEECH": 5.0, "EFFORT": 6.0, "LAUGH_SPEECH": 7.0}.get(str(value).upper(), 0.0)


def _finite(values: Mapping[str, float]) -> None:
    import math
    if not all(math.isfinite(value) for value in values.values()): raise ValueError("non-finite calibration feature")


__all__ = ["FEATURE_SCHEMA_VERSION", "NORMALIZATION_VERSION", "TARGET_FEATURES", "FINAL_ANCHOR_FEATURES", "FeatureRow", "target_features", "final_anchor_features"]
