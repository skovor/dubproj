"""Frozen feature extraction for target and final-anchor calibration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

FEATURE_SCHEMA_VERSION = "char-alignment-v3"
NORMALIZATION_VERSION = "alignment-text-normalization-v2"
PERFORMANCE_MODES = ("UNRESOLVED", "NEUTRAL", "FAST", "WHISPER", "SHOUT", "SCREAM_SPEECH", "CRYING_SPEECH", "EFFORT", "LAUGH_SPEECH")
PERFORMANCE_MODE_FEATURES = tuple(f"performance_mode_{mode.lower()}" for mode in PERFORMANCE_MODES)
TARGET_FEATURES = (
    "target_score", "native_char_coverage", "mean_char_score", "minimum_char_score", "p10_char_score",
    "delete_ratio", "substitute_ratio", "insert_ratio", "interpolated_ratio", "compression_ratio",
    "characters_per_second", "words_per_second", "duration",
)
TARGET_FEATURES += PERFORMANCE_MODE_FEATURES
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


def _required(value: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in value and value[name] not in (None, ""):
            try:
                result = float(value[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"calibration feature {name} is not numeric") from exc
            if not __import__("math").isfinite(result):
                raise ValueError(f"calibration feature {name} is non-finite")
            return result
    raise ValueError(f"missing calibration evidence field; expected one of {names}")


def target_features(evidence: Mapping[str, Any], *, performance_mode: str = "NEUTRAL") -> dict[str, float]:
    """Read exactly the frozen target fields; missing values are a hard error."""
    char_count = _required(evidence, "expected_characters")
    word_count = _required(evidence, "expected_words")
    duration = _required(evidence, "duration")
    if char_count <= 0 or word_count <= 0 or duration <= 0:
        raise ValueError("expected character/word counts and duration must be positive")
    values = {
        "target_score": _required(evidence, "raw_target_score", "target_score"),
        "native_char_coverage": _required(evidence, "native_char_coverage"),
        "mean_char_score": _required(evidence, "mean_char_score"),
        "minimum_char_score": _required(evidence, "minimum_char_score"),
        "p10_char_score": _required(evidence, "p10_char_score"),
        "delete_ratio": _required(evidence, "delete_ratio") if "delete_ratio" in evidence else _required(evidence, "delete_count") / char_count,
        "substitute_ratio": _required(evidence, "substitute_ratio") if "substitute_ratio" in evidence else _required(evidence, "substitute_count") / char_count,
        "insert_ratio": _required(evidence, "insert_ratio") if "insert_ratio" in evidence else _required(evidence, "insert_count") / char_count,
        "interpolated_ratio": _required(evidence, "interpolated_ratio") if "interpolated_ratio" in evidence else _required(evidence, "interpolated_count") / char_count,
        "compression_ratio": _required(evidence, "compression_ratio"),
        "characters_per_second": _required(evidence, "characters_per_second") if "characters_per_second" in evidence else char_count / duration,
        "words_per_second": _required(evidence, "words_per_second") if "words_per_second" in evidence else word_count / duration,
        "duration": duration,
    }
    values.update(_performance_one_hot(performance_mode))
    _finite(values)
    return values


def final_anchor_features(evidence: Mapping[str, Any]) -> dict[str, float]:
    anchor = evidence.get("final_anchor_evidence") if isinstance(evidence.get("final_anchor_evidence"), Mapping) else evidence
    if not anchor:
        raise ValueError("final-anchor evidence is missing")
    values = {
        "final_coverage": _required(anchor, "coverage", "final_coverage"),
        "final_minimum_score": _required(anchor, "minimum_score", "final_minimum_score"),
        "final_mean_score": _required(anchor, "mean_score", "final_mean_score"),
        "final_duration": _required(anchor, "duration_ms") / 1000.0 if "duration_ms" in anchor else _required(anchor, "final_duration"),
        "gap_to_active_speech_end_ms": _required(anchor, "gap_to_active_speech_end_ms"),
        "final_delete_count": _required(anchor, "deleted_characters", "delete_count"),
        "final_substitute_count": _required(anchor, "substituted_characters", "substitute_count"),
        "insertions_inside_anchor": _required(anchor, "insertions_inside_anchor", "insert_count"),
        "final_interpolated": 1.0 if "interpolated" in anchor and bool(anchor["interpolated"]) else (_required(anchor, "final_interpolated") if "final_interpolated" in anchor else 0.0),
    }
    _finite(values); return values


def _performance_one_hot(value: str) -> dict[str, float]:
    selected = str(value or "UNRESOLVED").upper()
    if selected not in PERFORMANCE_MODES:
        selected = "UNRESOLVED"
    return {name: 1.0 if name == f"performance_mode_{selected.lower()}" else 0.0 for name in PERFORMANCE_MODE_FEATURES}


def _finite(values: Mapping[str, float]) -> None:
    import math
    if not all(math.isfinite(value) for value in values.values()): raise ValueError("non-finite calibration feature")


__all__ = ["FEATURE_SCHEMA_VERSION", "NORMALIZATION_VERSION", "PERFORMANCE_MODES", "PERFORMANCE_MODE_FEATURES", "TARGET_FEATURES", "FINAL_ANCHOR_FEATURES", "FeatureRow", "target_features", "final_anchor_features"]
