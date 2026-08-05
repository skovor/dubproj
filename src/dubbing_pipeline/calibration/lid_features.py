"""Frozen features and row contract for independent LID calibration.

LID is deliberately not a variant of the alignment feature contract.  Keeping
its row type separate prevents a target/final row from being silently reused
for source-language calibration (and makes the schema visible in every
serialized training example).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping
LID_FEATURE_SCHEMA_VERSION = "lid-fusion-v3"
PERFORMANCE_MODES = ("UNRESOLVED", "NEUTRAL", "FAST", "WHISPER", "SHOUT", "SCREAM_SPEECH", "CRYING_SPEECH", "EFFORT", "LAUGH_SPEECH")
LID_FEATURES = ("lid_source_probability", "lid_target_probability", "whisper_source_probability", "whisper_target_probability", "ctc_target_raw_score", "ctc_target_calibrated_probability", "duration_seconds", "speech_ratio") + tuple(f"performance_mode_{mode.lower()}" for mode in PERFORMANCE_MODES)

@dataclass(frozen=True)
class LIDFeatureRow:
    clip_id: str
    split: str
    split_group: str
    label: int
    features: dict[str, float]
    performance_mode: str = "NEUTRAL"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.clip_id).strip() or not str(self.split_group).strip():
            raise ValueError("LID rows require clip_id and split_group")
        if self.split not in {"calibration", "validation", "hidden_test"}:
            raise ValueError("invalid split")
        if self.label not in (0, 1):
            raise ValueError("LID labels must be 0 or 1")
        if set(self.features) != set(LID_FEATURES):
            missing = sorted(set(LID_FEATURES) - set(self.features))
            unknown = sorted(set(self.features) - set(LID_FEATURES))
            raise ValueError(f"LID feature schema mismatch; missing={missing}, unknown={unknown}")
        if not all(math.isfinite(float(self.features[name])) for name in LID_FEATURES):
            raise ValueError("non-finite LID feature")

    def to_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "split": self.split, "split_group": self.split_group,
                "label": self.label, "features": dict(self.features),
                "performance_mode": self.performance_mode, "metadata": dict(self.metadata or {}),
                "feature_schema_version": LID_FEATURE_SCHEMA_VERSION}

def _language_code(value: Any) -> str:
    return str(value or "").casefold().split("-", 1)[0]


def features(value: Mapping[str, Any], *, performance_mode: str = "NEUTRAL", source_language: str | None = None, target_language: str | None = None) -> dict[str, float]:
    probs = value.get("probabilities") if isinstance(value.get("probabilities"), Mapping) else {}
    raw_ctc = value.get("ctc_target_raw_score", value.get("ctc_raw_score", 0.0))
    calibrated_ctc = value.get("ctc_target_calibrated_probability", 0.0)
    source_code = _language_code(source_language or value.get("source_language") or "en")
    target_code = _language_code(target_language or value.get("target_language") or "de")
    selected_mode = str(performance_mode or "UNRESOLVED").upper()
    if selected_mode not in PERFORMANCE_MODES:
        selected_mode = "UNRESOLVED"
    out = {"lid_source_probability": float(value.get("lid_source_probability", probs.get(source_code, 0.0))), "lid_target_probability": float(value.get("lid_target_probability", probs.get(target_code, 0.0))), "whisper_source_probability": float(value.get("whisper_source_probability", 0.0)), "whisper_target_probability": float(value.get("whisper_target_probability", 0.0)), "ctc_target_raw_score": float(raw_ctc), "ctc_target_calibrated_probability": float(calibrated_ctc), "duration_seconds": float(value.get("duration_seconds", value.get("duration", 0.0))), "speech_ratio": float(value.get("speech_ratio", 0.0))}
    out.update({f"performance_mode_{mode.lower()}": 1.0 if mode == selected_mode else 0.0 for mode in PERFORMANCE_MODES})
    if not all(math.isfinite(v) for v in out.values()): raise ValueError("non-finite LID feature")
    return out

__all__ = ["LID_FEATURE_SCHEMA_VERSION", "PERFORMANCE_MODES", "LID_FEATURES", "LIDFeatureRow", "features"]
