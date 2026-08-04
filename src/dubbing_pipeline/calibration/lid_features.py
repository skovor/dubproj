"""Frozen features for independent LID calibration."""
from __future__ import annotations
import math
from typing import Any, Mapping
LID_FEATURE_SCHEMA_VERSION = "lid-fusion-v1"
LID_FEATURES = ("lid_source_probability", "lid_target_probability", "whisper_source_probability", "ctc_target_probability", "duration_seconds", "speech_ratio", "performance_mode")

def features(value: Mapping[str, Any], *, performance_mode: str = "NEUTRAL") -> dict[str, float]:
    probs = value.get("probabilities") if isinstance(value.get("probabilities"), Mapping) else {}
    out = {"lid_source_probability": float(value.get("lid_source_probability", probs.get("en", 0.0))), "lid_target_probability": float(value.get("lid_target_probability", probs.get("de", 0.0))), "whisper_source_probability": float(value.get("whisper_source_probability", 0.0)), "ctc_target_probability": float(value.get("ctc_target_probability", 0.0)), "duration_seconds": float(value.get("duration_seconds", value.get("duration", 0.0))), "speech_ratio": float(value.get("speech_ratio", 0.0)), "performance_mode": {"NEUTRAL":0.0,"FAST":1.0,"WHISPER":2.0,"SHOUT":3.0,"SCREAM_SPEECH":4.0,"CRYING_SPEECH":5.0,"EFFORT":6.0,"LAUGH_SPEECH":7.0}.get(str(performance_mode).upper(), 0.0)}
    if not all(math.isfinite(v) for v in out.values()): raise ValueError("non-finite LID feature")
    return out

__all__ = ["LID_FEATURE_SCHEMA_VERSION", "LID_FEATURES", "features"]
