"""Feature schema used for performance diagnostics (never lexical authority)."""
from __future__ import annotations
from typing import Any, Mapping
FEATURE_SCHEMA="performance-features-v2"
PERFORMANCE_MODES=("UNRESOLVED","NEUTRAL","FAST","WHISPER","SHOUT","SCREAM_SPEECH","CRYING_SPEECH","EFFORT","LAUGH_SPEECH")
FEATURES=("rms_dbfs","pitch_hz","speech_ratio","duration_seconds","zero_crossing_rate","spectral_centroid_hz")
CATEGORICAL_FEATURES=tuple(f"performance_mode_{mode.lower()}" for mode in PERFORMANCE_MODES)

def normalize(value: Mapping[str, Any]) -> dict[str, float]:
    result={name:float(value.get(name,0.0) or 0.0) for name in FEATURES}
    import math
    if not all(math.isfinite(x) for x in result.values()): raise ValueError("non-finite performance feature")
    return result

def categorical_mode(mode: str | None) -> dict[str, float]:
    selected=str(mode or "UNRESOLVED").upper()
    if selected not in PERFORMANCE_MODES:
        selected="UNRESOLVED"
    return {name: 1.0 if name == f"performance_mode_{selected.lower()}" else 0.0 for name in CATEGORICAL_FEATURES}

__all__=["FEATURE_SCHEMA","PERFORMANCE_MODES","FEATURES","CATEGORICAL_FEATURES","normalize","categorical_mode"]
