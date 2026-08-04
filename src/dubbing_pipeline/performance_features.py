"""Feature schema used for performance diagnostics (never lexical authority)."""
from __future__ import annotations
from typing import Any, Mapping
FEATURE_SCHEMA="performance-features-v1"
FEATURES=("rms_dbfs","pitch_hz","speech_ratio","duration_seconds","zero_crossing_rate","spectral_centroid_hz")

def normalize(value: Mapping[str, Any]) -> dict[str, float]:
    result={name:float(value.get(name,0.0) or 0.0) for name in FEATURES}
    import math
    if not all(math.isfinite(x) for x in result.values()): raise ValueError("non-finite performance feature")
    return result

__all__=["FEATURE_SCHEMA","FEATURES","normalize"]
