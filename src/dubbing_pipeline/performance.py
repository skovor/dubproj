"""Performance-mode contracts and lightweight acoustic classification."""
from __future__ import annotations
import enum, math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

class PerformanceMode(str, enum.Enum):
    UNRESOLVED="UNRESOLVED"; NEUTRAL="NEUTRAL"; FAST="FAST"; WHISPER="WHISPER"; SHOUT="SHOUT"; SCREAM_SPEECH="SCREAM_SPEECH"; CRYING_SPEECH="CRYING_SPEECH"; EFFORT="EFFORT"; LAUGH_SPEECH="LAUGH_SPEECH"

@dataclass(frozen=True)
class PerformanceContract:
    semantic_text: str
    performance_text: str
    delivery_target: str
    performance_mode: PerformanceMode = PerformanceMode.NEUTRAL

@dataclass(frozen=True)
class PerformanceEvidence:
    mode: PerformanceMode
    confidence: float
    rms_dbfs: float | None = None
    pitch_hz: float | None = None
    speech_ratio: float | None = None
    duration_seconds: float | None = None
    source: str = "diagnostic"
    reason: str = ""

    def to_dict(self): return {"mode": self.mode.value, "confidence": self.confidence, "rms_dbfs": self.rms_dbfs, "pitch_hz": self.pitch_hz, "speech_ratio": self.speech_ratio, "duration_seconds": self.duration_seconds, "source": self.source, "reason": self.reason}

def classify_performance(*, metadata: Mapping[str, Any] | None = None, rms_dbfs: float | None = None, pitch_hz: float | None = None, speech_ratio: float | None = None, duration_seconds: float | None = None) -> PerformanceEvidence:
    metadata = metadata or {}; hint = str(metadata.get("performance_mode") or metadata.get("mode") or "").upper()
    try: mode = PerformanceMode(hint)
    except ValueError: mode = None
    if mode is not None: return PerformanceEvidence(mode, 1.0, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, "metadata", "explicit_performance_mode")
    if speech_ratio is not None and speech_ratio < .12: return PerformanceEvidence(PerformanceMode.EFFORT, .65, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, reason="low_speech_ratio")
    if rms_dbfs is not None and rms_dbfs < -48: return PerformanceEvidence(PerformanceMode.WHISPER, .62, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, reason="low_energy")
    if rms_dbfs is not None and rms_dbfs > -8 and pitch_hz is not None and pitch_hz > 250: return PerformanceEvidence(PerformanceMode.SHOUT, .62, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, reason="high_energy_pitch")
    if duration_seconds is not None and duration_seconds < 1.0: return PerformanceEvidence(PerformanceMode.FAST, .55, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, reason="short_delivery")
    return PerformanceEvidence(PerformanceMode.UNRESOLVED, 0.0, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, source="unresolved", reason="no_explicit_or_acoustic_signature")

def extract_basic_features(samples: Sequence[float], sample_rate: int) -> dict[str, float]:
    if sample_rate <= 0: raise ValueError("sample_rate must be positive")
    values=[float(x) for x in samples]
    if not values: raise ValueError("empty audio")
    rms=(sum(x*x for x in values)/len(values))**.5; rms_dbfs=20*math.log10(max(rms,1e-9)); crossings=sum(1 for a,b in zip(values,values[1:]) if (a<0) != (b<0)); duration=len(values)/sample_rate
    return {"rms_dbfs":rms_dbfs,"zero_crossing_rate":crossings/max(1,len(values)-1),"duration_seconds":duration}

__all__=["PerformanceMode","PerformanceContract","PerformanceEvidence","classify_performance","extract_basic_features"]
