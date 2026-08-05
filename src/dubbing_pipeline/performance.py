"""Performance-mode contracts and deterministic candidate measurements.

The declared mode is routing metadata.  It is never inferred from a subtitle
window's length.  A measured mode is derived only after reopening the audio
artifact, so short clips, music and noise cannot silently become ``FAST`` or
``EFFORT`` deliveries.
"""
from __future__ import annotations
import enum, math
from dataclasses import dataclass
from pathlib import Path
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
    declared_mode: str | None = None
    measured: bool = False

    def to_dict(self):
        return {
            "mode": self.mode.value,
            "declared_mode": self.declared_mode,
            "measured": self.measured,
            "confidence": self.confidence,
            "rms_dbfs": self.rms_dbfs,
            "pitch_hz": self.pitch_hz,
            "speech_ratio": self.speech_ratio,
            "duration_seconds": self.duration_seconds,
            "source": self.source,
            "reason": self.reason,
        }

def classify_performance(*, metadata: Mapping[str, Any] | None = None, rms_dbfs: float | None = None, pitch_hz: float | None = None, speech_ratio: float | None = None, duration_seconds: float | None = None) -> PerformanceEvidence:
    metadata = metadata or {}; hint = str(metadata.get("performance_mode") or metadata.get("mode") or "").upper()
    try: mode = PerformanceMode(hint)
    except ValueError: mode = None
    if mode is not None:
        return PerformanceEvidence(mode, 1.0, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, "metadata", "explicit_performance_mode", declared_mode=mode.value)
    # Acoustic classification is deliberately conservative.  A low speech
    # ratio by itself is compatible with music/noise, not proof of an effort.
    if speech_ratio is not None and speech_ratio >= .12 and rms_dbfs is not None and rms_dbfs < -48:
        return PerformanceEvidence(PerformanceMode.WHISPER, .62, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, source="acoustic", reason="low_energy", measured=True)
    if speech_ratio is not None and speech_ratio >= .12 and rms_dbfs is not None and rms_dbfs > -8 and pitch_hz is not None and pitch_hz > 250:
        return PerformanceEvidence(PerformanceMode.SHOUT, .62, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, source="acoustic", reason="high_energy_pitch", measured=True)
    return PerformanceEvidence(PerformanceMode.UNRESOLVED, 0.0, rms_dbfs, pitch_hz, speech_ratio, duration_seconds, source="unresolved", reason="no_explicit_or_acoustic_signature", measured=any(value is not None for value in (rms_dbfs, pitch_hz, speech_ratio, duration_seconds)))

def extract_basic_features(samples: Sequence[float], sample_rate: int) -> dict[str, float]:
    if sample_rate <= 0: raise ValueError("sample_rate must be positive")
    values=[float(x) for x in samples]
    if not values: raise ValueError("empty audio")
    rms=(sum(x*x for x in values)/len(values))**.5
    rms_dbfs=20*math.log10(max(rms,1e-9))
    crossings=sum(1 for a,b in zip(values,values[1:]) if (a<0) != (b<0))
    duration=len(values)/sample_rate
    activity_threshold=max(1e-4, rms * .20)
    active=[abs(value) > activity_threshold for value in values]
    speech_ratio=sum(active)/len(values)
    pitch_hz = _estimate_pitch(values, sample_rate, active)
    return {"rms_dbfs":rms_dbfs,"pitch_hz":pitch_hz or 0.0,"speech_ratio":speech_ratio,"zero_crossing_rate":crossings/max(1,len(values)-1),"duration_seconds":duration}


def _estimate_pitch(values: Sequence[float], sample_rate: int, active: Sequence[bool]) -> float | None:
    """Return a conservative F0 estimate, or ``None`` for unvoiced material."""
    active_values=[value for value, enabled in zip(values, active) if enabled]
    if len(active_values) < max(32, int(sample_rate * .02)):
        return None
    mean=sum(active_values)/len(active_values)
    centered=[value - mean for value in active_values]
    energy=sum(value * value for value in centered)
    if energy <= 1e-12:
        return None
    low_lag=max(1, int(sample_rate / 400.0)); high_lag=max(low_lag + 1, int(sample_rate / 70.0))
    best_lag=None; best_score=0.0
    for lag in range(low_lag, min(high_lag, len(centered) - 1) + 1):
        numerator=sum(centered[index] * centered[index + lag] for index in range(len(centered) - lag))
        denominator=(sum(value * value for value in centered[:-lag]) * sum(value * value for value in centered[lag:])) ** .5
        score=numerator / denominator if denominator > 1e-12 else 0.0
        if score > best_score:
            best_score=score; best_lag=lag
    return float(sample_rate / best_lag) if best_lag is not None and best_score >= .30 else None


def measure_audio(path: str | Path, *, declared: PerformanceEvidence | None = None) -> PerformanceEvidence:
    """Reopen an audio artifact and attach measured evidence to its declaration."""
    from .audio import read
    audio, sample_rate = read(path, always_2d=True)
    values=audio[:, 0].tolist()
    features=extract_basic_features(values, int(sample_rate))
    declared = declared or PerformanceEvidence(PerformanceMode.UNRESOLVED, 0.0)
    if declared.mode is not PerformanceMode.UNRESOLVED:
        return PerformanceEvidence(declared.mode, declared.confidence, features["rms_dbfs"], features.get("pitch_hz"), features["speech_ratio"], features["duration_seconds"], source="declared+measured", reason=declared.reason, declared_mode=declared.mode.value, measured=True)
    measured=classify_performance(metadata={}, rms_dbfs=features["rms_dbfs"], pitch_hz=features.get("pitch_hz"), speech_ratio=features["speech_ratio"], duration_seconds=features["duration_seconds"])
    return PerformanceEvidence(measured.mode, measured.confidence, measured.rms_dbfs, measured.pitch_hz, measured.speech_ratio, measured.duration_seconds, source=measured.source, reason=measured.reason, declared_mode=declared.mode.value, measured=True)

__all__=["PerformanceMode","PerformanceContract","PerformanceEvidence","classify_performance","extract_basic_features","measure_audio"]
