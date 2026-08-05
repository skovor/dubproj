"""Mode-specific QA requirements."""
from __future__ import annotations
from dataclasses import dataclass
from .performance import PerformanceMode

@dataclass(frozen=True)
class ModePolicy:
    mode: PerformanceMode
    require_content: bool
    require_final_word: bool
    require_loudness: bool
    require_pitch_identity: bool
    max_duration_error_ms: float | None
    notes: str

_POLICIES={
    PerformanceMode.UNRESOLVED: ModePolicy(PerformanceMode.UNRESOLVED, True, True, False, False, None, "performance unresolved; retain lexical gates and block promotion"),
    PerformanceMode.NEUTRAL: ModePolicy(PerformanceMode.NEUTRAL,True,True,True,True,80.0,"full semantic and delivery QA"),
    PerformanceMode.FAST: ModePolicy(PerformanceMode.FAST,True,True,True,False,100.0,"rate is diagnostic; preserve words"),
    PerformanceMode.WHISPER: ModePolicy(PerformanceMode.WHISPER,True,True,False,False,120.0,"energy gate relaxed"),
    PerformanceMode.SHOUT: ModePolicy(PerformanceMode.SHOUT,True,True,True,False,120.0,"clipping and timing still hard"),
    PerformanceMode.SCREAM_SPEECH: ModePolicy(PerformanceMode.SCREAM_SPEECH,True,True,False,False,150.0,"spectral/pitch diagnostics only"),
    PerformanceMode.CRYING_SPEECH: ModePolicy(PerformanceMode.CRYING_SPEECH,True,True,False,False,140.0,"pitch instability is expected"),
    PerformanceMode.EFFORT: ModePolicy(PerformanceMode.EFFORT,False,False,False,False,180.0,"nonlinguistic effort; do not force words"),
    PerformanceMode.LAUGH_SPEECH: ModePolicy(PerformanceMode.LAUGH_SPEECH,True,False,False,False,160.0,"laughter pattern is diagnostic"),
}

def policy_for(mode: PerformanceMode | str) -> ModePolicy:
    return _POLICIES[PerformanceMode(mode)]

__all__=["ModePolicy","policy_for"]
