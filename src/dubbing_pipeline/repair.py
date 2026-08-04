"""Bounded causal repair operations; uncertainty is held, not regenerated."""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from .attempts import AttemptSignature, AttemptStore

class FailureCause(str, enum.Enum):
    FINAL_ANCHOR_MISSING="FINAL_ANCHOR_MISSING"; ACTIVE_BODY_OVERFLOW="ACTIVE_BODY_OVERFLOW"; VOWEL_TOO_SHORT="VOWEL_TOO_SHORT"; SEAM_FAIL="SEAM_FAIL"; REFERENCE_LIMITED="REFERENCE_LIMITED"; LANGUAGE_LEAK_CONFIRMED="LANGUAGE_LEAK_CONFIRMED"; ASR_UNCERTAIN="ASR_UNCERTAIN"; DETERMINISTIC_CALIBRATION="DETERMINISTIC_CALIBRATION"

@dataclass(frozen=True)
class RepairAction:
    strategy: str
    cause: FailureCause
    parameters: dict[str, Any] = field(default_factory=dict)
    allows_tts: bool = False
    max_attempts: int = 1
    rationale: str = ""

@dataclass(frozen=True)
class RepairOutcome:
    status: str
    action: RepairAction | None
    attempt_id: str | None = None
    output_audio_sha256: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

def apply_repair(action: RepairAction, *, line_id: str, input_audio_sha256: str, reference_sha256: str | None, store: AttemptStore, executor: Callable[[RepairAction], Mapping[str, Any]] | None = None) -> RepairOutcome:
    signature=AttemptSignature(line_id,input_audio_sha256,action.strategy,action.parameters,reference_sha256)
    if store.seen(signature): return RepairOutcome("DUPLICATE_ATTEMPT",action,signature.digest(),diagnostics={"reason":"same causal attempt already recorded"})
    if action.cause in {FailureCause.ASR_UNCERTAIN, FailureCause.DETERMINISTIC_CALIBRATION}: return RepairOutcome("HOLD_NO_TTS",action,signature.digest(),diagnostics={"reason":"requires evidence/configuration, not regeneration"})
    result=dict(executor(action) if executor else {"status":"PLANNED"}); store.record(signature,result,result.get("status","PLANNED")); return RepairOutcome(result.get("status","PLANNED"),action,signature.digest(),result.get("output_audio_sha256"),result)

__all__=["FailureCause","RepairAction","RepairOutcome","apply_repair"]
