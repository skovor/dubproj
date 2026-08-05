"""Bounded causal repair operations; uncertainty is held, not regenerated."""
from __future__ import annotations
import enum
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from .attempts import AttemptSignature, AttemptStore

class FailureCause(str, enum.Enum):
    FINAL_ANCHOR_MISSING="FINAL_ANCHOR_MISSING"; ACTIVE_BODY_OVERFLOW="ACTIVE_BODY_OVERFLOW"; VOWEL_TOO_SHORT="VOWEL_TOO_SHORT"; SEAM_FAIL="SEAM_FAIL"; REFERENCE_LIMITED="REFERENCE_LIMITED"; LANGUAGE_LEAK_CONFIRMED="LANGUAGE_LEAK_CONFIRMED"; LANGUAGE_LEAK_SUSPECTED="LANGUAGE_LEAK_SUSPECTED"; ASR_UNCERTAIN="ASR_UNCERTAIN"; DETERMINISTIC_CALIBRATION="DETERMINISTIC_CALIBRATION"

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
    if action.cause in {FailureCause.ASR_UNCERTAIN, FailureCause.LANGUAGE_LEAK_SUSPECTED, FailureCause.DETERMINISTIC_CALIBRATION}: return RepairOutcome("HOLD_NO_TTS",action,signature.digest(),diagnostics={"reason":"requires evidence/configuration, not regeneration"})
    used = store.count_causal(signature)
    if used >= max(0, int(action.max_attempts)):
        return RepairOutcome("BUDGET_EXHAUSTED", action, signature.digest(), diagnostics={"attempts": used, "max_attempts": action.max_attempts})
    if executor is None:
        result = {"status": "BLOCKED_NO_EXECUTOR", "reason": "causal repair requires a concrete audio executor"}
    else:
        result = dict(executor(action))
        result.setdefault("status", "EXECUTOR_NO_STATUS")
        output_path = result.get("output_audio_path")
        if output_path and Path(str(output_path)).is_file():
            result["output_audio_sha256"] = hashlib.sha256(Path(str(output_path)).read_bytes()).hexdigest()
    status = str(result["status"])
    store.record(signature,result,status)
    return RepairOutcome(status,action,signature.digest(),result.get("output_audio_sha256"),{**result,"attempts":used + 1,"max_attempts":action.max_attempts})


def re_audit_repair(outcome: RepairOutcome, *, output_audio_path: str | Path, auditor: Callable[[str], Mapping[str, Any]]) -> RepairOutcome:
    """Reopen, hash and audit a repair before it can be selected.

    A repair executor's status is never a QA result.  The output must exist,
    its bytes are hashed after reopening, and the supplied auditor must return
    an explicit boolean.  This function is intentionally side-effect free
    with respect to the attempt ledger so a failed re-audit cannot create a
    hidden retry.
    """
    path = Path(output_audio_path)
    if outcome.status not in {"PASS", "EXECUTOR_NO_STATUS", "BLOCKED_NO_EXECUTOR"} or not path.is_file():
        return RepairOutcome("REPAIR_QA_BLOCKED", outcome.action, outcome.attempt_id, diagnostics={"reason": "repair output is unavailable or was not executed"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        evidence = dict(auditor(str(path)))
    except Exception as exc:
        return RepairOutcome("REPAIR_QA_ERROR", outcome.action, outcome.attempt_id, digest, {"error": str(exc), "output_audio_sha256": digest})
    passed = evidence.get("passed") is True
    return RepairOutcome("REPAIR_QA_PASS" if passed else "REPAIR_QA_FAIL", outcome.action, outcome.attempt_id, digest, {"output_audio_sha256": digest, "qa": evidence})

__all__=["FailureCause","RepairAction","RepairOutcome","apply_repair","re_audit_repair"]
