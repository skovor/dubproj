"""Post-transform QA and durable stage evidence.

Raw TTS quality is not delivery quality.  This module gives every transform a
named, serialisable audit and provides the small amount of atomic audio I/O
needed to audit the artifact that will actually be mounted or shipped.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audio import read, write
from .contracts import FailureClass, GateEvidence, GateStatus
from .hashing import contract_hash, sha256_file
from .qa_v2 import LanguageProfile, QAResultV2, evaluate_candidate_v2


POST_TRANSFORM_STAGES = (
    "RAW_TECHNICAL_QA",
    "PROCESSED_QA",
    "MOUNTED_QA",
    "SERIALIZED_QA",
    "SCENE_QA",
)


@dataclass
class StageAudit:
    """Evidence for one artifact at one point in the delivery pipeline."""

    stage: str
    passed: bool
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    qa_hash: str | None = None
    gates: dict[str, GateEvidence] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_class: FailureClass | None = None
    # Kept in memory for candidate selection; omitted from the public shape.
    result: QAResultV2 | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.stage not in POST_TRANSFORM_STAGES:
            raise ValueError(f"unknown post-transform QA stage: {self.stage}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": bool(self.passed),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "qa_hash": self.qa_hash,
            "gates": {name: gate.to_dict() for name, gate in self.gates.items()},
            "diagnostics": self.diagnostics,
            "failure_class": self.failure_class.value if self.failure_class else None,
        }


def _audio_failure(stage: str, path: str | Path | None, error: Exception) -> StageAudit:
    message = str(error)
    gate = GateEvidence(
        "serialization_contract",
        GateStatus.ERROR,
        details={"error": message},
    )
    return StageAudit(
        stage=stage,
        passed=False,
        artifact_path=str(path) if path is not None else None,
        gates={"serialization_contract": gate},
        diagnostics={"error": message},
        failure_class=FailureClass.DETERMINISTIC_SERIALIZATION,
    )


def audit_candidate_stage(
    path: str | Path,
    *,
    stage: str,
    expected_text: str,
    source_text: str = "",
    target_sample_rate: int | None = None,
    target_frames: int | None = None,
    channels: int | None = None,
    reference_end: float | None = None,
    transcript: str | None = None,
    language: str | None = None,
    language_probability: float | None = None,
    profile: LanguageProfile | None = None,
    hard_gates: Sequence[str] | None = None,
    final_word_min_tokens: int = 1,
    tail_guard_seconds: float = 0.08,
    splice_metrics: Mapping[str, tuple[bool, Any, Any, str]] | None = None,
    preserved_ok: bool | None = None,
    require_asr: bool = True,
    neutral_effort: bool = False,
    linguistic_evidence: Mapping[str, Any] | None = None,
    alignment_evidence: Mapping[str, Any] | None = None,
    lid_evidence: Mapping[str, Any] | None = None,
    alignment_min_target_score: float = .65,
    alignment_min_margin: float = .20,
    alignment_source_leak_score: float = .75,
    calibration_authority: bool = False,
    calibration_profile: Mapping[str, Any] | None = None,
    calibration_profile_root: str | Path | None = None,
    feature_schema_version: str = "char-alignment-v2",
    backend_id: str | None = None,
    runtime_lock_sha256: str | None = None,
    models_lock_sha256: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    performance_mode: str | None = None,
) -> StageAudit:
    """Run the normal candidate QA against a persisted artifact.

    ``evaluate_candidate_v2`` always reopens the path.  Calling it for each
    stage makes a processed or mounted in-memory array unable to masquerade as
    the delivered file.
    """
    if stage not in POST_TRANSFORM_STAGES:
        raise ValueError(f"unknown post-transform QA stage: {stage}")
    path = Path(path)
    try:
        result = evaluate_candidate_v2(
            str(path),
            expected_text=expected_text,
            source_text=source_text,
            target_sample_rate=target_sample_rate,
            target_frames=target_frames,
            channels=channels,
            reference_end=reference_end,
            transcript=transcript,
            language=language,
            language_probability=language_probability,
            profile=profile,
            hard_gates=hard_gates,
            final_word_min_tokens=final_word_min_tokens,
            tail_guard_seconds=tail_guard_seconds,
            splice_metrics=splice_metrics,
            preserved_ok=preserved_ok,
            require_asr=require_asr,
            neutral_effort=neutral_effort,
            linguistic_evidence=linguistic_evidence,
            alignment_evidence=alignment_evidence,
            lid_evidence=lid_evidence,
            alignment_min_target_score=alignment_min_target_score,
            alignment_min_margin=alignment_min_margin,
            alignment_source_leak_score=alignment_source_leak_score,
            calibration_authority=calibration_authority,
            calibration_profile=calibration_profile,
            calibration_profile_root=calibration_profile_root,
            feature_schema_version=feature_schema_version,
            backend_id=backend_id,
            runtime_lock_sha256=runtime_lock_sha256,
            models_lock_sha256=models_lock_sha256,
            model_id=model_id,
            model_revision=model_revision,
            performance_mode=performance_mode,
        )
        artifact_sha = sha256_file(path)
        qa_hash = contract_hash(
            "qa-v2",
            {
                "stage": stage,
                "artifact_sha256": artifact_sha,
                "result": result.to_dict(),
            },
        )
        return StageAudit(
            stage=stage,
            passed=result.passed,
            artifact_path=str(path),
            artifact_sha256=artifact_sha,
            qa_hash=qa_hash,
            gates=result.gates,
            diagnostics=result.diagnostics,
            failure_class=result.failure_class,
            result=result,
        )
    except Exception as exc:  # an audit error is a failed artifact, never a pass
        return _audio_failure(stage, path, exc)


def audit_scene_stage(
    path: str | Path,
    *,
    stage: str = "SCENE_QA",
    expected_sample_rate: int | None = None,
    expected_frames: int | None = None,
    expected_channels: int | None = None,
    protected_intervals_ok: bool | None = None,
    untouched_channels_ok: bool | None = None,
) -> StageAudit:
    """Audit a full mounted scene after serialization and reopening."""
    if stage not in {"SERIALIZED_QA", "SCENE_QA"}:
        raise ValueError("scene audits are only valid for SERIALIZED_QA or SCENE_QA")
    path = Path(path)
    try:
        import numpy as np

        audio, sample_rate = read(path, always_2d=True)
        gates: dict[str, GateEvidence] = {
            "serialization_contract": GateEvidence("serialization_contract", GateStatus.PASS, measured_value=True, details={"reopened": True}),
            "not_empty": GateEvidence("not_empty", GateStatus.PASS if len(audio) > 0 else GateStatus.FAIL, measured_value=int(len(audio)), threshold=1, units="frames"),
            "finite_audio": GateEvidence("finite_audio", GateStatus.PASS if bool(np.isfinite(audio).all()) else GateStatus.FAIL, measured_value=bool(np.isfinite(audio).all())),
            "sample_rate": GateEvidence("sample_rate", GateStatus.NOT_APPLICABLE if expected_sample_rate is None else (GateStatus.PASS if sample_rate == expected_sample_rate else GateStatus.FAIL), measured_value=sample_rate, threshold=expected_sample_rate, units="Hz"),
            "channels": GateEvidence("channels", GateStatus.NOT_APPLICABLE if expected_channels is None else (GateStatus.PASS if audio.shape[1] == expected_channels else GateStatus.FAIL), measured_value=int(audio.shape[1]), threshold=expected_channels, units="channels"),
            "frames": GateEvidence("frames", GateStatus.NOT_APPLICABLE if expected_frames is None else (GateStatus.PASS if len(audio) == expected_frames else GateStatus.FAIL), measured_value=int(len(audio)), threshold=expected_frames, units="frames"),
        }
        if protected_intervals_ok is not None:
            gates["preserved_intervals"] = GateEvidence("preserved_intervals", GateStatus.PASS if protected_intervals_ok else GateStatus.FAIL, measured_value=protected_intervals_ok)
        else:
            gates["preserved_intervals"] = GateEvidence("preserved_intervals", GateStatus.NOT_APPLICABLE)
        if untouched_channels_ok is not None:
            gates["untouched_channels"] = GateEvidence("untouched_channels", GateStatus.PASS if untouched_channels_ok else GateStatus.FAIL, measured_value=untouched_channels_ok)
        else:
            gates["untouched_channels"] = GateEvidence("untouched_channels", GateStatus.NOT_APPLICABLE)
        failures = [name for name, gate in gates.items() if gate.status is GateStatus.FAIL or gate.status is GateStatus.ERROR]
        passed = not failures and all(gate.status is not GateStatus.NOT_RUN for gate in gates.values())
        diagnostics = {"sample_rate": sample_rate, "frames": int(len(audio)), "channels": int(audio.shape[1]), "failed_gates": failures}
        artifact_sha = sha256_file(path)
        qa_hash = contract_hash("qa-v2", {"stage": stage, "artifact_sha256": artifact_sha, "gates": {key: value.to_dict() for key, value in gates.items()}})
        return StageAudit(
            stage=stage,
            passed=passed,
            artifact_path=str(path),
            artifact_sha256=artifact_sha,
            qa_hash=qa_hash,
            gates=gates,
            diagnostics=diagnostics,
            failure_class=FailureClass.DETERMINISTIC_SERIALIZATION if failures else None,
        )
    except Exception as exc:
        return _audio_failure(stage, path, exc)


def persist_audio_atomic(path: str | Path, audio: Any, sample_rate: int) -> Path:
    """Persist audio, fsync the temporary WAV, replace atomically, reopen it."""
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(audio, dtype="float32")
    if value.ndim not in (1, 2) or len(value) == 0 or not bool(np.isfinite(value).all()):
        raise ValueError("cannot persist empty or non-finite audio")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".wav", dir=str(target.parent))
    temporary = Path(name)
    try:
        os.close(descriptor)
        write(temporary, value, int(sample_rate))
        check, rate = read(temporary, always_2d=True)
        if rate != int(sample_rate) or len(check) != len(value) or not bool(np.isfinite(check).all()):
            raise IOError("serialized temporary audio failed readback contract")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # The post-transform QA will reopen again; this immediate readback
        # catches a partial/invalid replacement before it enters the queue.
        reopened, reopened_rate = read(target, always_2d=True)
        if reopened_rate != int(sample_rate) or len(reopened) != len(check):
            raise IOError("serialized audio failed final readback contract")
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["POST_TRANSFORM_STAGES", "StageAudit", "audit_candidate_stage", "audit_scene_stage", "persist_audio_atomic"]
