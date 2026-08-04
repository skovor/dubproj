"""Strict, serialisable contracts used by the V2 runtime.

The legacy models remain available for characterization.  V2 deliberately uses
typed evidence objects so that a missing measurement is represented as
``NOT_RUN`` instead of a misleading boolean PASS.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when an artifact or manifest violates a V2 contract."""


class GateStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"


class FailureClass(str, enum.Enum):
    DETERMINISTIC_REFERENCE = "DETERMINISTIC_REFERENCE"
    DETERMINISTIC_TEXT = "DETERMINISTIC_TEXT"
    DETERMINISTIC_MAPPING = "DETERMINISTIC_MAPPING"
    DETERMINISTIC_WINDOW = "DETERMINISTIC_WINDOW"
    DETERMINISTIC_PROCESSING = "DETERMINISTIC_PROCESSING"
    DETERMINISTIC_SERIALIZATION = "DETERMINISTIC_SERIALIZATION"
    STOCHASTIC_TTS = "STOCHASTIC_TTS"
    ASR_UNCERTAIN = "ASR_UNCERTAIN"
    PERCEPTUAL_REVIEW = "PERCEPTUAL_REVIEW"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"


class EvidenceFamily(str, enum.Enum):
    """Families are independent only when their acoustic/model path differs."""

    WHISPER_ASR = "WHISPER_ASR"
    CTC_FORCED_ALIGNER = "CTC_FORCED_ALIGNER"
    KALDI_FORCED_ALIGNER = "KALDI_FORCED_ALIGNER"
    AUDIO_LANGUAGE_ID = "AUDIO_LANGUAGE_ID"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass(frozen=True)
class EvidenceRecord:
    """Auditable output of one evidence family and one decode/alignment mode."""

    evidence_id: str
    evidence_family: EvidenceFamily | str
    backend_id: str
    model_id: str
    model_revision: str
    mode: str
    audio_sha256: str
    semantic_key: str | None
    output: Any
    confidence: float | None
    evidence_hash: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.backend_id or not self.mode or not self.audio_sha256:
            raise ContractError("evidence record identity fields must be non-empty")
        if not isinstance(self.evidence_family, EvidenceFamily):
            object.__setattr__(self, "evidence_family", EvidenceFamily(str(self.evidence_family)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_family": self.evidence_family.value,
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "mode": self.mode,
            "audio_sha256": self.audio_sha256,
            "semantic_key": self.semantic_key,
            "output": self.output,
            "confidence": self.confidence,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        known = {"evidence_id", "evidence_family", "backend_id", "model_id", "model_revision", "mode", "audio_sha256", "semantic_key", "output", "confidence", "evidence_hash"}
        _unknown(value, known, "evidence")
        return cls(
            evidence_id=str(_required(value, "evidence_id")),
            evidence_family=EvidenceFamily(str(_required(value, "evidence_family"))),
            backend_id=str(_required(value, "backend_id")),
            model_id=str(_required(value, "model_id")),
            model_revision=str(value.get("model_revision", "unknown")),
            mode=str(_required(value, "mode")),
            audio_sha256=str(_required(value, "audio_sha256")),
            semantic_key=value.get("semantic_key"),
            output=value.get("output"),
            confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
            evidence_hash=str(_required(value, "evidence_hash")),
        )


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value or value[key] in (None, ""):
        raise ContractError(f"missing required field: {key}")
    return value[key]


def _unknown(value: Mapping[str, Any], known: set[str], kind: str) -> None:
    extra = set(value) - known - {"extensions"}
    if extra:
        raise ContractError(f"unknown {kind} fields: {sorted(extra)}")


@dataclass(frozen=True)
class GateEvidence:
    gate_name: str
    status: GateStatus
    measured_value: Any = None
    threshold: Any = None
    units: str | None = None
    evidence_hash: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    implementation_version: str = "v2"

    def __post_init__(self) -> None:
        if not self.gate_name.strip():
            raise ContractError("gate_name must be non-empty")
        if not isinstance(self.status, GateStatus):
            object.__setattr__(self, "status", GateStatus(str(self.status)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "units": self.units,
            "evidence_hash": self.evidence_hash,
            "details": self.details,
            "implementation_version": self.implementation_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateEvidence":
        known = {"gate_name", "status", "measured_value", "threshold", "units", "evidence_hash", "details", "implementation_version"}
        _unknown(value, known, "gate")
        return cls(
            gate_name=str(_required(value, "gate_name")),
            status=GateStatus(str(_required(value, "status"))),
            measured_value=value.get("measured_value"), threshold=value.get("threshold"),
            units=value.get("units"), evidence_hash=value.get("evidence_hash"),
            details=dict(value.get("details") or {}), implementation_version=str(value.get("implementation_version", "v2")),
        )


@dataclass(frozen=True)
class ReferenceEvidence:
    reference_id: str
    audio_path: str
    audio_sha256: str
    native_sample_rate: int
    channels: int
    samples: int
    start_sample: int
    end_sample: int
    channel: int | None
    exact_transcript: str
    language: str
    speaker_id: str
    source_line_id: str
    extraction_tool: str
    extraction_tool_version: str
    validation_hash: str
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.native_sample_rate <= 0 or self.channels <= 0 or self.samples <= 0:
            raise ContractError("reference audio spec must be positive")
        if not 0 <= self.start_sample < self.end_sample <= self.samples:
            raise ContractError("reference sample range is outside the materialized audio")
        if self.channel is not None and not 0 <= self.channel < self.channels:
            raise ContractError("reference channel is outside the audio")
        if not self.exact_transcript.strip():
            raise ContractError("exact_transcript must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"reference_id": self.reference_id, "audio_path": self.audio_path, "audio_sha256": self.audio_sha256,
                "native_sample_rate": self.native_sample_rate, "channels": self.channels, "samples": self.samples,
                "start_sample": self.start_sample, "end_sample": self.end_sample, "channel": self.channel,
                "exact_transcript": self.exact_transcript, "language": self.language, "speaker_id": self.speaker_id,
                "source_line_id": self.source_line_id, "extraction_tool": self.extraction_tool,
                "extraction_tool_version": self.extraction_tool_version, "validation_hash": self.validation_hash,
                "validated_at": self.validated_at}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReferenceEvidence":
        known = {"reference_id", "audio_path", "audio_sha256", "native_sample_rate", "channels", "samples", "start_sample", "end_sample", "channel", "exact_transcript", "language", "speaker_id", "source_line_id", "extraction_tool", "extraction_tool_version", "validation_hash", "validated_at"}
        _unknown(value, known, "reference")
        return cls(reference_id=str(_required(value, "reference_id")), audio_path=str(_required(value, "audio_path")),
                    audio_sha256=str(_required(value, "audio_sha256")), native_sample_rate=int(_required(value, "native_sample_rate")),
                    channels=int(_required(value, "channels")), samples=int(_required(value, "samples")),
                    start_sample=int(_required(value, "start_sample")), end_sample=int(_required(value, "end_sample")),
                    channel=int(value["channel"]) if value.get("channel") is not None else None,
                    exact_transcript=str(_required(value, "exact_transcript")), language=str(_required(value, "language")),
                    speaker_id=str(_required(value, "speaker_id")), source_line_id=str(_required(value, "source_line_id")),
                    extraction_tool=str(_required(value, "extraction_tool")), extraction_tool_version=str(_required(value, "extraction_tool_version")),
                    validation_hash=str(_required(value, "validation_hash")), validated_at=str(value.get("validated_at", datetime.now(timezone.utc).isoformat())))


@dataclass(frozen=True)
class AudioArtifact:
    path: str
    sha256: str
    native_sample_rate: int
    frames: int
    channels: int
    subtype: str
    duration_seconds: float
    nonfinite_samples: int
    clipping_samples: int
    producer: str
    producer_version: str

    def __post_init__(self) -> None:
        if self.native_sample_rate <= 0 or self.frames <= 0 or self.channels <= 0:
            raise ContractError("audio artifact dimensions must be positive")
        if self.nonfinite_samples < 0 or self.clipping_samples < 0:
            raise ContractError("audio diagnostics cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AudioArtifact":
        known = set(cls.__dataclass_fields__)
        _unknown(value, known, "audio")
        return cls(**{key: value[key] for key in known})


@dataclass(frozen=True)
class DeliveryWindow:
    scene_id: str
    line_id: str
    start_sample: int
    end_sample: int
    speech_start_sample: int
    speech_end_sample: int
    preserved_source_intervals: tuple[tuple[int, int], ...] = ()
    source_resume_sample: int | None = None
    dialogue_channel: int = 0
    timebase_hash: str = ""
    ownership_id: str = ""

    def __post_init__(self) -> None:
        if not self.scene_id or not self.line_id or self.end_sample <= self.start_sample:
            raise ContractError("invalid delivery window")
        if not self.start_sample <= self.speech_start_sample <= self.speech_end_sample <= self.end_sample:
            raise ContractError("speech interval must be within delivery window")
        for start, end in self.preserved_source_intervals:
            if not self.start_sample <= start < end <= self.end_sample:
                raise ContractError("preserved interval outside delivery window")

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["preserved_source_intervals"] = [list(item) for item in self.preserved_source_intervals]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeliveryWindow":
        known = set(cls.__dataclass_fields__)
        _unknown(value, known, "delivery window")
        intervals = tuple((int(item[0]), int(item[1])) for item in value.get("preserved_source_intervals", ()))
        return cls(scene_id=str(_required(value, "scene_id")), line_id=str(_required(value, "line_id")),
                   start_sample=int(_required(value, "start_sample")), end_sample=int(_required(value, "end_sample")),
                   speech_start_sample=int(_required(value, "speech_start_sample")), speech_end_sample=int(_required(value, "speech_end_sample")),
                   preserved_source_intervals=intervals, source_resume_sample=int(value["source_resume_sample"]) if value.get("source_resume_sample") is not None else None,
                   dialogue_channel=int(value.get("dialogue_channel", 0)), timebase_hash=str(value.get("timebase_hash", "")), ownership_id=str(value.get("ownership_id", "")))


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    line_id: str
    generation_hash: str
    processing_hash: str | None
    qa_hash: str | None
    round_index: int
    take_index: int
    seed: int | None
    raw_audio: str
    processed_audio: str | None
    mounted_delivery: str | None
    status: str
    failure_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RunState:
    run_id: str
    phase: str
    statuses: dict[str, str] = field(default_factory=dict)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    cursor: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def gate_passes(gate: GateEvidence, *, allow_not_applicable: bool = True) -> bool:
    return gate.status is GateStatus.PASS or (allow_not_applicable and gate.status is GateStatus.NOT_APPLICABLE)


def serialize_roundtrip(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"not a V2 contract: {type(value)!r}")


__all__ = ["AudioArtifact", "CandidateArtifact", "ContractError", "DeliveryWindow", "EvidenceFamily", "EvidenceRecord", "FailureClass", "GateEvidence", "GateStatus", "ReferenceEvidence", "RunState", "gate_passes"]
