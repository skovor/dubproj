"""Independent spoken-language evidence and conservative fusion policy."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from .hashing import canonical_json, sha256_bytes, sha256_file

@dataclass(frozen=True)
class LIDPolicy:
    minimum_duration_seconds: float = .45
    minimum_speech_ratio: float = .20
    minimum_confidence: float = .70
    source_language: str = "en"
    target_language: str = "de"

@dataclass(frozen=True)
class LIDEvidence:
    status: str
    language: str | None
    probabilities: dict[str, float]
    backend_id: str
    model_id: str
    model_revision: str
    audio_sha256: str
    duration_seconds: float
    sample_rate: int
    speech_ratio: float
    evidence_hash: str
    reason: str = ""
    record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)


def independent_lid(backend: Any, audio_path: str | Path, *, policy: LIDPolicy, duration_seconds: float, speech_ratio: float, sample_rate: int, audio_sha256: str | None = None) -> LIDEvidence:
    digest = audio_sha256 or sha256_file(audio_path)
    backend_id = str(getattr(backend, "backend_id", "independent-lid")); model_id = str(getattr(backend, "model_id", "unknown")); revision = str(getattr(backend, "model_revision", "unknown"))
    if duration_seconds < policy.minimum_duration_seconds: return _evidence("LID_NOT_APPLICABLE", None, {}, backend_id, model_id, revision, digest, duration_seconds, sample_rate, speech_ratio, "clip_too_short")
    if speech_ratio < policy.minimum_speech_ratio: return _evidence("LID_NOT_APPLICABLE", None, {}, backend_id, model_id, revision, digest, duration_seconds, sample_rate, speech_ratio, "insufficient_speech_or_nonlinguistic_effort")
    try: raw = backend.detect(str(audio_path), sample_rate=16000)
    except TypeError: raw = backend.detect(str(audio_path))
    if not isinstance(raw, Mapping): raise ValueError("independent LID backend must return a mapping")
    probabilities = {str(key).casefold().split("-", 1)[0]: float(value) for key, value in (raw.get("probabilities") or raw.get("scores") or {}).items() if math.isfinite(float(value))}
    language = str(raw.get("language") or (max(probabilities, key=probabilities.get) if probabilities else "")).casefold().split("-", 1)[0] or None
    confidence = float(raw.get("probability", probabilities.get(language or "", 0.0)) or 0.0)
    if language and language not in probabilities:
        probabilities[language] = confidence
    status = "LID_CONFIDENT" if language and confidence >= policy.minimum_confidence else "LID_UNCERTAIN"
    return _evidence(status, language, probabilities, backend_id, model_id, revision, digest, duration_seconds, sample_rate, speech_ratio, "" if status == "LID_CONFIDENT" else "confidence_below_threshold")


def fuse_language_evidence(*, whisper_language: str | None, whisper_probability: float | None, lid: LIDEvidence | None, ctc_target_probability: float | None = None, ctc_target_raw_score: float | None = None, ctc_target_calibrated_probability: float | None = None, policy: LIDPolicy) -> dict[str, Any]:
    """Fuse independent LID with Whisper/CTC without treating either as truth."""
    if lid is None or lid.status == "LID_NOT_APPLICABLE": return {"status": "LID_NOT_APPLICABLE", "reason": lid.reason if lid else "backend_unavailable", "evidence_families": ["WHISPER_ASR"]}
    whisper_source = str(whisper_language or "").casefold().split("-", 1)[0] == policy.source_language and float(whisper_probability or 0.0) >= .70
    lid_source = lid.language == policy.source_language and float(lid.probabilities.get(policy.source_language, 0.0)) >= policy.minimum_confidence
    lid_target = lid.language == policy.target_language and float(lid.probabilities.get(policy.target_language, 0.0)) >= policy.minimum_confidence
    raw_ctc = float(ctc_target_raw_score) if ctc_target_raw_score is not None and math.isfinite(float(ctc_target_raw_score)) else None
    legacy_ctc = float(ctc_target_probability) if ctc_target_probability is not None and math.isfinite(float(ctc_target_probability)) else None
    # ``ctc_target_probability`` is a legacy name and is diagnostic-only.  A
    # hard fusion branch may consume only a probability produced by the
    # versioned CTC calibrator.
    ctc = float(ctc_target_calibrated_probability) if ctc_target_calibrated_probability is not None and math.isfinite(float(ctc_target_calibrated_probability)) else None
    if whisper_source and lid_source and ctc is not None and ctc < .45: status = "LANGUAGE_LEAK_CONFIRMED"
    elif lid_source and ctc is not None and ctc >= .80: status = "EVIDENCE_CONFLICT"
    elif lid.status == "LID_UNCERTAIN" or (not lid_source and not lid_target): status = "LID_UNCERTAIN"
    else: status = "NO_LANGUAGE_LEAK_EVIDENCE"
    return {"status": status, "whisper_source": whisper_source, "lid_source": lid_source, "lid_target": lid_target, "ctc_target_raw_score": raw_ctc, "ctc_target_calibrated_probability": ctc, "legacy_ctc_target_probability": legacy_ctc, "ctc_calibration_status": "CALIBRATED" if ctc is not None else "UNAVAILABLE", "evidence_families": ["WHISPER_ASR", "AUDIO_LANGUAGE_ID"], "evidence_hashes": [lid.evidence_hash], "reason": lid.reason}


def _evidence(status, language, probabilities, backend_id, model_id, revision, digest, duration, sample_rate, speech_ratio, reason):
    payload = {"status": status, "language": language, "probabilities": probabilities, "audio_sha256": digest, "backend_id": backend_id, "model_id": model_id, "model_revision": revision, "duration": duration, "sample_rate": sample_rate, "speech_ratio": speech_ratio, "reason": reason}
    evidence_hash = sha256_bytes(canonical_json(payload))
    record = {"evidence_id": evidence_hash, "evidence_family": "AUDIO_LANGUAGE_ID", "backend_id": backend_id, "model_id": model_id, "model_revision": revision, "mode": "spoken_language_id", "audio_sha256": digest, "semantic_key": None, "output": {"status": status, "language": language, "probabilities": probabilities, "probability": probabilities.get(language or "", 0.0), "duration_seconds": duration, "speech_ratio": speech_ratio}, "confidence": probabilities.get(language or "", 0.0), "evidence_hash": evidence_hash}
    return LIDEvidence(status, language, probabilities, backend_id, model_id, revision, digest, duration, sample_rate, speech_ratio, evidence_hash, reason, record)

__all__ = ["LIDPolicy", "LIDEvidence", "independent_lid", "fuse_language_evidence"]
