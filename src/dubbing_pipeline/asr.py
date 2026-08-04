"""ASR evidence used for screening and confirmation, never for mapping authority.

The V2 pipeline deliberately treats ASR as evidence rather than truth.  A
candidate is read twice: once with the target language forced and once with
automatic language detection.  The two readings are cached by the SHA-256 of
the audio artifact (and by an explicit semantic alias when a transform is
known to preserve speech).  This keeps the expensive linguistic QA out of
serialization-only stages without silently reusing evidence after a speech
changing transform.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .hashing import atomic_json, canonical_json, contract_hash, sha256_bytes, sha256_file


class ASRBackend(Protocol):
    def transcribe(self, path: str | Path, *, language: str | None = None) -> dict[str, Any]: ...


class ForcedAligner(Protocol):
    """Optional escalation interface; WhisperX/MFA adapters implement this later."""

    def align(self, path: str | Path, *, text: str, language: str) -> dict[str, Any]: ...


class FasterWhisperBackend:
    """Lazy faster-whisper adapter; model stays alive for one QA round."""

    def __init__(self, model_size: str = "large-v3-turbo", *, device: str = "cuda", compute_type: str = "float16"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("install the optional qa dependencies") from exc
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    backend_id = "faster-whisper-large-v3-turbo"

    def transcribe(self, path: str | Path, *, language: str | None = None) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            str(path), language=language, beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        rows = list(segments)
        text = " ".join(item.text.strip() for item in rows).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "probability": getattr(info, "language_probability", None),
            "segments": [{"start": item.start, "end": item.end, "text": item.text} for item in rows],
        }


@dataclass(frozen=True)
class ASRReading:
    """One reproducible ASR hypothesis for one artifact and one decode mode."""

    mode: str
    text: str
    language: str | None
    probability: float | None
    segments: list[dict[str, Any]]
    audio_sha256: str
    evidence_hash: str
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "text": self.text,
            "language": self.language,
            "probability": self.probability,
            "segments": self.segments,
            "audio_sha256": self.audio_sha256,
            "evidence_hash": self.evidence_hash,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, audio_sha256: str | None = None, cache_hit: bool = True) -> "ASRReading":
        return cls(
            mode=str(value.get("mode", "unknown")),
            text=str(value.get("text", "")),
            language=value.get("language"),
            probability=float(value["probability"]) if value.get("probability") is not None else None,
            segments=[dict(item) for item in value.get("segments", [])],
            audio_sha256=str(audio_sha256 or value.get("audio_sha256", "")),
            evidence_hash=str(value.get("evidence_hash", "")),
            cache_hit=cache_hit,
        )


@dataclass(frozen=True)
class DualASREvidence:
    """The two ASR views that a linguistic decision is allowed to consume."""

    audio_sha256: str
    source_language: str
    target_language: str
    forced_target: ASRReading
    automatic: ASRReading
    semantic_key: str | None = None

    @property
    def forced_transcript(self) -> str:
        return self.forced_target.text

    @property
    def automatic_transcript(self) -> str:
        return self.automatic.text

    @property
    def detected_language(self) -> str | None:
        return self.automatic.language

    @property
    def evidence_hashes(self) -> list[str]:
        return [item.evidence_hash for item in (self.forced_target, self.automatic) if item.evidence_hash]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_sha256": self.audio_sha256,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "forced_target": self.forced_target.to_dict(),
            "automatic": self.automatic.to_dict(),
            "semantic_key": self.semantic_key,
            "evidence_hashes": self.evidence_hashes,
        }


class ASRCache:
    """Memory/disk cache keyed by artifact SHA, decode mode and backend."""

    def __init__(self, root: str | Path | None = None, *, backend_id: str = "unknown") -> None:
        self.root = Path(root) if root is not None else None
        self.backend_id = backend_id
        self._readings: dict[str, dict[str, Any]] = {}
        self._semantic: dict[str, dict[str, Any]] = {}

    def _key(self, audio_sha256: str, mode: str, language: str | None) -> str:
        return sha256_bytes(canonical_json({"audio_sha256": audio_sha256, "mode": mode, "language": language, "backend": self.backend_id}))

    def _semantic_key(self, semantic_key: str, mode: str, language: str | None) -> str:
        return sha256_bytes(canonical_json({"semantic_key": semantic_key, "mode": mode, "language": language, "backend": self.backend_id}))

    def _path(self, key: str) -> Path | None:
        return self.root / f"{key}.json" if self.root is not None else None

    def _get_disk(self, path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def get(self, audio_sha256: str, mode: str, language: str | None) -> dict[str, Any] | None:
        key = self._key(audio_sha256, mode, language)
        value = self._readings.get(key) or self._get_disk(self._path(key))
        if value is not None:
            self._readings[key] = value
        return value

    def put(self, audio_sha256: str, mode: str, language: str | None, value: Mapping[str, Any]) -> None:
        key = self._key(audio_sha256, mode, language)
        payload = dict(value)
        self._readings[key] = payload
        path = self._path(key)
        if path is not None:
            atomic_json(path, payload)

    def get_semantic(self, semantic_key: str, mode: str, language: str | None) -> dict[str, Any] | None:
        key = self._semantic_key(semantic_key, mode, language)
        value = self._semantic.get(key) or self._get_disk(self._path(f"semantic-{key}"))
        if value is not None:
            self._semantic[key] = value
        return value

    def put_semantic(self, semantic_key: str, mode: str, language: str | None, value: Mapping[str, Any]) -> None:
        key = self._semantic_key(semantic_key, mode, language)
        payload = dict(value)
        self._semantic[key] = payload
        path = self._path(f"semantic-{key}")
        if path is not None:
            atomic_json(path, payload)


def _backend_id(backend: Any) -> str:
    return str(getattr(backend, "backend_id", f"{backend.__class__.__module__}.{backend.__class__.__qualname__}"))


def _call_transcribe(backend: Any, path: str | Path, language: str | None) -> dict[str, Any]:
    """Support old test/adapters that only accept ``transcribe(path)``."""
    try:
        value = backend.transcribe(str(path), language=language)
    except TypeError as exc:
        if "language" not in str(exc) and "keyword" not in str(exc):
            raise
        value = backend.transcribe(str(path))
    if isinstance(value, dict):
        return dict(value)
    return {"text": str(value), "language": None, "probability": None, "segments": []}


def _reading_from_value(value: Mapping[str, Any], *, mode: str, audio_sha256: str, cache_hit: bool) -> ASRReading:
    payload = {
        "mode": mode,
        "text": str(value.get("text", "")),
        "language": value.get("language"),
        "probability": value.get("probability"),
        "segments": [dict(item) for item in value.get("segments", [])],
    }
    evidence_hash = contract_hash("asr-reading-v3", {**payload, "audio_sha256": audio_sha256})
    return ASRReading(
        mode=mode,
        text=payload["text"],
        language=payload["language"],
        probability=float(payload["probability"]) if payload["probability"] is not None else None,
        segments=payload["segments"],
        audio_sha256=audio_sha256,
        evidence_hash=evidence_hash,
        cache_hit=cache_hit,
    )


def transcribe_dual(
    backend: ASRBackend,
    path: str | Path,
    *,
    source_language: str,
    target_language: str,
    cache: ASRCache | None = None,
    semantic_key: str | None = None,
) -> DualASREvidence:
    """Collect forced-target and automatic-language evidence exactly once."""
    audio_sha256 = sha256_file(path)
    cache = cache or ASRCache(backend_id=_backend_id(backend))
    if cache.backend_id == "unknown":
        cache.backend_id = _backend_id(backend)

    def one(mode: str, language: str | None) -> ASRReading:
        value = cache.get(audio_sha256, mode, language)
        cache_hit = value is not None
        if value is None and semantic_key:
            value = cache.get_semantic(semantic_key, mode, language)
            cache_hit = value is not None
        if value is None:
            value = _call_transcribe(backend, path, language)
            cache.put(audio_sha256, mode, language, value)
            if semantic_key:
                cache.put_semantic(semantic_key, mode, language, value)
        return _reading_from_value(value, mode=mode, audio_sha256=audio_sha256, cache_hit=cache_hit)

    forced = one("forced_target", target_language)
    automatic = one("automatic", None)
    return DualASREvidence(audio_sha256, source_language, target_language, forced, automatic, semantic_key)


@dataclass(frozen=True)
class WhisperXEscalationRequest:
    """Serializable request prepared for a future selective WhisperX/MFA pass."""

    audio_path: str
    expected_text: str
    source_text: str
    language: str
    reason: str
    evidence_hashes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "expected_text": self.expected_text,
            "source_text": self.source_text,
            "language": self.language,
            "reason": self.reason,
            "evidence_hashes": list(self.evidence_hashes),
            "status": "PENDING",
            "backend": "whisperx_or_mfa",
        }


def prepare_whisperx_escalation(
    path: str | Path,
    *,
    expected_text: str,
    source_text: str = "",
    language: str = "de",
    reason: str = "ASR_UNCERTAIN",
    evidence_hashes: list[str] | tuple[str, ...] = (),
) -> WhisperXEscalationRequest:
    """Create an escalation request without loading or running WhisperX."""
    return WhisperXEscalationRequest(str(path), expected_text, source_text, language, reason, tuple(evidence_hashes))


@dataclass
class ASRScreeningResult:
    text: str
    language: str | None
    probability: float | None
    confirmed_target: bool = False
    segments: list[dict[str, Any]] | None = None


def screen_and_confirm(backend: ASRBackend, path: str | Path, *, source_language: str, target_language: str) -> ASRScreeningResult:
    evidence = transcribe_dual(backend, path, source_language=source_language, target_language=target_language)
    source = evidence.automatic.to_dict()
    language = source.get("language"); probability = source.get("probability")
    target = evidence.forced_target.to_dict()
    return ASRScreeningResult(
        text=str(source.get("text", "")), language=language, probability=probability,
        confirmed_target=bool(target.get("text", "").strip()) and (target.get("language") in {None, target_language}) and (language in {None, target_language}),
        segments=source.get("segments"),
    )


__all__ = [
    "ASRBackend", "ASRCache", "ASRReading", "ASRScreeningResult", "DualASREvidence",
    "ForcedAligner", "FasterWhisperBackend", "WhisperXEscalationRequest",
    "prepare_whisperx_escalation", "screen_and_confirm", "transcribe_dual",
]
