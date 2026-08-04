"""Selective evidence from a second acoustic/model family.

This module intentionally keeps WhisperX, MFA and SpeechBrain optional.  The
pipeline can prepare/cache requests without importing any of those packages;
only a selected provisional winner or an uncertain candidate is escalated.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .contracts import EvidenceFamily, EvidenceRecord
from .hashing import atomic_json, canonical_json, contract_hash, sha256_bytes, sha256_file


class AlignmentUnavailable(RuntimeError):
    """Raised when an optional alignment family is not installed/configured."""


def _fold_word(value: Any) -> str:
    """Compare aligned words without punctuation/diacritic differences."""
    normal = unicodedata.normalize("NFKD", str(value or "").casefold())
    normal = "".join(char for char in normal if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", "", normal, flags=re.UNICODE)


class CTCAligner(Protocol):
    evidence_family: EvidenceFamily | str
    backend_id: str
    model_id: str
    model_revision: str

    def align(self, path: str | Path, *, text: str, language: str) -> dict[str, Any]: ...


class LanguageIdentifier(Protocol):
    evidence_family: EvidenceFamily | str
    backend_id: str
    model_id: str
    model_revision: str

    def detect(self, path: str | Path) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AlignmentCache:
    """Disk-backed cache keying alignment by audio, text and model revision."""

    root: Path | None = None
    backend_id: str = "unknown"
    model_id: str = "unknown"
    model_revision: str = "unknown"

    def _key(self, audio_sha256: str, text: str, language: str) -> str:
        return sha256_bytes(canonical_json({
            "audio_sha256": audio_sha256,
            "text": text,
            "language": language,
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }))

    def _path(self, key: str) -> Path | None:
        return self.root / f"{key}.json" if self.root is not None else None

    def get(self, audio_sha256: str, text: str, language: str) -> dict[str, Any] | None:
        path = self._path(self._key(audio_sha256, text, language))
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def put(self, audio_sha256: str, text: str, language: str, value: Mapping[str, Any]) -> None:
        path = self._path(self._key(audio_sha256, text, language))
        if path is not None:
            atomic_json(path, dict(value))


@dataclass(frozen=True)
class AlignmentReading:
    text: str
    language: str
    score: float
    coverage: float | None
    final_anchor_present: bool | None
    words: list[dict[str, Any]]
    record: EvidenceRecord
    cache_hit: bool = False
    char_segments: list[dict[str, Any]] = field(default_factory=list)
    native_char_coverage: float | None = None
    mean_char_score: float | None = None
    minimum_char_score: float | None = None
    p10_char_score: float | None = None
    unaligned_characters: list[str] = field(default_factory=list)
    interpolated_characters: list[str] = field(default_factory=list)
    compression_ratio: float | None = None
    final_anchor_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "score": self.score,
            "coverage": self.coverage,
            "final_anchor_present": self.final_anchor_present,
            "words": self.words,
            "char_segments": [dict(item) for item in (self.char_segments or [])],
            "native_char_coverage": self.native_char_coverage,
            "mean_char_score": self.mean_char_score,
            "minimum_char_score": self.minimum_char_score,
            "p10_char_score": self.p10_char_score,
            "unaligned_characters": list(self.unaligned_characters or []),
            "interpolated_characters": list(self.interpolated_characters or []),
            "compression_ratio": self.compression_ratio,
            "final_anchor_evidence": dict(self.final_anchor_evidence) if self.final_anchor_evidence is not None else None,
            "record": self.record.to_dict(),
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class ContrastiveAlignment:
    target: AlignmentReading
    source: AlignmentReading | None
    target_score: float
    source_score: float | None
    margin: float | None
    evidence_records: list[EvidenceRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "source": self.source.to_dict() if self.source is not None else None,
            "target_score": self.target_score,
            "source_score": self.source_score,
            "margin": self.margin,
            "evidence_records": [item.to_dict() for item in self.evidence_records],
            "evidence_families": sorted({item.evidence_family.value for item in self.evidence_records}),
        }


def _record(
    *,
    family: EvidenceFamily | str,
    backend_id: str,
    model_id: str,
    model_revision: str,
    mode: str,
    audio_sha256: str,
    semantic_key: str | None,
    output: Mapping[str, Any],
    confidence: float | None,
) -> EvidenceRecord:
    payload = {
        "evidence_family": str(family.value if isinstance(family, EvidenceFamily) else family),
        "backend_id": backend_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "mode": mode,
        "audio_sha256": audio_sha256,
        "semantic_key": semantic_key,
        "output": dict(output),
        "confidence": confidence,
    }
    evidence_hash = contract_hash("evidence-record-v1", payload)
    return EvidenceRecord(
        evidence_id=evidence_hash,
        evidence_family=family,
        backend_id=backend_id,
        model_id=model_id,
        model_revision=model_revision,
        mode=mode,
        audio_sha256=audio_sha256,
        semantic_key=semantic_key,
        output=dict(output),
        confidence=confidence,
        evidence_hash=evidence_hash,
    )


def _raw_char_segments(value: Mapping[str, Any], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = value.get("char_segments", value.get("charSegments"))
    if isinstance(rows, list):
        return [dict(item) for item in rows if isinstance(item, Mapping)]
    rows = []
    for word in words:
        nested = word.get("chars", word.get("char_segments", word.get("charSegments", [])))
        if isinstance(nested, list):
            rows.extend(dict(item) for item in nested if isinstance(item, Mapping))
    return rows


def _character_evidence(text: str, value: Mapping[str, Any], words: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize WhisperX/MFA character output into diagnostic-only metrics."""
    expected = [char for char in str(text or "") if not char.isspace()]
    raw_rows = _raw_char_segments(value, words)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        char = str(raw.get("char", raw.get("text", raw.get("label", ""))) or "")
        if not char or char.isspace():
            continue
        start = raw.get("start")
        end = raw.get("end")
        try:
            start_value = float(start) if start is not None else None
            end_value = float(end) if end is not None else None
        except (TypeError, ValueError):
            start_value, end_value = None, None
        score_value = raw.get("score", raw.get("confidence", raw.get("probability")))
        try:
            score = max(0.0, min(1.0, float(score_value))) if score_value is not None else None
        except (TypeError, ValueError):
            score = None
        rows.append({
            "char": char,
            "start": start_value,
            "end": end_value,
            "score": score,
            "aligned": start_value is not None and end_value is not None,
            "interpolated": bool(raw.get("interpolated", raw.get("is_interpolated", False))),
            "index": index,
        })
    aligned_count = 0
    unaligned: list[str] = []
    interpolated: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, char in enumerate(expected):
        row = rows[index] if index < len(rows) else {"char": "", "start": None, "end": None, "score": None, "aligned": False, "interpolated": False}
        row = dict(row)
        row["expected_char"] = char
        if row.get("aligned"):
            aligned_count += 1
        else:
            unaligned.append(char)
        if row.get("interpolated"):
            interpolated.append(char)
        normalized_rows.append(row)
    scores = [float(row["score"]) for row in normalized_rows if row.get("score") is not None]
    scores.sort()
    coverage = (aligned_count / len(expected)) if expected else 0.0
    p10 = scores[max(0, min(len(scores) - 1, int(round((len(scores) - 1) * .10))))] if scores else None
    final_token_match = re.findall(r"[^\W\d_]+", str(text or ""), flags=re.UNICODE)
    final_chars = [char for char in (final_token_match[-1] if final_token_match else "") if not char.isspace()]
    final_rows = normalized_rows[-len(final_chars):] if final_chars else []
    final_aligned = [row for row in final_rows if row.get("aligned")]
    final_scores = [float(row["score"]) for row in final_rows if row.get("score") is not None]
    final_start = min((row["start"] for row in final_aligned), default=None)
    final_end = max((row["end"] for row in final_aligned), default=None)
    active_end = max((float(row["end"]) for row in normalized_rows if row.get("end") is not None), default=None)
    final_coverage = (len(final_aligned) / len(final_chars)) if final_chars else 0.0
    final_interpolated = any(bool(row.get("interpolated")) for row in final_rows)
    if not final_rows or final_coverage <= 0.0:
        final_status = "FINAL_ANCHOR_UNALIGNED"
    elif final_interpolated:
        final_status = "FINAL_ANCHOR_INTERPOLATED"
    elif final_coverage < 1.0 or not final_scores or min(final_scores) < .5:
        final_status = "FINAL_ANCHOR_WEAK"
    else:
        final_status = "FINAL_ANCHOR_EVIDENCE_COLLECTED"
    final_evidence = {
        "token": final_token_match[-1] if final_token_match else "",
        "expected_characters": len(final_chars),
        "aligned_characters": len(final_aligned),
        "coverage": final_coverage,
        "start": final_start,
        "end": final_end,
        "duration_ms": ((final_end - final_start) * 1000.0) if final_start is not None and final_end is not None else None,
        "mean_score": (sum(final_scores) / len(final_scores)) if final_scores else None,
        "minimum_score": min(final_scores) if final_scores else None,
        "interpolated": final_interpolated,
        "gap_to_active_speech_end_ms": ((active_end - final_end) * 1000.0) if active_end is not None and final_end is not None else None,
        "status": final_status,
        "authority": "DIAGNOSTIC_ONLY",
    }
    return {
        "char_segments": normalized_rows,
        "native_char_coverage": coverage,
        "mean_char_score": (sum(scores) / len(scores)) if scores else None,
        "minimum_char_score": min(scores) if scores else None,
        "p10_char_score": p10,
        "unaligned_characters": unaligned,
        "interpolated_characters": interpolated,
        "compression_ratio": (len(rows) / len(expected)) if expected else 0.0,
        "final_anchor_evidence": final_evidence,
    }


def _normalise_alignment(value: Mapping[str, Any], *, text: str, language: str, record: EvidenceRecord, cache_hit: bool) -> AlignmentReading:
    words = [dict(item) for item in value.get("words", value.get("word_segments", []))]
    score = float(value.get("score", value.get("alignment_score", value.get("confidence", 0.0))) or 0.0)
    coverage_value = value.get("coverage", value.get("word_coverage"))
    coverage = float(coverage_value) if coverage_value is not None else None
    final = value.get("final_anchor_present", value.get("final_word_present"))
    char_metrics = _character_evidence(text, value, words)
    return AlignmentReading(
        text, language, max(0.0, min(1.0, score)), coverage,
        bool(final) if final is not None else None, words, record, cache_hit,
        **char_metrics,
    )


def contrastive_align(
    aligner: CTCAligner,
    path: str | Path,
    *,
    target_text: str,
    source_text: str = "",
    target_language: str = "de",
    source_language: str = "en",
    cache: AlignmentCache | None = None,
    semantic_key: str | None = None,
) -> ContrastiveAlignment:
    """Align target and source hypotheses using one non-Whisper family."""
    audio_sha256 = sha256_file(path)
    family = getattr(aligner, "evidence_family", EvidenceFamily.CTC_FORCED_ALIGNER)
    backend_id = str(getattr(aligner, "backend_id", aligner.__class__.__name__))
    model_id = str(getattr(aligner, "model_id", "unknown"))
    model_revision = str(getattr(aligner, "model_revision", "unknown"))
    cache = cache or AlignmentCache(None, backend_id, model_id, model_revision)
    if cache.backend_id == "unknown":
        cache = AlignmentCache(cache.root, backend_id, model_id, model_revision)

    def one(text: str, language: str, mode: str) -> AlignmentReading:
        cached = cache.get(audio_sha256, text, language)
        cache_hit = cached is not None
        if cached is None:
            value = aligner.align(path, text=text, language=language)
            cached = dict(value)
            cache.put(audio_sha256, text, language, cached)
        output = dict(cached)
        record = _record(
            family=family,
            backend_id=backend_id,
            model_id=model_id,
            model_revision=model_revision,
            mode=mode,
            audio_sha256=audio_sha256,
            semantic_key=semantic_key,
            output=output,
            confidence=float(output.get("score", output.get("alignment_score", 0.0)) or 0.0),
        )
        return _normalise_alignment(output, text=text, language=language, record=record, cache_hit=cache_hit)

    target = one(target_text, target_language, "expected_text_alignment")
    source = one(source_text, source_language, "source_text_alignment") if source_text.strip() else None
    source_score = source.score if source is not None else None
    margin = target.score - source_score if source_score is not None else None
    return ContrastiveAlignment(target, source, target.score, source_score, margin, [item.record for item in (target, source) if item is not None])


class WhisperXCTCAligner:
    """WhisperX wav2vec2/CTC alignment adapter; never asks Whisper to transcribe."""

    evidence_family = EvidenceFamily.CTC_FORCED_ALIGNER
    backend_id = "whisperx-align"

    def __init__(self, *, model_id: str = "whisperx-default-german-ctc", model_revision: str = "unknown", device: str = "cuda") -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self._models: dict[str, tuple[Any, Any]] = {}

    def _model(self, language: str) -> tuple[Any, Any, Any]:
        language = str(language).replace("_", "-").split("-")[0].casefold()
        try:
            import whisperx
        except ImportError as exc:
            raise AlignmentUnavailable("WhisperX is not installed; install the optional alignment dependencies") from exc
        if language not in self._models:
            model, metadata = whisperx.load_align_model(language_code=language, device=self.device)
            self._models[language] = (model, metadata)
        model, metadata = self._models[language]
        return whisperx, model, metadata

    def align(self, path: str | Path, *, text: str, language: str) -> dict[str, Any]:
        whisperx, model, metadata = self._model(language)
        audio = whisperx.load_audio(str(path))
        duration = float(len(audio) / 16000.0) if len(audio) else 0.0
        # The known subtitle is the transcript input.  No Whisper decode is
        # performed here; WhisperX only aligns this text with CTC emissions.
        # WhisperX accepts an iterable of segment dictionaries, not the outer
        # transcription-result object used by Whisper's decode API.
        segments = [{"start": 0.0, "end": duration, "text": text}]
        aligned = whisperx.align(segments, model, metadata, audio, self.device, return_char_alignments=True)
        words = [dict(item) for item in aligned.get("word_segments", [])]
        expected_count = max(1, len(text.split()))
        covered = sum(1 for item in words if item.get("start") is not None and item.get("end") is not None)
        coverage = min(1.0, covered / expected_count)
        score_values = [float(item.get("score", 1.0)) for item in words if item.get("score") is not None]
        score = (sum(score_values) / len(score_values)) * coverage if score_values else coverage
        # Character-level evidence is explicitly diagnostic.  The caller must
        # use ``final_anchor_evidence.status``; no legacy boolean PASS is
        # emitted by this adapter.
        char_metrics = _character_evidence(text, aligned, words)
        return {"score": score, "coverage": coverage, "words": words, **char_metrics}


def _extract_mfa_words(value: Any) -> list[dict[str, Any]]:
    """Read MFA JSON word tiers without coupling the core to one schema."""
    if isinstance(value, dict):
        for key in ("words", "word_segments", "wordSegments"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            tiers = value.get("tiers")
            if isinstance(tiers, dict):
                for key, tier in tiers.items():
                    if "word" in str(key).casefold() and isinstance(tier, dict):
                        entries = tier.get("entries", tier.get("intervals"))
                        if isinstance(entries, list):
                            value = entries
                            break
    rows: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, dict):
            word = item.get("word", item.get("label", item.get("text", "")))
            start = item.get("start", item.get("xmin"))
            end = item.get("end", item.get("xmax"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, word = item[0], item[1], item[2]
        else:
            continue
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError):
            continue
        if not str(word or "").strip():
            continue
        rows.append({"start": start_value, "end": end_value, "word": str(word), "score": 1.0})
    return rows


class MFAAlignerAdapter:
    """MFA ``align_one`` adapter reserved for persistent difficult cases.

    MFA is intentionally never selected by the normal scheduler.  When a
    caller supplies dictionary and acoustic-model paths, this adapter invokes
    the documented single-file command and converts its JSON word tier to the
    same score/coverage contract as the CTC adapter.
    """

    evidence_family = EvidenceFamily.KALDI_FORCED_ALIGNER
    backend_id = "mfa-align-one"

    def __init__(self, *, executable: str = "mfa", model_id: str = "german_acoustic_model", model_revision: str = "unknown", dictionary_path: str | Path | None = None, acoustic_model_path: str | Path | None = None, timeout_seconds: float = 180.0) -> None:
        self.executable = executable
        self.model_id = model_id
        self.model_revision = model_revision
        self.dictionary_path = Path(dictionary_path) if dictionary_path is not None else None
        self.acoustic_model_path = Path(acoustic_model_path) if acoustic_model_path is not None else None
        self.timeout_seconds = float(timeout_seconds)

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def align(self, path: str | Path, *, text: str, language: str) -> dict[str, Any]:
        if not self.available():
            raise AlignmentUnavailable(f"MFA executable is not available: {self.executable}")
        if self.dictionary_path is None or self.acoustic_model_path is None:
            raise AlignmentUnavailable("MFA align_one requires dictionary_path and acoustic_model_path")
        if not self.dictionary_path.is_file() or not self.acoustic_model_path.exists():
            raise AlignmentUnavailable("MFA dictionary/acoustic model path does not exist")
        audio_path = Path(path)
        if not audio_path.is_file():
            raise AlignmentUnavailable(f"MFA input audio does not exist: {audio_path}")
        with tempfile.TemporaryDirectory(prefix="mfa-align-one-") as directory:
            root = Path(directory)
            text_path = root / "transcript.txt"
            output_path = root / "alignment.json"
            text_path.write_text(text.strip() + "\n", encoding="utf-8")
            command = [
                str(self.executable), "align_one", "--output_format", "json",
                str(audio_path), str(text_path), str(self.dictionary_path),
                str(self.acoustic_model_path), str(output_path),
            ]
            try:
                completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self.timeout_seconds)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AlignmentUnavailable(f"MFA align_one failed to start or timed out: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "MFA returned a non-zero status").strip()
                raise AlignmentUnavailable(f"MFA align_one failed ({completed.returncode}): {detail[-1000:]}")
            candidates = [output_path]
            candidates.extend(root.glob("*.json"))
            result_path = next((item for item in candidates if item.is_file()), None)
            if result_path is None:
                raise AlignmentUnavailable("MFA align_one did not produce a JSON alignment")
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise AlignmentUnavailable(f"MFA alignment JSON is invalid: {exc}") from exc
        words = _extract_mfa_words(raw)
        expected = max(1, len(text.split()))
        covered = sum(1 for item in words if item.get("start") is not None and item.get("end") is not None)
        final_token = _fold_word(text.split()[-1]) if text.split() else ""
        final_present = bool(words and final_token == _fold_word(words[-1].get("word", "")))
        coverage = min(1.0, covered / expected)
        return {"score": coverage, "coverage": coverage, "final_anchor_present": final_present, "words": words}


class SpeechBrainVoxLingua107:
    """Independent spoken-language ID adapter (optional SpeechBrain)."""

    evidence_family = EvidenceFamily.AUDIO_LANGUAGE_ID
    backend_id = "speechbrain-voxlingua107"

    def __init__(self, *, model_id: str = "speechbrain/lang-id-voxlingua107-ecapa", model_revision: str = "unknown", savedir: str | Path | None = None) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.savedir = str(savedir) if savedir is not None else None
        self._classifier = None

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:
            raise AlignmentUnavailable("SpeechBrain VoxLingua107 is not installed") from exc
        kwargs = {"source": self.model_id}
        if self.savedir:
            kwargs["savedir"] = self.savedir
        self._classifier = EncoderClassifier.from_hparams(**kwargs)
        return self._classifier

    def detect(self, path: str | Path) -> dict[str, Any]:
        classifier = self._load()
        output = classifier.classify_file(str(path))
        # SpeechBrain returns (scores, score, index, text_lab); retain a
        # conservative parser so this adapter remains optional and auditable.
        label_value = output[3] if isinstance(output, (tuple, list)) and len(output) > 3 else ""
        while isinstance(label_value, (tuple, list)) and label_value:
            label_value = label_value[0]
        if hasattr(label_value, "item"):
            try:
                label_value = label_value.item()
            except (TypeError, ValueError):
                pass
        label = str(label_value)
        score_value = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else 0.0
        try:
            while hasattr(score_value, "__getitem__") and not isinstance(score_value, (str, bytes)):
                score_value = score_value[0]
            if hasattr(score_value, "item"):
                score_value = score_value.item()
            confidence = float(score_value)
        except (TypeError, ValueError, IndexError):
            confidence = 0.0
        return {"language": label, "probability": max(0.0, min(1.0, confidence))}


def language_id_evidence(backend: LanguageIdentifier, path: str | Path) -> dict[str, Any]:
    """Run an independent LID family and return an evidence record."""
    audio_sha256 = sha256_file(path)
    value = dict(backend.detect(path))
    backend_id = str(getattr(backend, "backend_id", backend.__class__.__name__))
    model_id = str(getattr(backend, "model_id", "unknown"))
    model_revision = str(getattr(backend, "model_revision", "unknown"))
    record = _record(
        family=getattr(backend, "evidence_family", EvidenceFamily.AUDIO_LANGUAGE_ID),
        backend_id=backend_id,
        model_id=model_id,
        model_revision=model_revision,
        mode="spoken_language_id",
        audio_sha256=audio_sha256,
        semantic_key=None,
        output=value,
        confidence=float(value.get("probability", 0.0) or 0.0),
    )
    return {**value, "record": record.to_dict(), "evidence_families": [record.evidence_family.value]}


__all__ = [
    "AlignmentCache", "AlignmentReading", "AlignmentUnavailable", "ContrastiveAlignment",
    "CTCAligner", "LanguageIdentifier", "MFAAlignerAdapter", "SpeechBrainVoxLingua107",
    "WhisperXCTCAligner", "contrastive_align", "language_id_evidence",
]
