"""Measured QA gates and fail-closed candidate selection."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .audio import clipping, peak_dbfs, read
from .contracts import FailureClass, GateEvidence, GateStatus, gate_passes
from .hashing import canonical_json, sha256_bytes, sha256_file
from .timing import speech_end

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_CALIBRATOR_SCHEMA = "platt-calibrator-v1"
_CALIBRATOR_ENGINE = "builtin"
_CALIBRATOR_FORMAT = "json"
_FEATURE_SCHEMA_VERSION = "char-alignment-v2"
_NORMALIZATION_VERSION = "alignment-text-normalization-v2"
_CALIBRATOR_FEATURES = (
    "target_score",
    "native_char_coverage",
    "mean_char_score",
    "minimum_char_score",
    "p10_char_score",
    "delete_ratio",
    "substitute_ratio",
    "insert_ratio",
    "interpolated_ratio",
    "compression_ratio",
    "characters_per_second",
    "words_per_second",
    "duration",
    "performance_mode",
)
_FINAL_ANCHOR_FEATURES = (
    "final_coverage", "final_minimum_score", "final_mean_score", "final_duration",
    "gap_to_active_speech_end_ms", "final_delete_count", "final_substitute_count",
    "insertions_inside_anchor", "final_interpolated",
)
_LID_FEATURE_SCHEMA_VERSION = "lid-fusion-v1"
_LID_FEATURES = ("lid_source_probability", "lid_target_probability", "whisper_source_probability", "ctc_target_probability", "duration_seconds", "speech_ratio", "performance_mode")
_PERFORMANCE_MODE_CODES = {
    "NEUTRAL": 0.0,
    "FAST": 1.0,
    "WHISPER": 2.0,
    "SHOUT": 3.0,
    "SCREAM_SPEECH": 4.0,
    "CRYING_SPEECH": 5.0,
    "EFFORT": 6.0,
    "LAUGH_SPEECH": 7.0,
}


def fold(text: str) -> str:
    """Fold only case and presentation variants; retain German contrasts."""
    value = unicodedata.normalize("NFC", str(text or "").lower()).replace("\u2019", "'")
    return value


def tokens(text: str) -> list[str]:
    return [fold(item) for item in _TOKEN.findall(str(text or ""))]


def _language_code(value: Any) -> str:
    return str(value or "").casefold().replace("_", "-").split(":", 1)[0].split("-", 1)[0].strip()


@dataclass(frozen=True)
class LanguageProfile:
    source_language: str = "en"
    target_language: str = "de"
    source_markers: tuple[str, ...] = ("the", "you", "what", "why", "yes", "no", "not", "are", "is", "can", "will", "this", "that", "your", "to", "of", "and", "nearby", "enemies", "detected")
    strong_source_words: tuple[str, ...] = ()
    neutral_efforts: tuple[str, ...] = ("ugh", "geez", "ah", "oh", "hmm", "huh", "oof", "ow", "argh", "gah", "tsk")


@dataclass
class QAResultV2:
    passed: bool
    gates: dict[str, GateEvidence]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_class: FailureClass | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "gates": {key: value.to_dict() for key, value in self.gates.items()}, "diagnostics": self.diagnostics, "failure_class": self.failure_class.value if self.failure_class else None}


LinguisticStatus = Literal[
    "PASS_SCREENED",
    "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT",
    "PASS_CONFIRMED",
    "PASS_PHONETIC",
    "TARGET_ALIGNMENT_SUPPORT",
    "TARGET_ALIGNMENT_WEAK",
    "EVIDENCE_CONFLICT",
    "ASR_UNCERTAIN",
    "ALIGNMENT_UNCERTAIN",
    "LANGUAGE_LEAK_CONFIRMED",
    "LANGUAGE_LEAK_SUSPECTED",
    "LANGUAGE_LEAK_STRONG_SUSPICION",
    "LEXICAL_FAILURE_CONFIRMED",
    "LEXICAL_FAILURE_SUSPECTED",
    "ALIGNER_NOT_APPLICABLE",
    "PERFORMANCE_UNCERTAIN",
    "HUMAN_REVIEW",
    "BLOCKED",
    "NOT_APPLICABLE",
]


@dataclass(frozen=True)
class LinguisticDecision:
    """Structured linguistic verdict; never collapse evidence to one boolean."""

    status: LinguisticStatus
    expected_alignment_score: float | None = None
    source_alignment_score: float | None = None
    alignment_margin: float | None = None
    cross_language_margin: float | None = None
    forced_transcript: str | None = None
    automatic_transcript: str | None = None
    detected_language: str | None = None
    language_probability: float | None = None
    word_coverage: float | None = None
    phone_coverage: float | None = None
    final_anchor_present: bool | None = None
    final_anchor_evidence: dict[str, Any] | None = None
    char_segments: list[dict[str, Any]] = field(default_factory=list)
    native_char_coverage: float | None = None
    mean_char_score: float | None = None
    minimum_char_score: float | None = None
    p10_char_score: float | None = None
    unaligned_characters: list[str] = field(default_factory=list)
    interpolated_characters: list[str] = field(default_factory=list)
    delete_count: int = 0
    insert_count: int = 0
    substitute_count: int = 0
    interpolated_count: int = 0
    alignment_operation_hash: str | None = None
    compression_ratio: float | None = None
    duration: float | None = None
    characters_per_second: float | None = None
    words_per_second: float | None = None
    normalization_version: str = _NORMALIZATION_VERSION
    missing_tokens: list[str] = field(default_factory=list)
    expected_tokens: list[str] = field(default_factory=list)
    forced_final_tokens: list[str] = field(default_factory=list)
    automatic_final_tokens: list[str] = field(default_factory=list)
    evidence_hashes: list[str] = field(default_factory=list)
    evidence_families: list[str] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    audio_sha256: str | None = None
    screened: bool = False
    confirmed: bool = False
    calibration_authority: bool = False
    calibration_profile_status: str = "DISABLED"
    raw_target_score: float | None = None
    calibrated_target_probability: float | None = None
    raw_final_anchor_score: float | None = None
    calibrated_final_anchor_probability: float | None = None
    feature_vector: dict[str, float] | None = None
    feature_vector_hash: str | None = None
    final_anchor_feature_vector: dict[str, float] | None = None
    final_anchor_feature_vector_hash: str | None = None
    calibrator_hash: str | None = None
    calibrator_artifact_sha256: str | None = None
    final_anchor_calibrator_hash: str | None = None
    final_anchor_calibrator_artifact_sha256: str | None = None
    calibrated_lid_probability: float | None = None
    lid_feature_vector: dict[str, float] | None = None
    lid_feature_vector_hash: str | None = None
    lid_calibrator_hash: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_alignment_score": self.expected_alignment_score,
            "source_alignment_score": self.source_alignment_score,
            "alignment_margin": self.alignment_margin,
            "cross_language_margin": self.cross_language_margin,
            "forced_transcript": self.forced_transcript,
            "automatic_transcript": self.automatic_transcript,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "word_coverage": self.word_coverage,
            "phone_coverage": self.phone_coverage,
            "final_anchor_present": self.final_anchor_present,
            "final_anchor_evidence": dict(self.final_anchor_evidence) if self.final_anchor_evidence is not None else None,
            "char_segments": [dict(item) for item in self.char_segments],
            "native_char_coverage": self.native_char_coverage,
            "mean_char_score": self.mean_char_score,
            "minimum_char_score": self.minimum_char_score,
            "p10_char_score": self.p10_char_score,
            "unaligned_characters": list(self.unaligned_characters),
            "interpolated_characters": list(self.interpolated_characters),
            "delete_count": self.delete_count,
            "insert_count": self.insert_count,
            "substitute_count": self.substitute_count,
            "interpolated_count": self.interpolated_count,
            "alignment_operation_hash": self.alignment_operation_hash,
            "compression_ratio": self.compression_ratio,
            "duration": self.duration,
            "characters_per_second": self.characters_per_second,
            "words_per_second": self.words_per_second,
            "normalization_version": self.normalization_version,
            "missing_tokens": list(self.missing_tokens),
            "expected_tokens": list(self.expected_tokens),
            "forced_final_tokens": list(self.forced_final_tokens),
            "automatic_final_tokens": list(self.automatic_final_tokens),
            "evidence_hashes": list(self.evidence_hashes),
            "evidence_families": list(self.evidence_families),
            "evidence_records": list(self.evidence_records),
            "audio_sha256": self.audio_sha256,
            "screened": self.screened,
            "confirmed": self.confirmed,
            "calibration_authority": self.calibration_authority,
            "calibration_profile_status": self.calibration_profile_status,
            "raw_target_score": self.raw_target_score,
            "calibrated_target_probability": self.calibrated_target_probability,
            "raw_final_anchor_score": self.raw_final_anchor_score,
            "calibrated_final_anchor_probability": self.calibrated_final_anchor_probability,
            "feature_vector": dict(self.feature_vector) if self.feature_vector is not None else None,
            "feature_vector_hash": self.feature_vector_hash,
            "final_anchor_feature_vector": dict(self.final_anchor_feature_vector) if self.final_anchor_feature_vector is not None else None,
            "final_anchor_feature_vector_hash": self.final_anchor_feature_vector_hash,
            "calibrator_hash": self.calibrator_hash,
            "calibrator_artifact_sha256": self.calibrator_artifact_sha256,
            "final_anchor_calibrator_hash": self.final_anchor_calibrator_hash,
            "final_anchor_calibrator_artifact_sha256": self.final_anchor_calibrator_artifact_sha256,
            "calibrated_lid_probability": self.calibrated_lid_probability,
            "lid_feature_vector": dict(self.lid_feature_vector) if self.lid_feature_vector is not None else None,
            "lid_feature_vector_hash": self.lid_feature_vector_hash,
            "lid_calibrator_hash": self.lid_calibrator_hash,
            "reason": self.reason,
        }


def _gate(name: str, status: GateStatus, measured: Any = None, threshold: Any = None, units: str | None = None, details: Mapping[str, Any] | None = None, evidence_hash: str | None = None) -> GateEvidence:
    return GateEvidence(name, status, measured_value=measured, threshold=threshold, units=units, evidence_hash=evidence_hash, details=dict(details or {}))


def ordered_content(expected: str, actual: str) -> tuple[bool, dict[str, Any]]:
    want, got = tokens(expected), tokens(actual)
    cursor = 0
    matched: list[str] = []
    reorder = False
    for token in got:
        if cursor < len(want) and token == want[cursor]:
            matched.append(token); cursor += 1
        elif token in want:
            reorder = True
    missing = want[cursor:]
    # A generated sentence may contain a harmless filler, but an out-of-order
    # transcript or a strong insertion is not equivalent to the subtitle.
    passed = bool(want) and not missing and not reorder and len(got) <= len(want) + 2
    return passed, {"expected_tokens": want, "heard_tokens": got, "matched_in_order": matched, "missing_tokens": missing, "reordered_expected_tokens": reorder, "extra_tokens": max(0, len(got) - len(want))}


def final_word(target: str, transcript: str, min_tokens: int = 1) -> tuple[bool, dict[str, Any]]:
    expected, actual = tokens(target), tokens(transcript)
    count = max(1, int(min_tokens))
    required = expected[-count:] if expected else []
    heard = actual[-count:] if actual else []
    passed = bool(required) and len(actual) >= len(required) and heard == required
    return passed, {"expected_final_tokens": required, "heard_final_tokens": heard, "actual_tokens": actual}


def source_language_leak(source_text: str, transcript: str, language: str | None, probability: float | None, profile: LanguageProfile) -> tuple[bool, dict[str, Any]]:
    heard = set(tokens(transcript)); source = set(tokens(source_text))
    marker_hits = sorted(heard.intersection(tokens(" ".join(profile.source_markers))))
    strong_hits = sorted(heard.intersection(tokens(" ".join(profile.strong_source_words))))
    source_overlap = sorted(heard.intersection(source))
    probable = _language_code(language) == _language_code(profile.source_language) and float(probability or 0.0) >= .70
    source_ratio = (len(source_overlap) / len(source)) if source else 0.0
    likely = bool(strong_hits) or (probable and (len(marker_hits) >= 2 or (len(source_overlap) >= 2 and source_ratio >= .60)))
    return not likely, {"language": language, "probability": probability, "marker_hits": marker_hits, "strong_source_hits": strong_hits, "source_overlap": source_overlap, "source_overlap_ratio": source_ratio}


def _coerce_reading(value: Mapping[str, Any] | None, *, default_mode: str) -> dict[str, Any]:
    value = dict(value or {})
    return {
        "mode": str(value.get("mode", default_mode)),
        "text": str(value.get("text", "")),
        "language": value.get("language"),
        "probability": float(value["probability"]) if value.get("probability") is not None else None,
        "evidence_hash": str(value.get("evidence_hash", "")),
    }


def decide_linguistic_evidence(
    expected_text: str,
    source_text: str,
    *,
    forced_target: Mapping[str, Any],
    automatic: Mapping[str, Any],
    target_language: str,
    profile: LanguageProfile,
    final_word_min_tokens: int = 1,
    evidence_hashes: Sequence[str] = (),
    evidence_records: Sequence[Mapping[str, Any]] = (),
    audio_sha256: str | None = None,
) -> LinguisticDecision:
    """Screen with correlated Whisper readings; never call them independent."""
    forced = _coerce_reading(forced_target, default_mode="forced_target")
    automatic_row = _coerce_reading(automatic, default_mode="automatic")
    forced_text = forced["text"]
    automatic_text = automatic_row["text"]
    forced_content_ok, forced_content = ordered_content(expected_text, forced_text)
    automatic_content_ok, automatic_content = ordered_content(expected_text, automatic_text)
    forced_final_ok, forced_final = final_word(expected_text, forced_text, final_word_min_tokens)
    automatic_final_ok, automatic_final = final_word(expected_text, automatic_text, final_word_min_tokens)
    leak_ok, leak_details = source_language_leak(
        source_text,
        automatic_text,
        automatic_row["language"],
        automatic_row["probability"],
        profile,
    )
    detected = automatic_row["language"]
    language_known = detected is not None
    automatic_target = _language_code(detected) == _language_code(target_language)
    target_confirmed = language_known and automatic_target and float(automatic_row["probability"] or 0.0) >= .70
    # A backend without language metadata is not allowed to create a new
    # hard PASS under the dual policy.  Legacy single-transcript callers keep
    # their old behavior because this helper is only used with dual evidence.
    if leak_details.get("strong_source_hits") or (not leak_ok and detected and not automatic_target):
        status: LinguisticStatus = "LANGUAGE_LEAK_SUSPECTED"
        reason = "automatic Whisper evidence suggests source-language speech; independent LID/alignment required"
    elif forced_content_ok and forced_final_ok and automatic_content_ok and automatic_final_ok and target_confirmed:
        status = "PASS_SCREENED"
        reason = "forced and automatic Whisper readings agree; CTC confirmation still required"
    elif forced_content_ok and forced_final_ok and automatic_target and (not automatic_content_ok or not automatic_final_ok):
        status = "ASR_UNCERTAIN"
        reason = "forced target decode passes but automatic decode disagrees"
    elif automatic_content_ok and automatic_final_ok and (not forced_content_ok or not forced_final_ok):
        status = "ASR_UNCERTAIN"
        reason = "automatic decode passes but forced target decode disagrees"
    else:
        status = "ASR_UNCERTAIN"
        reason = "correlated Whisper evidence is insufficient for a definitive verdict"

    expected = tokens(expected_text)
    matched = len(forced_content.get("matched_in_order", []))
    coverage = (matched / len(expected)) if expected else None
    final_anchor = bool(forced_final_ok and automatic_final_ok)
    return LinguisticDecision(
        status=status,
        forced_transcript=forced_text,
        automatic_transcript=automatic_text,
        detected_language=detected,
        language_probability=automatic_row["probability"],
        word_coverage=coverage,
        final_anchor_present=final_anchor,
        missing_tokens=list(forced_content.get("missing_tokens", [])),
        expected_tokens=expected,
        forced_final_tokens=list(forced_final.get("heard_final_tokens", [])),
        automatic_final_tokens=list(automatic_final.get("heard_final_tokens", [])),
        evidence_hashes=list(evidence_hashes),
        evidence_families=["WHISPER_ASR"],
        evidence_records=[dict(item) for item in evidence_records],
        audio_sha256=audio_sha256,
        screened=status == "PASS_SCREENED",
        confirmed=False,
        reason=reason,
    )


def _resolve_calibrator_path(
    calibrator: Mapping[str, Any],
    profile: Mapping[str, Any],
    calibrator_root: str | Path | None,
) -> Path | None:
    """Resolve a profile artifact without allowing an implicit CWD fallback."""
    raw_path = str(calibrator.get("artifact_path", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    root_value = calibrator_root or profile.get("profile_root")
    if root_value in (None, ""):
        return None
    return Path(root_value) / path


def load_safe_calibrator(
    path: str | Path,
    expected_sha256: str,
    expected_feature_schema: str = _FEATURE_SCHEMA_VERSION,
    expected_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load the only production calibration format: deterministic JSON."""
    artifact_path = Path(path)
    expected_hash = str(expected_sha256 or "").casefold()
    if not artifact_path.is_file() or not _SHA256.fullmatch(expected_hash) or sha256_file(artifact_path).casefold() != expected_hash:
        raise ValueError("calibrator artifact is missing or has a mismatched SHA-256")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("calibrator artifact is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("calibrator artifact must contain a JSON object")
    if payload.get("schema") != _CALIBRATOR_SCHEMA or payload.get("feature_schema_version") != expected_feature_schema or payload.get("normalization_version") != _NORMALIZATION_VERSION:
        raise ValueError("calibrator schema or normalization version is incompatible")
    expected_feature_names = list(expected_features or _CALIBRATOR_FEATURES)
    if list(payload.get("features") or ()) != expected_feature_names:
        raise ValueError("calibrator feature order is incompatible")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != len(expected_feature_names):
        raise ValueError("calibrator coefficient length is incompatible")
    try:
        intercept = float(payload["intercept"])
        values = [float(value) for value in coefficients]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibrator coefficients are not numeric") from exc
    normalization = payload.get("normalization")
    if not isinstance(normalization, list) or len(normalization) != len(expected_feature_names):
        raise ValueError("calibrator normalization length is incompatible")
    normalized: list[dict[str, float]] = []
    for item in normalization:
        if not isinstance(item, Mapping):
            raise ValueError("calibrator normalization entry is invalid")
        try:
            mean, scale = float(item["mean"]), float(item["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibrator normalization is not numeric") from exc
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("calibrator normalization must be finite with positive scales")
        normalized.append({"mean": mean, "scale": scale})
    if not math.isfinite(intercept) or any(not math.isfinite(value) for value in values):
        raise ValueError("calibrator coefficients must be finite")
    return {
        "schema": _CALIBRATOR_SCHEMA,
        "feature_schema_version": expected_feature_schema,
        "normalization_version": _NORMALIZATION_VERSION,
        "features": expected_feature_names,
        "coefficients": values,
        "intercept": intercept,
        "normalization": normalized,
    }


def predict_probability(calibrator: Mapping[str, Any], feature_vector: Mapping[str, Any]) -> float:
    """Evaluate a safe Platt calibrator; never coerce missing data to a score."""
    if (
        calibrator.get("schema") != _CALIBRATOR_SCHEMA
        or calibrator.get("feature_schema_version") not in {_FEATURE_SCHEMA_VERSION, "final-anchor-v1", "lid-fusion-v1"}
        or calibrator.get("normalization_version") != _NORMALIZATION_VERSION
        or not list(calibrator.get("features") or ())
        or len(list(calibrator.get("features") or ())) != len(list(calibrator.get("coefficients") or ()))
    ):
        raise ValueError("unsupported calibrator schema")
    try:
        features = list(calibrator["features"])
        normalization = list(calibrator["normalization"])
        if len(normalization) != len(features):
            raise ValueError("calibrator normalization is incomplete")
        logit = float(calibrator["intercept"])
        for name, coefficient, normalization_item in zip(features, calibrator["coefficients"], normalization):
            value = float(feature_vector[name])
            coefficient_value = float(coefficient)
            mean = float(normalization_item["mean"])
            scale = float(normalization_item["scale"])
            if not math.isfinite(value) or not math.isfinite(coefficient_value) or not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("calibration features and coefficients must be finite")
            logit += coefficient_value * ((value - mean) / scale)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration feature vector is incomplete or invalid") from exc
    if not math.isfinite(logit):
        raise ValueError("calibrated logit is not finite")
    if logit >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probability = exp_logit / (1.0 + exp_logit)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("calibrated probability is invalid")
    return probability


def _load_calibrator_artifact(
    profile: Mapping[str, Any],
    *,
    role: str = "target",
    calibrator_root: str | Path | None,
    feature_schema_version: str,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    """Load only the safe built-in JSON calibrator format.

    Arbitrary pickle/Python artifacts are deliberately unsupported.  A profile
    can therefore never obtain authority merely by pointing at a hashed file;
    the runtime must parse and execute this exact versioned schema.
    """
    calibrator_set = profile.get("calibrators")
    calibrator = calibrator_set.get(role) if isinstance(calibrator_set, Mapping) else None
    # Legacy profiles are diagnostic-only after the multi-artifact contract;
    # accepting their target artifact here keeps old reports readable while
    # authority validation below rejects them.
    if calibrator is None and role == "target" and not isinstance(calibrator_set, Mapping):
        calibrator = profile.get("calibrator")
    if not isinstance(calibrator, Mapping):
        return None, None, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    if str(calibrator.get("type", "")) != "platt":
        return None, None, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    if str(calibrator.get("engine", "")) != _CALIBRATOR_ENGINE:
        return None, None, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    if str(calibrator.get("format", "")) != _CALIBRATOR_FORMAT:
        return None, None, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    expected_artifact_schema = "final-anchor-v1" if role == "final_anchor" else (_LID_FEATURE_SCHEMA_VERSION if role == "lid" else str(feature_schema_version))
    if str(calibrator.get("feature_schema_version", "")) != expected_artifact_schema or str(calibrator.get("normalization_version", "")) != _NORMALIZATION_VERSION:
        return None, None, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    path = _resolve_calibrator_path(calibrator, profile, calibrator_root)
    expected_hash = str(calibrator.get("artifact_sha256", "")).casefold()
    if path is None or not path.is_file() or not _SHA256.fullmatch(expected_hash):
        return None, path, "BLOCKED_CALIBRATOR_ARTIFACT"
    try:
        expected_features = _CALIBRATOR_FEATURES if role == "target" else (_FINAL_ANCHOR_FEATURES if role == "final_anchor" else _LID_FEATURES)
        expected_artifact_schema = str(feature_schema_version) if role == "target" else ("final-anchor-v1" if role == "final_anchor" else _LID_FEATURE_SCHEMA_VERSION)
        payload = load_safe_calibrator(path, expected_hash, expected_artifact_schema, expected_features)
    except ValueError:
        return None, path, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    return payload, path, None


def _alignment_feature_vector(
    alignment_target: Mapping[str, Any],
    *,
    target_score: float,
    performance_mode: str | None,
) -> dict[str, float] | None:
    """Build the exact feature contract consumed by the calibrator."""
    expected_count = len([row for row in (alignment_target.get("char_segments") or []) if isinstance(row, Mapping) and row.get("expected_index") is not None])
    denominator = float(max(1, expected_count))
    values: dict[str, Any] = {
        "target_score": target_score,
        "native_char_coverage": alignment_target.get("native_char_coverage"),
        "mean_char_score": alignment_target.get("mean_char_score"),
        "minimum_char_score": alignment_target.get("minimum_char_score"),
        "p10_char_score": alignment_target.get("p10_char_score"),
        "delete_ratio": float(alignment_target.get("delete_count", 0) or 0) / denominator,
        "substitute_ratio": float(alignment_target.get("substitute_count", 0) or 0) / denominator,
        "insert_ratio": float(alignment_target.get("insert_count", 0) or 0) / denominator,
        "interpolated_ratio": float(alignment_target.get("interpolated_count", 0) or 0) / denominator,
        "compression_ratio": alignment_target.get("compression_ratio"),
        "characters_per_second": alignment_target.get("characters_per_second"),
        "words_per_second": alignment_target.get("words_per_second"),
        "duration": alignment_target.get("duration"),
        "performance_mode": _PERFORMANCE_MODE_CODES.get(str(performance_mode or "NEUTRAL").upper()),
    }
    try:
        result = {key: float(values[key]) for key in _CALIBRATOR_FEATURES}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in result.values()):
        return None
    return result


def _final_anchor_feature_vector(
    alignment_target: Mapping[str, Any],
    *,
    target_score: float,
    performance_mode: str | None,
) -> dict[str, float] | None:
    evidence = alignment_target.get("final_anchor_evidence")
    if not isinstance(evidence, Mapping):
        return None
    expected_count = max(1, int(evidence.get("expected_characters", 0) or 0))
    duration = (float(evidence["duration_ms"]) / 1000.0) if evidence.get("duration_ms") is not None else None
    values: dict[str, Any] = {
        "final_coverage": evidence.get("coverage"),
        "final_minimum_score": evidence.get("minimum_score"),
        "final_mean_score": evidence.get("mean_score"),
        "final_duration": duration,
        "gap_to_active_speech_end_ms": evidence.get("gap_to_active_speech_end_ms"),
        "final_delete_count": evidence.get("deleted_characters", 0),
        "final_substitute_count": evidence.get("substituted_characters", 0),
        "insertions_inside_anchor": evidence.get("insertions_inside_anchor", 0),
        "final_interpolated": 1.0 if evidence.get("interpolated") else 0.0,
    }
    try:
        result = {key: float(values[key]) for key in _FINAL_ANCHOR_FEATURES}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in result.values()):
        return None
    return result


def _feature_vector_hash(features: Mapping[str, float], schema: str = _FEATURE_SCHEMA_VERSION) -> str:
    return sha256_bytes(canonical_json({"schema": schema, "normalization_version": _NORMALIZATION_VERSION, "features": dict(features)}))


def _execute_platt_calibrator(
    artifact: Mapping[str, Any],
    features: Mapping[str, float],
) -> float | None:
    try:
        return predict_probability(artifact, features)
    except ValueError:
        return None


def calibration_profile_status(
    profile: Mapping[str, Any] | None,
    *,
    authority: bool,
    model_id: str | None,
    model_revision: str | None,
    backend_id: str | None = None,
    target_language: str,
    source_language: str,
    performance_mode: str | None = None,
    feature_schema_version: str = _FEATURE_SCHEMA_VERSION,
    calibrator_root: str | Path | None = None,
    runtime_lock_sha256: str | None = None,
    models_lock_sha256: str | None = None,
    expected_code_commit: str | None = None,
    require_promotion_receipt: bool = False,
) -> str:
    """Validate the complete artifact that is allowed to grant QA authority."""
    if not authority:
        return "DISABLED"
    if not isinstance(profile, Mapping):
        return "BLOCKED_INCOMPLETE_PROFILE"
    if str(profile.get("schema", "")) != "generic-dubbing-alignment-calibration-profile-v2":
        return "BLOCKED_SCHEMA"
    if str(profile.get("status", "")) != "VALIDATED":
        return "BLOCKED_PROFILE_STATUS"
    if profile.get("authority", profile.get("calibration_authority")) is not True:
        return "BLOCKED_PROFILE_NOT_AUTHORIZED"
    if not str(profile.get("profile_id", "")).strip():
        return "BLOCKED_INCOMPLETE_PROFILE"
    identity = profile.get("identity")
    thresholds = profile.get("thresholds")
    calibrators = profile.get("calibrators")
    if not isinstance(calibrators, Mapping) or set(calibrators) != {"target", "final_anchor", "lid"}:
        return "BLOCKED_SCHEMA"
    if "language_id" in calibrators:
        return "BLOCKED_SCHEMA"
    calibrator = calibrators.get("target") if isinstance(calibrators, Mapping) else None
    final_anchor_calibrator = calibrators.get("final_anchor") if isinstance(calibrators, Mapping) else None
    dataset = profile.get("dataset")
    metrics = profile.get("metrics")
    provenance = profile.get("provenance")
    if not all(isinstance(item, Mapping) for item in (identity, thresholds, calibrator, final_anchor_calibrator, dataset, metrics, provenance)):
        return "BLOCKED_INCOMPLETE_PROFILE"
    required_identity = {"backend_id", "model_id", "model_revision", "feature_schema_version", "target_language", "source_language", "performance_modes"}
    if not required_identity.issubset(identity) or not isinstance(identity.get("performance_modes"), (list, tuple, set)) or not identity.get("performance_modes"):
        return "BLOCKED_INCOMPLETE_PROFILE"
    expected_identity = {
        "backend_id": backend_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "feature_schema_version": feature_schema_version,
        "target_language": target_language,
        "source_language": source_language,
    }
    for key, value in expected_identity.items():
        if value in (None, "") or str(identity.get(key, "")) != str(value):
            return "BLOCKED_IDENTITY_MISMATCH" if key != "feature_schema_version" else "BLOCKED_FEATURE_SCHEMA"
    if str(performance_mode or "") not in {str(item) for item in identity["performance_modes"]}:
        return "BLOCKED_MODE_MISMATCH"
    threshold_keys = {"target_pass_probability", "target_failure_probability", "final_anchor_pass_probability", "source_lid_probability"}
    if not threshold_keys.issubset(thresholds):
        return "BLOCKED_INCOMPLETE_PROFILE"
    try:
        threshold_values = {key: float(thresholds[key]) for key in threshold_keys}
    except (TypeError, ValueError):
        return "BLOCKED_INCOMPLETE_PROFILE"
    if any(value < 0.0 or value > 1.0 for value in threshold_values.values()) or threshold_values["target_failure_probability"] >= threshold_values["target_pass_probability"]:
        return "BLOCKED_INVALID_THRESHOLDS"
    for role in ("target", "final_anchor", "lid"):
        spec = calibrators.get(role) if isinstance(calibrators, Mapping) else None
        if not isinstance(spec, Mapping):
            return "BLOCKED_CALIBRATOR_SET"
        if not all(str(spec.get(key, "")).strip() for key in ("type", "artifact_path", "artifact_sha256")) or not _SHA256.fullmatch(str(spec.get("artifact_sha256", ""))):
            return "BLOCKED_INCOMPLETE_PROFILE"
        artifact_path = _resolve_calibrator_path(spec, profile, calibrator_root)
        if artifact_path is None or not artifact_path.is_file():
            return "BLOCKED_CALIBRATOR_ARTIFACT"
        try:
            if sha256_file(artifact_path).casefold() != str(spec["artifact_sha256"]).casefold():
                return "BLOCKED_CALIBRATOR_HASH"
        except OSError:
            return "BLOCKED_CALIBRATOR_ARTIFACT"
        expected_schema = "final-anchor-v1" if role == "final_anchor" else (_LID_FEATURE_SCHEMA_VERSION if role == "lid" else feature_schema_version)
        if str(spec.get("feature_schema_version", "")) != expected_schema:
            return "BLOCKED_CALIBRATOR_SCHEMA"
        if not all(str(spec.get(key, "")).strip() for key in ("engine", "format", "feature_schema_version", "normalization_version")):
            return "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
        loaded, _loaded_path, executable_status = _load_calibrator_artifact(
            profile,
            role=role,
            calibrator_root=calibrator_root,
            feature_schema_version=feature_schema_version,
        )
        if executable_status:
            return executable_status
        if loaded is None:
            return "BLOCKED_CALIBRATOR_NOT_EXECUTABLE"
    dataset_hashes = {"manifest_sha256", "labels_sha256", "split_manifest_sha256"}
    if not dataset_hashes.issubset(dataset) or any(not _SHA256.fullmatch(str(dataset.get(key, ""))) for key in dataset_hashes):
        return "BLOCKED_INCOMPLETE_PROFILE"
    for key in ("calibration_count", "validation_count", "hidden_test_count"):
        if not isinstance(dataset.get(key), int) or dataset[key] <= 0:
            return "BLOCKED_INCOMPLETE_PROFILE"
    metric_keys = {"hidden_false_pass_count", "hidden_false_fail_count", "brier_score", "expected_calibration_error"}
    if not metric_keys.issubset(metrics):
        return "BLOCKED_INCOMPLETE_PROFILE"
    try:
        if any(float(metrics[key]) < 0.0 for key in ("hidden_false_pass_count", "hidden_false_fail_count", "brier_score", "expected_calibration_error")):
            return "BLOCKED_INCOMPLETE_PROFILE"
    except (TypeError, ValueError):
        return "BLOCKED_INCOMPLETE_PROFILE"
    provenance_keys = {"code_commit", "runtime_lock_sha256", "models_lock_sha256", "created_at"}
    if not provenance_keys.issubset(provenance) or not all(str(provenance.get(key, "")).strip() for key in provenance_keys):
        return "BLOCKED_INCOMPLETE_PROFILE"
    for key in ("runtime_lock_sha256", "models_lock_sha256"):
        if not _SHA256.fullmatch(str(provenance.get(key, ""))):
            return "BLOCKED_INCOMPLETE_PROFILE"
    if runtime_lock_sha256 is None or models_lock_sha256 is None:
        return "BLOCKED_RUNTIME_LOCK_UNAVAILABLE"
    if str(provenance["runtime_lock_sha256"]).casefold() != str(runtime_lock_sha256).casefold() or str(provenance["models_lock_sha256"]).casefold() != str(models_lock_sha256).casefold():
        return "BLOCKED_RUNTIME_MODEL_MISMATCH"
    if expected_code_commit is not None and str(provenance.get("code_commit", "")).casefold() != str(expected_code_commit).casefold():
        return "BLOCKED_CODE_COMMIT_MISMATCH"
    receipt_sha = provenance.get("promotion_receipt_sha256")
    receipt_path_value = provenance.get("promotion_receipt_path")
    if require_promotion_receipt and (not receipt_sha or not receipt_path_value):
        return "BLOCKED_PROMOTION_RECEIPT"
    if receipt_sha or receipt_path_value:
        if not isinstance(receipt_sha, str) or not _SHA256.fullmatch(receipt_sha):
            return "BLOCKED_PROMOTION_RECEIPT"
        receipt_path = Path(str(receipt_path_value))
        if not receipt_path.is_absolute() and calibrator_root is not None:
            receipt_path = Path(calibrator_root) / receipt_path
        if not receipt_path.is_file():
            return "BLOCKED_PROMOTION_RECEIPT"
        try:
            receipt_bytes = receipt_path.read_bytes()
            if sha256_bytes(receipt_bytes).casefold() != receipt_sha.casefold():
                return "BLOCKED_PROMOTION_RECEIPT"
            receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return "BLOCKED_PROMOTION_RECEIPT"
        if not isinstance(receipt, Mapping) or receipt.get("schema") != "dubproj-promotion-receipt-v1" or str(receipt.get("profile_id", "")) != str(profile.get("profile_id", "")) or str(receipt.get("code_commit", "")).casefold() != str(provenance.get("code_commit", "")).casefold():
            return "BLOCKED_PROMOTION_RECEIPT"
        receipt_artifacts = receipt.get("artifact_sha256")
        if not isinstance(receipt_artifacts, Mapping) or any(str(receipt_artifacts.get(role, "")).casefold() != str(calibrators[role].get("artifact_sha256", "")).casefold() for role in ("target", "final_anchor", "lid")):
            return "BLOCKED_PROMOTION_RECEIPT"
        receipt_locks = receipt.get("lock_sha256")
        if not isinstance(receipt_locks, Mapping) or str(receipt_locks.get("runtime", "")).casefold() != str(provenance.get("runtime_lock_sha256", "")).casefold() or str(receipt_locks.get("models", "")).casefold() != str(provenance.get("models_lock_sha256", "")).casefold():
            return "BLOCKED_PROMOTION_RECEIPT"
    return "MATCHED_VALIDATED"


def _validated_profile_threshold(profile: Mapping[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((profile or {}).get("thresholds", {}).get(key, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _final_anchor_is_calibrated(
    decision: LinguisticDecision,
    *,
    profile: Mapping[str, Any] | None,
    feature_schema_version: str,
) -> bool:
    """Require calibrated final-anchor evidence for hard authority."""
    evidence = decision.final_anchor_evidence
    if feature_schema_version == _FEATURE_SCHEMA_VERSION:
        if not isinstance(evidence, Mapping):
            return False
        if bool(evidence.get("interpolated")):
            return False
        threshold = _validated_profile_threshold(profile, "final_anchor_pass_probability", 1.0)
        try:
            probability = decision.calibrated_final_anchor_probability
            return (
                probability is not None
                and float(probability) >= threshold
                and evidence.get("status") == "FINAL_ANCHOR_EVIDENCE_COLLECTED"
                and evidence.get("timing_valid") is True
                and float(evidence.get("duration_ms") or 0.0) > 0.0
                and int(evidence.get("substituted_characters", 0) or 0) == 0
                and int(evidence.get("deleted_characters", 0) or 0) == 0
                and int(evidence.get("insertions_inside_anchor", 0) or 0) == 0
            )
        except (TypeError, ValueError):
            return False
    return decision.final_anchor_present is True


def _lid_feature_vector(lid_evidence: Mapping[str, Any] | None, *, whisper_probability: float | None, ctc_target_probability: float | None, performance_mode: str | None) -> dict[str, float] | None:
    if not isinstance(lid_evidence, Mapping):
        return None
    probabilities = lid_evidence.get("probabilities") if isinstance(lid_evidence.get("probabilities"), Mapping) else {}
    try:
        values = {
            "lid_source_probability": float(lid_evidence.get("source_probability", probabilities.get("en", 0.0))),
            "lid_target_probability": float(lid_evidence.get("target_probability", probabilities.get("de", 0.0))),
            "whisper_source_probability": float(whisper_probability or 0.0),
            "ctc_target_probability": float(ctc_target_probability if ctc_target_probability is not None else 0.0),
            "duration_seconds": float(lid_evidence.get("duration_seconds", lid_evidence.get("duration", 0.0))),
            "speech_ratio": float(lid_evidence.get("speech_ratio", 0.0)),
            "performance_mode": _PERFORMANCE_MODE_CODES.get(str(performance_mode or "NEUTRAL").upper(), 0.0),
        }
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values.values()) else None


def apply_independent_evidence(
    base: LinguisticDecision,
    alignment_evidence: Mapping[str, Any] | None,
    *,
    lid_evidence: Mapping[str, Any] | None = None,
    min_target_score: float = .65,
    min_margin: float = .20,
    source_leak_score: float = .75,
    target_language: str = "de",
    source_language: str = "en",
    calibration_authority: bool = False,
    calibration_profile: Mapping[str, Any] | None = None,
    calibration_profile_root: str | Path | None = None,
    feature_schema_version: str = _FEATURE_SCHEMA_VERSION,
    backend_id: str | None = None,
    runtime_lock_sha256: str | None = None,
    models_lock_sha256: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    performance_mode: str | None = None,
) -> LinguisticDecision:
    """Promote a screen only after a genuinely different evidence family.

    Two Whisper modes remain one ``WHISPER_ASR`` family.  Cross-language CTC
    margins are retained as diagnostics only; they are never a hard gate.
    A source-language leak additionally requires automatic Whisper to favour
    the source language, independent LID, and a weak target-only CTC result.
    """
    profile_status = calibration_profile_status(
        calibration_profile,
        authority=bool(calibration_authority),
        model_id=model_id,
        model_revision=model_revision,
        backend_id=backend_id,
        target_language=target_language,
        source_language=source_language,
        performance_mode=performance_mode,
        feature_schema_version=feature_schema_version,
        calibrator_root=calibration_profile_root,
        runtime_lock_sha256=runtime_lock_sha256,
        models_lock_sha256=models_lock_sha256,
    )
    calibrated = profile_status == "MATCHED_VALIDATED"
    if calibration_authority and not calibrated and not alignment_evidence:
        return LinguisticDecision(
            **{**base.__dict__, "status": "BLOCKED", "confirmed": False,
               "calibration_authority": False, "calibration_profile_status": profile_status,
               "reason": "calibration authority requested but profile does not match the active model, revision, language pair, or mode"}
        )
    if not alignment_evidence:
        return LinguisticDecision(
            **{**base.__dict__, "status": "ALIGNER_NOT_APPLICABLE", "confirmed": False,
               "calibration_authority": calibrated, "calibration_profile_status": profile_status,
               "reason": "no independent alignment family is available; candidate is held"}
        )
    target_score = float(alignment_evidence.get("target_score", (alignment_evidence.get("target") or {}).get("score", 0.0)) or 0.0)
    source_value = alignment_evidence.get("source_score")
    if source_value is None and alignment_evidence.get("source") is not None:
        source_value = (alignment_evidence.get("source") or {}).get("score")
    source_score = float(source_value) if source_value is not None else None
    margin_value = alignment_evidence.get("margin")
    margin = float(margin_value) if margin_value is not None else (target_score - source_score if source_score is not None else None)
    alignment_target = alignment_evidence.get("target") or {}
    final_anchor = alignment_target.get("final_anchor_present", base.final_anchor_present)
    final_anchor_evidence = alignment_target.get("final_anchor_evidence")
    if isinstance(final_anchor_evidence, Mapping):
        final_anchor_evidence = dict(final_anchor_evidence)
    else:
        final_anchor_evidence = None
    records = list(base.evidence_records)
    records.extend(list(alignment_evidence.get("evidence_records") or []))
    if lid_evidence and lid_evidence.get("record"):
        records.append(dict(lid_evidence["record"]))
    families = sorted({str(family) for family in base.evidence_families if family} | {str(item.get("evidence_family")) for item in records if item.get("evidence_family")})
    raw_target_score = target_score
    calibrated_target_probability: float | None = None
    raw_final_anchor_score: float | None = None
    calibrated_final_anchor_probability: float | None = None
    feature_vector: dict[str, float] | None = None
    feature_vector_hash: str | None = None
    final_anchor_feature_vector: dict[str, float] | None = None
    final_anchor_feature_vector_hash: str | None = None
    calibrated_lid_probability: float | None = None
    lid_feature_vector: dict[str, float] | None = None
    lid_feature_vector_hash: str | None = None
    calibrator_hash: str | None = None
    final_anchor_calibrator_hash: str | None = None
    lid_calibrator_hash: str | None = None
    if calibrated:
        calibrator, _calibrator_path, execution_status = _load_calibrator_artifact(
            calibration_profile or {},
            role="target",
            calibrator_root=calibration_profile_root,
            feature_schema_version=feature_schema_version,
        )
        final_anchor_calibrator, _final_anchor_path, final_anchor_status = _load_calibrator_artifact(
            calibration_profile or {},
            role="final_anchor",
            calibrator_root=calibration_profile_root,
            feature_schema_version=feature_schema_version,
        )
        lid_calibrator, _lid_path, lid_status = _load_calibrator_artifact(
            calibration_profile or {}, role="lid", calibrator_root=calibration_profile_root,
            feature_schema_version=feature_schema_version,
        )
        execution_status = execution_status or final_anchor_status or lid_status
        if execution_status or calibrator is None or final_anchor_calibrator is None or lid_calibrator is None:
            return LinguisticDecision(
                **{**base.__dict__, "status": "BLOCKED", "expected_alignment_score": raw_target_score,
                   "source_alignment_score": source_score, "alignment_margin": margin,
                   "cross_language_margin": margin, "final_anchor_present": None,
                   "final_anchor_evidence": final_anchor_evidence,
                   "char_segments": [dict(item) for item in (alignment_target.get("char_segments") or [])],
                   "native_char_coverage": alignment_target.get("native_char_coverage"),
                   "mean_char_score": alignment_target.get("mean_char_score"),
                   "minimum_char_score": alignment_target.get("minimum_char_score"),
                   "p10_char_score": alignment_target.get("p10_char_score"),
                   "unaligned_characters": list(alignment_target.get("unaligned_characters") or []),
                   "interpolated_characters": list(alignment_target.get("interpolated_characters") or []),
                   "compression_ratio": alignment_target.get("compression_ratio"),
                   "raw_target_score": raw_target_score,
                   "calibrated_target_probability": None,
                   "evidence_records": records, "evidence_families": families,
                   "calibration_authority": False,
                   "calibration_profile_status": execution_status or "BLOCKED_CALIBRATOR_NOT_EXECUTABLE",
                   "reason": "validated profile could not be executed by the built-in safe calibrator"}
            )
        calibrator_hash = str(((calibration_profile or {}).get("calibrators", {}).get("target", {})).get("artifact_sha256", ""))
        final_anchor_calibrator_hash = str(((calibration_profile or {}).get("calibrators", {}).get("final_anchor", {})).get("artifact_sha256", ""))
        lid_calibrator_hash = str(((calibration_profile or {}).get("calibrators", {}).get("lid", {})).get("artifact_sha256", ""))
        feature_vector = _alignment_feature_vector(alignment_target, target_score=raw_target_score, performance_mode=performance_mode)
        final_anchor_feature_vector = _final_anchor_feature_vector(alignment_target, target_score=raw_target_score, performance_mode=performance_mode)
        lid_feature_vector = _lid_feature_vector(lid_evidence, whisper_probability=base.language_probability, ctc_target_probability=raw_target_score, performance_mode=performance_mode)
        if feature_vector is None or final_anchor_feature_vector is None:
            return LinguisticDecision(
                **{**base.__dict__, "status": "BLOCKED", "expected_alignment_score": raw_target_score,
                   "source_alignment_score": source_score, "alignment_margin": margin,
                   "cross_language_margin": margin, "final_anchor_present": None,
                   "final_anchor_evidence": final_anchor_evidence,
                   "char_segments": [dict(item) for item in (alignment_target.get("char_segments") or [])],
                   "native_char_coverage": alignment_target.get("native_char_coverage"),
                   "mean_char_score": alignment_target.get("mean_char_score"),
                   "minimum_char_score": alignment_target.get("minimum_char_score"),
                   "p10_char_score": alignment_target.get("p10_char_score"),
                   "unaligned_characters": list(alignment_target.get("unaligned_characters") or []),
                   "interpolated_characters": list(alignment_target.get("interpolated_characters") or []),
                   "compression_ratio": alignment_target.get("compression_ratio"),
                   "raw_target_score": raw_target_score,
                   "calibrated_target_probability": None,
                   "feature_vector": feature_vector,
                   "feature_vector_hash": _feature_vector_hash(feature_vector) if feature_vector is not None else None,
                   "calibrator_hash": calibrator_hash,
                   "calibrator_artifact_sha256": calibrator_hash,
                   "evidence_records": records, "evidence_families": families,
                   "calibration_authority": False,
                   "lid_feature_vector": lid_feature_vector, "lid_calibrator_hash": lid_calibrator_hash,
                   "calibration_profile_status": "BLOCKED_CALIBRATION_FEATURES",
                   "reason": "calibrated alignment requires complete character-level features and final-anchor evidence"}
            )
        feature_vector_hash = _feature_vector_hash(feature_vector)
        final_anchor_feature_vector_hash = _feature_vector_hash(final_anchor_feature_vector, "final-anchor-v1")
        lid_feature_vector_hash = _feature_vector_hash(lid_feature_vector, _LID_FEATURE_SCHEMA_VERSION) if lid_feature_vector is not None else None
        calibrated_target_probability = _execute_platt_calibrator(calibrator, feature_vector)
        calibrated_final_anchor_probability = _execute_platt_calibrator(final_anchor_calibrator, final_anchor_feature_vector)
        calibrated_lid_probability = _execute_platt_calibrator(lid_calibrator, lid_feature_vector) if lid_feature_vector is not None else None
        if calibrated_target_probability is None or calibrated_final_anchor_probability is None or (lid_feature_vector is not None and calibrated_lid_probability is None):
            return LinguisticDecision(
                **{**base.__dict__, "status": "BLOCKED", "expected_alignment_score": raw_target_score,
                   "source_alignment_score": source_score, "alignment_margin": margin,
                   "cross_language_margin": margin, "final_anchor_present": None,
                   "final_anchor_evidence": final_anchor_evidence,
                   "raw_target_score": raw_target_score,
                   "calibrated_target_probability": calibrated_target_probability,
                   "raw_final_anchor_score": (final_anchor_evidence or {}).get("minimum_score"),
                   "calibrated_final_anchor_probability": calibrated_final_anchor_probability,
                   "feature_vector": feature_vector, "feature_vector_hash": feature_vector_hash,
                   "final_anchor_feature_vector": final_anchor_feature_vector,
                   "final_anchor_feature_vector_hash": final_anchor_feature_vector_hash,
                   "calibrator_hash": calibrator_hash,
                   "calibrator_artifact_sha256": calibrator_hash,
                   "final_anchor_calibrator_hash": final_anchor_calibrator_hash,
                   "final_anchor_calibrator_artifact_sha256": final_anchor_calibrator_hash,
                   "calibrated_lid_probability": calibrated_lid_probability,
                   "lid_feature_vector": lid_feature_vector,
                   "lid_feature_vector_hash": lid_feature_vector_hash,
                   "lid_calibrator_hash": lid_calibrator_hash,
                   "evidence_records": records, "evidence_families": families,
                   "calibration_authority": False,
                   "calibration_profile_status": "BLOCKED_CALIBRATION_EXECUTION",
                   "reason": "calibrator execution returned no finite probability"}
            )
        if final_anchor_evidence is not None:
            final_anchor_evidence["calibrated_probability"] = calibrated_final_anchor_probability
            final_anchor_evidence["calibrator_artifact_sha256"] = final_anchor_calibrator_hash
            final_anchor_evidence["feature_vector_hash"] = final_anchor_feature_vector_hash
    if base.audio_sha256:
        mismatched = [item for item in records if item.get("audio_sha256") and item.get("audio_sha256") != base.audio_sha256]
        if mismatched:
            return LinguisticDecision(
                **{**base.__dict__, "status": "ALIGNMENT_UNCERTAIN", "expected_alignment_score": target_score,
                   "source_alignment_score": source_score, "alignment_margin": margin, "cross_language_margin": margin,
                   "evidence_records": records, "evidence_families": families,
                   "calibration_authority": calibrated, "calibration_profile_status": profile_status,
                   "reason": "independent evidence belongs to a different audio artifact"}
            )
    independent_alignment = any(family in {"CTC_FORCED_ALIGNER", "KALDI_FORCED_ALIGNER"} for family in families)
    # A hard linguistic verdict cannot be manufactured from a bare score or
    # from one family whose record was accidentally omitted.  Keep the
    # candidate in review until Whisper + a second acoustic family are both
    # represented in the decision contract.
    if not independent_alignment or len(families) < 2:
        return LinguisticDecision(
            **{**base.__dict__, "status": "ALIGNMENT_UNCERTAIN", "expected_alignment_score": target_score,
               "source_alignment_score": source_score, "alignment_margin": margin, "cross_language_margin": margin,
               "evidence_records": records, "evidence_families": families,
               "calibration_authority": calibrated, "calibration_profile_status": profile_status,
               "reason": "at least two independent evidence families are required for a hard linguistic verdict"}
        )

    # An explicitly requested but non-matching profile is a configuration
    # block, never permission to use the default .65 score as authority.
    if calibration_authority and not calibrated:
        return LinguisticDecision(
            **{**base.__dict__, "status": "BLOCKED", "expected_alignment_score": target_score,
               "source_alignment_score": source_score, "alignment_margin": margin, "cross_language_margin": margin,
               "final_anchor_present": bool(final_anchor) if final_anchor is not None else base.final_anchor_present,
               "evidence_records": records, "evidence_families": families,
               "calibration_authority": False, "calibration_profile_status": profile_status,
               "reason": "calibration authority requested but profile does not match the active model, revision, language pair, or mode"}
        )

    lid_language = _language_code((lid_evidence or {}).get("language", ""))
    source_code = _language_code(source_language)
    lid_probability = float((lid_evidence or {}).get("probability", 0.0) or 0.0)
    lid_record = (lid_evidence or {}).get("record") or {}
    lid_family = str(lid_record.get("evidence_family", ""))
    independent_source_lid = bool(
        lid_evidence
        and lid_family == "AUDIO_LANGUAGE_ID"
        and lid_language == source_code
        and (calibrated_lid_probability if calibrated else lid_probability) >= _validated_profile_threshold(calibration_profile if calibrated else None, "source_lid_probability", .70)
    )
    # The raw source score and target-source margin are diagnostic telemetry.
    # German and English CTC models are not calibrated onto one probability
    # scale, so neither may decide a hard verdict.
    whisper_source = (
        _language_code(base.detected_language) == source_code
        and float(base.language_probability or 0.0) >= .70
    ) or base.status in {"LANGUAGE_LEAK_SUSPECTED", "LANGUAGE_LEAK_STRONG_SUSPICION"}
    target_pass_threshold = _validated_profile_threshold(calibration_profile if calibrated else None, "target_pass_probability", min_target_score)
    target_failure_threshold = _validated_profile_threshold(calibration_profile if calibrated else None, "target_failure_probability", min_target_score - .15)
    target_alignment_wins = (
        calibrated_target_probability is not None and calibrated_target_probability >= target_pass_threshold
        if calibrated else target_score >= min_target_score
    )
    target_alignment_weak = (
        calibrated_target_probability is not None and calibrated_target_probability < target_failure_threshold
        if calibrated else target_score < min_target_score
    )

    if not calibrated:
        if whisper_source and target_alignment_wins:
            # A high target score cannot overwrite source-language evidence
            # until that score has a matched calibration profile.
            status: LinguisticStatus = "EVIDENCE_CONFLICT"
            reason = "Whisper favors source language while uncalibrated target CTC appears strong; evidence is held"
        elif whisper_source and target_alignment_weak and independent_source_lid:
            status = "LANGUAGE_LEAK_STRONG_SUSPICION"
            reason = "Whisper and independent LID favor source language while target CTC is weak; calibration is required"
        elif whisper_source and target_alignment_weak:
            status = "LANGUAGE_LEAK_STRONG_SUSPICION"
            reason = "Whisper favors source language and target CTC is weak; calibration is required before confirmation"
        elif target_alignment_wins:
            status = "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT" if base.status == "PASS_SCREENED" else "TARGET_ALIGNMENT_SUPPORT"
            reason = "uncalibrated target-only CTC supports the target phonetically; production eligibility remains blocked"
        elif target_score < (min_target_score - .15):
            status = "LEXICAL_FAILURE_SUSPECTED"
            reason = "target-only alignment is weak; calibration is required before a lexical failure is confirmed"
        else:
            status = "TARGET_ALIGNMENT_WEAK"
            reason = "target-only alignment is below threshold; result is diagnostic-only until calibration"
    elif whisper_source and target_alignment_weak and independent_source_lid:
        status = "LANGUAGE_LEAK_CONFIRMED"
        reason = "Whisper and independent LID favor source language while calibrated target CTC is weak"
    elif target_alignment_wins:
        status = "PASS_CONFIRMED" if base.status == "PASS_SCREENED" else "PASS_PHONETIC"
        reason = "calibrated target-only CTC supports target phonetic content; cross-language margin is diagnostic"
    elif calibrated_target_probability is not None and calibrated_target_probability < target_failure_threshold:
        status = "LEXICAL_FAILURE_CONFIRMED"
        reason = "calibrated target-only alignment rejects the target content"
    else:
        status = "ALIGNMENT_UNCERTAIN"
        reason = "calibrated target alignment is inconclusive"
    return LinguisticDecision(
        **{**base.__dict__, "status": status, "expected_alignment_score": target_score,
           "source_alignment_score": source_score, "alignment_margin": margin, "cross_language_margin": margin,
           "final_anchor_present": (None if final_anchor_evidence is not None else (bool(final_anchor) if final_anchor is not None else base.final_anchor_present)),
           "final_anchor_evidence": final_anchor_evidence,
           "char_segments": [dict(item) for item in (alignment_target.get("char_segments") or [])],
           "native_char_coverage": alignment_target.get("native_char_coverage"),
           "mean_char_score": alignment_target.get("mean_char_score"),
           "minimum_char_score": alignment_target.get("minimum_char_score"),
           "p10_char_score": alignment_target.get("p10_char_score"),
           "unaligned_characters": list(alignment_target.get("unaligned_characters") or []),
           "interpolated_characters": list(alignment_target.get("interpolated_characters") or []),
           "delete_count": int(alignment_target.get("delete_count", 0) or 0),
           "insert_count": int(alignment_target.get("insert_count", 0) or 0),
           "substitute_count": int(alignment_target.get("substitute_count", 0) or 0),
           "interpolated_count": int(alignment_target.get("interpolated_count", 0) or 0),
           "alignment_operation_hash": alignment_target.get("alignment_operation_hash"),
           "compression_ratio": alignment_target.get("compression_ratio"),
           "duration": alignment_target.get("duration"),
           "characters_per_second": alignment_target.get("characters_per_second"),
           "words_per_second": alignment_target.get("words_per_second"),
           "normalization_version": alignment_target.get("normalization_version", _NORMALIZATION_VERSION),
           "raw_target_score": raw_target_score,
           "calibrated_target_probability": calibrated_target_probability,
           "raw_final_anchor_score": (final_anchor_evidence or {}).get("minimum_score"),
           "calibrated_final_anchor_probability": calibrated_final_anchor_probability,
           "feature_vector": feature_vector,
           "feature_vector_hash": feature_vector_hash,
           "final_anchor_feature_vector": final_anchor_feature_vector,
           "final_anchor_feature_vector_hash": final_anchor_feature_vector_hash,
           "calibrator_hash": calibrator_hash,
           "calibrator_artifact_sha256": calibrator_hash,
           "final_anchor_calibrator_hash": final_anchor_calibrator_hash,
           "final_anchor_calibrator_artifact_sha256": final_anchor_calibrator_hash,
           "calibrated_lid_probability": calibrated_lid_probability,
           "lid_feature_vector": lid_feature_vector,
           "lid_feature_vector_hash": lid_feature_vector_hash,
           "lid_calibrator_hash": lid_calibrator_hash,
           "evidence_records": records, "evidence_families": families,
           "confirmed": status in {"PASS_CONFIRMED", "PASS_PHONETIC", "LANGUAGE_LEAK_CONFIRMED", "LEXICAL_FAILURE_CONFIRMED"},
           "calibration_authority": calibrated, "calibration_profile_status": profile_status,
           "reason": reason}
    )


def _failure_for(name: str) -> FailureClass:
    if name in {"not_empty", "finite_audio", "sample_rate", "channels", "frames", "clipping", "serialization_contract"}:
        return FailureClass.DETERMINISTIC_SERIALIZATION
    if name in {"splice_seam", "splice_boundary", "splice_speech_timing", "preserved_intervals"}:
        return FailureClass.DETERMINISTIC_PROCESSING
    if name in {"content", "final_word"}:
        return FailureClass.STOCHASTIC_TTS
    if name == "source_language":
        return FailureClass.STOCHASTIC_TTS
    return FailureClass.DETERMINISTIC_WINDOW


def evaluate_candidate_v2(path: str, *, expected_text: str, source_text: str = "", target_sample_rate: int | None = None,
                          target_frames: int | None = None, channels: int | None = None, reference_end: float | None = None,
                          transcript: str | None = None, language: str | None = None, language_probability: float | None = None,
                          profile: LanguageProfile | None = None, hard_gates: Sequence[str] | None = None,
                          final_word_min_tokens: int = 1, tail_guard_seconds: float = .08,
                          splice_metrics: Mapping[str, tuple[bool, Any, Any, str]] | None = None,
                          preserved_ok: bool | None = None, require_asr: bool = True, neutral_effort: bool = False,
                          linguistic_evidence: Mapping[str, Any] | None = None,
                          alignment_evidence: Mapping[str, Any] | None = None,
                          lid_evidence: Mapping[str, Any] | None = None,
                          alignment_min_target_score: float = .65,
                          alignment_min_margin: float = .20,
                          alignment_source_leak_score: float = .75,
                          calibration_authority: bool = False,
                          calibration_profile: Mapping[str, Any] | None = None,
                          calibration_profile_root: str | Path | None = None,
                          feature_schema_version: str = _FEATURE_SCHEMA_VERSION,
                          backend_id: str | None = None,
                          runtime_lock_sha256: str | None = None,
                          models_lock_sha256: str | None = None,
                          model_id: str | None = None,
                          model_revision: str | None = None,
                          performance_mode: str | None = None) -> QAResultV2:
    profile = profile or LanguageProfile()
    gates: dict[str, GateEvidence] = {}
    diagnostics: dict[str, Any] = {}
    try:
        audio, sample_rate = read(path, always_2d=True)
    except Exception as exc:
        gates["serialization_contract"] = _gate("serialization_contract", GateStatus.ERROR, details={"error": str(exc)})
        return QAResultV2(False, gates, {"error": str(exc)}, FailureClass.DETERMINISTIC_SERIALIZATION)
    import numpy as np
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    diagnostics.update({"sample_rate": sample_rate, "frames": int(len(audio)), "channels": int(audio.shape[1]), "peak_dbfs": peak_dbfs(audio)})
    gates["serialization_contract"] = _gate("serialization_contract", GateStatus.PASS, measured=True, details={"reopened": True})
    gates["not_empty"] = _gate("not_empty", GateStatus.PASS if len(audio) > 0 and peak > 0 else GateStatus.FAIL, measured=int(len(audio)), threshold=1, units="frames")
    gates["finite_audio"] = _gate("finite_audio", GateStatus.PASS if finite else GateStatus.FAIL, measured=bool(finite))
    gates["sample_rate"] = _gate("sample_rate", GateStatus.NOT_APPLICABLE if target_sample_rate is None else (GateStatus.PASS if sample_rate == target_sample_rate else GateStatus.FAIL), measured=sample_rate, threshold=target_sample_rate, units="Hz")
    gates["channels"] = _gate("channels", GateStatus.NOT_APPLICABLE if channels is None else (GateStatus.PASS if audio.shape[1] == channels else GateStatus.FAIL), measured=int(audio.shape[1]), threshold=channels, units="channels")
    gates["frames"] = _gate("frames", GateStatus.NOT_APPLICABLE if target_frames is None else (GateStatus.PASS if len(audio) == target_frames else GateStatus.FAIL), measured=int(len(audio)), threshold=target_frames, units="frames")
    clipped = clipping(audio)
    gates["clipping"] = _gate("clipping", GateStatus.FAIL if clipped else GateStatus.PASS, measured=int(np.count_nonzero(np.abs(audio) >= .999)), threshold=0, units="samples")
    active_rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    diagnostics["active_rms_db"] = -120.0 if active_rms <= 1e-12 else 20.0 * __import__("math").log10(active_rms)
    gates["active_loudness"] = _gate("active_loudness", GateStatus.PASS if active_rms > 1e-7 else GateStatus.FAIL, measured=diagnostics["active_rms_db"], threshold=-140.0, units="dBFS")
    gates["lufs"] = _gate("lufs", gates["active_loudness"].status, measured=diagnostics["active_rms_db"], threshold=-140.0, units="dBFS")
    if reference_end is None:
        gates["tail"] = _gate("tail", GateStatus.NOT_APPLICABLE)
    else:
        end = speech_end(audio, sample_rate); diagnostics["voice_end"] = end
        gates["tail"] = _gate("tail", GateStatus.PASS if end <= reference_end + tail_guard_seconds else GateStatus.FAIL, measured=end, threshold=reference_end + tail_guard_seconds, units="seconds")
    lexical_decision: LinguisticDecision | None = None
    lexical_ready = bool(transcript and transcript.strip()) or bool(linguistic_evidence)
    if linguistic_evidence:
        forced_row = dict(linguistic_evidence.get("forced_target") or {})
        automatic_row = dict(linguistic_evidence.get("automatic") or {})
        declared_records = [dict(item) for item in linguistic_evidence.get("evidence_records", [])]
        if not declared_records:
            declared_records = [
                dict(row["evidence_record"])
                for row in (forced_row, automatic_row)
                if isinstance(row.get("evidence_record"), Mapping)
            ]
        declared_hashes = list(linguistic_evidence.get("evidence_hashes") or [])
        if not declared_hashes:
            declared_hashes = [str(row.get("evidence_hash")) for row in (forced_row, automatic_row) if row.get("evidence_hash")]
        target_language = str(linguistic_evidence.get("target_language") or profile.target_language)
        lexical_decision = decide_linguistic_evidence(
            expected_text,
            source_text,
            forced_target=forced_row,
            automatic=automatic_row,
            target_language=target_language,
            profile=profile,
            final_word_min_tokens=final_word_min_tokens,
            evidence_hashes=declared_hashes,
            evidence_records=declared_records,
            audio_sha256=str(linguistic_evidence.get("audio_sha256") or "") or None,
        )
        if alignment_evidence is not None or lid_evidence is not None:
            lexical_decision = apply_independent_evidence(
                lexical_decision,
                alignment_evidence,
                lid_evidence=lid_evidence,
                min_target_score=alignment_min_target_score,
                min_margin=alignment_min_margin,
                source_leak_score=alignment_source_leak_score,
                target_language=target_language,
                source_language=profile.source_language,
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
        forced_text = lexical_decision.forced_transcript or ""
        automatic_text = lexical_decision.automatic_transcript or ""
        forced_content_ok, forced_content_details = ordered_content(expected_text, forced_text)
        forced_final_ok, forced_final_details = final_word(expected_text, forced_text, final_word_min_tokens)
        leak_ok, leak_details = source_language_leak(
            source_text,
            automatic_text,
            lexical_decision.detected_language,
            lexical_decision.language_probability,
            profile,
        )
        evidence_hash = lexical_decision.evidence_hashes[0] if lexical_decision.evidence_hashes else None
        alignment_calibrated = bool(
            lexical_decision.calibration_authority
            and lexical_decision.calibration_profile_status == "MATCHED_VALIDATED"
            and lexical_decision.calibrated_target_probability is not None
        )
        target_pass_threshold = _validated_profile_threshold(calibration_profile if alignment_calibrated else None, "target_pass_probability", alignment_min_target_score)
        source_lid_threshold = _validated_profile_threshold(calibration_profile if alignment_calibrated else None, "source_lid_probability", .70)
        alignment_confirms_content = alignment_calibrated and lexical_decision.status in {"PASS_CONFIRMED", "PASS_PHONETIC"} and lexical_decision.calibrated_target_probability >= target_pass_threshold
        # A CTC score may rescue a Whisper lexical miss, but it may not
        # silently certify an absent final-word anchor.  ``None`` remains
        # unknown and therefore keeps the hard final-word gate closed.
        alignment_confirms_final = alignment_confirms_content and _final_anchor_is_calibrated(lexical_decision, profile=calibration_profile, feature_schema_version=feature_schema_version)
        gates["content"] = _gate("content", GateStatus.PASS if (forced_content_ok or alignment_confirms_content) else GateStatus.FAIL, measured=forced_content_details.get("matched_in_order"), details={**forced_content_details, "linguistic_status": lexical_decision.status, "alignment_confirmed": alignment_confirms_content, "raw_target_score": lexical_decision.raw_target_score, "calibrated_target_probability": lexical_decision.calibrated_target_probability}, evidence_hash=evidence_hash)
        gates["final_word"] = _gate("final_word", GateStatus.PASS if (forced_final_ok or alignment_confirms_final) else GateStatus.FAIL, measured=forced_final_details.get("heard_final_tokens"), details={**forced_final_details, "linguistic_status": lexical_decision.status, "alignment_confirmed": alignment_confirms_final, "raw_final_anchor_score": lexical_decision.raw_final_anchor_score, "calibrated_final_anchor_probability": lexical_decision.calibrated_final_anchor_probability}, evidence_hash=evidence_hash)
        independent_source_lid = bool(
            lid_evidence
            and str((lid_evidence.get("record") or {}).get("evidence_family", "")) == "AUDIO_LANGUAGE_ID"
            and _language_code(lid_evidence.get("language")) == _language_code(profile.source_language)
            and float(lid_evidence.get("probability", 0.0) or 0.0) >= source_lid_threshold
        )
        independent_target_lid = bool(
            lid_evidence
            and str((lid_evidence.get("record") or {}).get("evidence_family", "")) == "AUDIO_LANGUAGE_ID"
            and _language_code(lid_evidence.get("language")) == _language_code(profile.target_language)
            and float(lid_evidence.get("probability", 0.0) or 0.0) >= source_lid_threshold
        )
        # Before calibration, a target score can never override a source
        # language gate.  Even after calibration, the override needs a final
        # anchor and independent LID that does not favour the source.
        alignment_overrides_whisper_leak = bool(
            alignment_calibrated
            and alignment_confirms_final
            and independent_target_lid
            and not independent_source_lid
            and lexical_decision.status in {"PASS_CONFIRMED", "PASS_PHONETIC"}
        )
        source_conflict = lexical_decision.status == "EVIDENCE_CONFLICT"
        gates["source_language"] = _gate(
            "source_language",
            GateStatus.PASS if ((leak_ok and not source_conflict) or alignment_overrides_whisper_leak) else GateStatus.FAIL,
            details={**leak_details, "linguistic_status": lexical_decision.status, "automatic_transcript": automatic_text, "alignment_overrode_whisper_leak": alignment_overrides_whisper_leak},
            evidence_hash=evidence_hash,
        )
        diagnostics["linguistic_decision"] = lexical_decision.to_dict()
        diagnostics["asr"] = dict(linguistic_evidence)
        if alignment_evidence is not None:
            diagnostics["alignment"] = dict(alignment_evidence)
        if lid_evidence is not None:
            diagnostics["language_id"] = dict(lid_evidence)
    elif not lexical_ready:
        lexical_status = GateStatus.NOT_RUN
        lexical_details = {"reason": "ASR_NOT_RUN"}
        if neutral_effort:
            lexical_status = GateStatus.NOT_APPLICABLE; lexical_details = {"reason": "neutral_effort"}
        gates["content"] = _gate("content", lexical_status, details=lexical_details)
        gates["final_word"] = _gate("final_word", lexical_status, details=lexical_details)
        gates["source_language"] = _gate("source_language", lexical_status, details=lexical_details)
    else:
        content_ok, content_details = ordered_content(expected_text, transcript)
        final_ok, final_details = final_word(expected_text, transcript, final_word_min_tokens)
        leak_ok, leak_details = source_language_leak(source_text, transcript, language, language_probability, profile)
        gates["content"] = _gate("content", GateStatus.PASS if content_ok else GateStatus.FAIL, measured=content_details.get("matched_in_order"), details=content_details)
        gates["final_word"] = _gate("final_word", GateStatus.PASS if final_ok else GateStatus.FAIL, measured=final_details.get("heard_final_tokens"), details=final_details)
        gates["source_language"] = _gate("source_language", GateStatus.PASS if leak_ok else GateStatus.FAIL, details=leak_details)
    if splice_metrics:
        for name, (ok, measured, threshold, units) in splice_metrics.items():
            gates[name] = _gate(name, GateStatus.PASS if ok else GateStatus.FAIL, measured=measured, threshold=threshold, units=units)
    else:
        for name in ("splice_seam", "splice_boundary", "splice_speech_timing"):
            gates[name] = _gate(name, GateStatus.NOT_APPLICABLE)
    gates["preserved_intervals"] = _gate("preserved_intervals", GateStatus.NOT_APPLICABLE if preserved_ok is None else (GateStatus.PASS if preserved_ok else GateStatus.FAIL), measured=preserved_ok)
    required = list(hard_gates or ("not_empty", "finite_audio", "sample_rate", "frames", "clipping", "active_loudness", "tail", "content", "final_word", "source_language", "serialization_contract"))
    if not require_asr:
        required = [name for name in required if name not in {"content", "final_word", "source_language"}]
    failures = [name for name in required if not gate_passes(gates.get(name, _gate(name, GateStatus.NOT_RUN)), allow_not_applicable=True)]
    # NOT_RUN is never silently accepted when it was a required gate.
    passed = not failures and all(gates[name].status is not GateStatus.NOT_RUN for name in required)
    if lexical_decision is not None and lexical_decision.status not in {"PASS_CONFIRMED", "PASS_PHONETIC"}:
        # Uncertainty is a hold/review state, never an implicit PASS and never
        # a reason for the cohort scheduler to regenerate the line.
        passed = False
    if lexical_decision is not None and lexical_decision.status == "BLOCKED":
        failure = FailureClass.DETERMINISTIC_CALIBRATION
    elif lexical_decision is not None and lexical_decision.status in {
        "ASR_UNCERTAIN", "ALIGNMENT_UNCERTAIN", "LANGUAGE_LEAK_SUSPECTED",
        "LANGUAGE_LEAK_STRONG_SUSPICION", "LEXICAL_FAILURE_SUSPECTED",
        "TARGET_ALIGNMENT_SUPPORT", "TARGET_ALIGNMENT_WEAK",
        "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT", "EVIDENCE_CONFLICT",
        "ALIGNER_NOT_APPLICABLE", "HUMAN_REVIEW",
    }:
        failure = FailureClass.ASR_UNCERTAIN
    elif lexical_decision is not None and lexical_decision.status in {"LANGUAGE_LEAK_CONFIRMED", "LEXICAL_FAILURE_CONFIRMED"}:
        failure = FailureClass.STOCHASTIC_TTS
    elif lexical_decision is not None and lexical_decision.status in {"PASS_CONFIRMED", "PASS_PHONETIC"} and "source_language" in failures:
        # A calibrated content result without the independent target-language
        # evidence required to clear the source gate is still uncertain; do
        # not misclassify that policy hold as a TTS retry.
        failure = FailureClass.ASR_UNCERTAIN
    else:
        failure = _failure_for(failures[0]) if failures else None
    return QAResultV2(passed, gates, diagnostics, failure)


def rank_candidate_v2(result: QAResultV2) -> float:
    """Rank final candidates; provisional candidates use ``rank_provisional``."""
    if not result.passed:
        return float("-inf")
    score = 0.0
    score += 2.0 if result.gates.get("final_word", _gate("final_word", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score += 2.0 if result.gates.get("content", _gate("content", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score -= abs(float(result.diagnostics.get("active_rms_db", -20.0)) + 20.0) * .01
    return score


def linguistic_status(result: QAResultV2) -> str | None:
    value = result.diagnostics.get("linguistic_decision")
    return str(value.get("status")) if isinstance(value, Mapping) and value.get("status") else None


def is_provisional_result(result: QAResultV2) -> bool:
    return linguistic_status(result) in {
        "PASS_SCREENED", "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT",
        "TARGET_ALIGNMENT_SUPPORT", "TARGET_ALIGNMENT_WEAK",
        "LEXICAL_FAILURE_SUSPECTED", "ASR_UNCERTAIN",
        "LANGUAGE_LEAK_SUSPECTED", "LANGUAGE_LEAK_STRONG_SUSPICION",
        "EVIDENCE_CONFLICT", "ALIGNMENT_UNCERTAIN", "ALIGNER_NOT_APPLICABLE",
    }


def rank_provisional_v2(result: QAResultV2) -> float:
    """Rank a candidate for selective alignment without declaring final PASS."""
    if not is_provisional_result(result):
        return float("-inf")
    technical = ("not_empty", "finite_audio", "sample_rate", "channels", "frames", "clipping", "active_loudness", "tail", "serialization_contract")
    if any(result.gates.get(name, _gate(name, GateStatus.NOT_RUN)).status is GateStatus.FAIL for name in technical):
        return float("-inf")
    status = linguistic_status(result)
    score = 0.0
    score += 3.0 if status in {"PASS_SCREENED", "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT"} else 1.0
    score += 1.5 if result.gates.get("final_word", _gate("final_word", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score += 1.5 if result.gates.get("content", _gate("content", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score += float(result.diagnostics.get("linguistic_decision", {}).get("word_coverage") or 0.0)
    return score


def select_passed_v2(evaluations: Sequence[tuple[Any, QAResultV2]]) -> tuple[Any, QAResultV2] | None:
    passed = [(candidate, result) for candidate, result in evaluations if result.passed]
    return max(passed, key=lambda item: rank_candidate_v2(item[1])) if passed else None


__all__ = [
    "LanguageProfile", "LinguisticDecision", "LinguisticStatus", "QAResultV2",
    "apply_independent_evidence", "calibration_profile_status", "decide_linguistic_evidence", "evaluate_candidate_v2", "final_word", "load_safe_calibrator", "predict_probability",
    "is_provisional_result", "linguistic_status", "ordered_content", "rank_provisional_v2",
    "select_passed_v2", "source_language_leak",
]
