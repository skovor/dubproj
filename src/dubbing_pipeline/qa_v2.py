"""Measured QA gates and fail-closed candidate selection."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .audio import clipping, peak_dbfs, read
from .contracts import FailureClass, GateEvidence, GateStatus, gate_passes
from .timing import speech_end

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def tokens(text: str) -> list[str]:
    return [fold(item) for item in _TOKEN.findall(str(text or ""))]


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
    "PASS_EXACT",
    "PASS_PHONETIC",
    "FAIL_CONFIRMED",
    "ASR_UNCERTAIN",
    "LANGUAGE_LEAK_CONFIRMED",
    "PERFORMANCE_UNCERTAIN",
    "NOT_APPLICABLE",
]


@dataclass(frozen=True)
class LinguisticDecision:
    """Structured linguistic verdict; never collapse evidence to one boolean."""

    status: LinguisticStatus
    expected_alignment_score: float | None = None
    source_alignment_score: float | None = None
    alignment_margin: float | None = None
    forced_transcript: str | None = None
    automatic_transcript: str | None = None
    detected_language: str | None = None
    language_probability: float | None = None
    word_coverage: float | None = None
    phone_coverage: float | None = None
    final_anchor_present: bool | None = None
    missing_tokens: list[str] = field(default_factory=list)
    expected_tokens: list[str] = field(default_factory=list)
    forced_final_tokens: list[str] = field(default_factory=list)
    automatic_final_tokens: list[str] = field(default_factory=list)
    evidence_hashes: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_alignment_score": self.expected_alignment_score,
            "source_alignment_score": self.source_alignment_score,
            "alignment_margin": self.alignment_margin,
            "forced_transcript": self.forced_transcript,
            "automatic_transcript": self.automatic_transcript,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "word_coverage": self.word_coverage,
            "phone_coverage": self.phone_coverage,
            "final_anchor_present": self.final_anchor_present,
            "missing_tokens": list(self.missing_tokens),
            "expected_tokens": list(self.expected_tokens),
            "forced_final_tokens": list(self.forced_final_tokens),
            "automatic_final_tokens": list(self.automatic_final_tokens),
            "evidence_hashes": list(self.evidence_hashes),
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
    probable = (language or "").casefold() == profile.source_language.casefold() and float(probability or 0.0) >= .70
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
) -> LinguisticDecision:
    """Fuse two ASR readings without relaxing the lexical gates.

    The exact content and final-word predicates remain unchanged.  The only
    new behavior is explicit uncertainty when the two decodes disagree, which
    prevents a single Whisper mistake from becoming either a regeneration
    order or a false PASS.
    """
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
    automatic_target = (detected or "").casefold() == target_language.casefold()
    target_confirmed = language_known and automatic_target and float(automatic_row["probability"] or 0.0) >= .70
    # A backend without language metadata is not allowed to create a new
    # hard PASS under the dual policy.  Legacy single-transcript callers keep
    # their old behavior because this helper is only used with dual evidence.
    if leak_details.get("strong_source_hits") or (not leak_ok and detected and not automatic_target):
        status: LinguisticStatus = "LANGUAGE_LEAK_CONFIRMED"
        reason = "automatic ASR independently indicates source-language speech"
    elif forced_content_ok and forced_final_ok and automatic_content_ok and automatic_final_ok and target_confirmed:
        status = "PASS_EXACT"
        reason = "forced and automatic ASR agree on target content and final anchor"
    elif forced_content_ok and forced_final_ok and automatic_target and (not automatic_content_ok or not automatic_final_ok):
        status = "ASR_UNCERTAIN"
        reason = "forced target decode passes but automatic decode disagrees"
    elif automatic_content_ok and automatic_final_ok and (not forced_content_ok or not forced_final_ok):
        status = "ASR_UNCERTAIN"
        reason = "automatic decode passes but forced target decode disagrees"
    elif target_confirmed and not forced_content_ok and not automatic_content_ok:
        status = "FAIL_CONFIRMED"
        reason = "both independent target-language readings miss required content"
    else:
        status = "ASR_UNCERTAIN"
        reason = "language metadata or lexical evidence is insufficient for a definitive verdict"

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
        reason=reason,
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
                          linguistic_evidence: Mapping[str, Any] | None = None) -> QAResultV2:
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
        target_language = str(linguistic_evidence.get("target_language") or profile.target_language)
        lexical_decision = decide_linguistic_evidence(
            expected_text,
            source_text,
            forced_target=forced_row,
            automatic=automatic_row,
            target_language=target_language,
            profile=profile,
            final_word_min_tokens=final_word_min_tokens,
            evidence_hashes=list(linguistic_evidence.get("evidence_hashes") or []),
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
        gates["content"] = _gate("content", GateStatus.PASS if forced_content_ok else GateStatus.FAIL, measured=forced_content_details.get("matched_in_order"), details={**forced_content_details, "linguistic_status": lexical_decision.status}, evidence_hash=evidence_hash)
        gates["final_word"] = _gate("final_word", GateStatus.PASS if forced_final_ok else GateStatus.FAIL, measured=forced_final_details.get("heard_final_tokens"), details={**forced_final_details, "linguistic_status": lexical_decision.status}, evidence_hash=evidence_hash)
        gates["source_language"] = _gate("source_language", GateStatus.PASS if leak_ok else GateStatus.FAIL, details={**leak_details, "linguistic_status": lexical_decision.status, "automatic_transcript": automatic_text}, evidence_hash=evidence_hash)
        diagnostics["linguistic_decision"] = lexical_decision.to_dict()
        diagnostics["asr"] = dict(linguistic_evidence)
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
    if lexical_decision is not None and lexical_decision.status not in {"PASS_EXACT", "PASS_PHONETIC"}:
        # Uncertainty is a hold/review state, never an implicit PASS and never
        # a reason for the cohort scheduler to regenerate the line.
        passed = False
    if lexical_decision is not None and lexical_decision.status == "ASR_UNCERTAIN":
        failure = FailureClass.ASR_UNCERTAIN
    elif lexical_decision is not None and lexical_decision.status == "LANGUAGE_LEAK_CONFIRMED":
        failure = FailureClass.STOCHASTIC_TTS
    else:
        failure = _failure_for(failures[0]) if failures else None
    return QAResultV2(passed, gates, diagnostics, failure)


def rank_candidate_v2(result: QAResultV2) -> float:
    """Ranking only compares already-passed candidates using measured data."""
    if not result.passed:
        return float("-inf")
    score = 0.0
    score += 2.0 if result.gates.get("final_word", _gate("final_word", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score += 2.0 if result.gates.get("content", _gate("content", GateStatus.NOT_RUN)).status is GateStatus.PASS else 0.0
    score -= abs(float(result.diagnostics.get("active_rms_db", -20.0)) + 20.0) * .01
    return score


def select_passed_v2(evaluations: Sequence[tuple[Any, QAResultV2]]) -> tuple[Any, QAResultV2] | None:
    passed = [(candidate, result) for candidate, result in evaluations if result.passed]
    return max(passed, key=lambda item: rank_candidate_v2(item[1])) if passed else None


__all__ = [
    "LanguageProfile", "LinguisticDecision", "LinguisticStatus", "QAResultV2",
    "decide_linguistic_evidence", "evaluate_candidate_v2", "final_word",
    "ordered_content", "select_passed_v2", "source_language_leak",
]
