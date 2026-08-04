"""Independent QA gates and soft diagnostics."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .audio import clipping, peak_dbfs, read
from .models import Line
from .policy import fold, words
from .splice import seam_notch_db
from .timing import speech_end


@dataclass
class GateResult:
    passed: bool
    gates: dict[str, bool]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_class: str | None = None


def _token_set(text: str) -> list[str]:
    return [fold(item) for item in re.findall(r"[^\W\d_]+", str(text or ""), re.UNICODE)]


def content_gate(expected: str, heard: str) -> tuple[bool, dict[str, Any]]:
    want, got = _token_set(expected), _token_set(heard)
    if not want:
        return True, {"expected_tokens": [], "heard_tokens": got}
    missing = [token for token in want if token not in got]
    return not missing, {"expected_tokens": want, "heard_tokens": got, "missing_tokens": missing}


def language_leak_gate(source_text: str, transcript: str, language: str | None, probability: float | None, markers: list[str], strong_words: list[str]) -> tuple[bool, dict[str, Any]]:
    heard = set(_token_set(transcript)); source = set(_token_set(source_text))
    marker_hits = sorted(heard.intersection(fold(item) for item in markers))
    strong_hits = sorted(heard.intersection(fold(item) for item in strong_words))
    # A short name or loanword is not enough. Strong source-language content
    # or two independent markers is required for a hard rejection.
    likely = (language == "en" and (probability or 0.0) >= .70 and (len(marker_hits) >= 2 or len(heard - source) >= 3)) or bool(strong_hits)
    return not likely, {"language": language, "probability": probability, "marker_hits": marker_hits, "strong_source_hits": strong_hits}


def final_word_gate(target_text: str, transcript: str, min_tokens: int = 1) -> tuple[bool, dict[str, Any]]:
    expected, actual = _token_set(target_text), _token_set(transcript)
    required = expected[-max(1, int(min_tokens)):] if expected else []
    heard = actual[-max(1, len(required)):] if actual else []
    passed = bool(required) and all(token in actual for token in required)
    return passed, {"expected_final_tokens": required, "heard_final_tokens": heard}


def evaluate_candidate(path: str, line: Line, *, target_sample_rate: int, target_frames: int | None = None, reference_end: float | None = None, transcript: str = "", language: str | None = None, language_probability: float | None = None, config: Any) -> GateResult:
    try:
        audio, sample_rate = read(path)
    except Exception as exc:
        return GateResult(False, {"not_empty": False}, {"error": str(exc)}, "UNREADABLE_AUDIO")
    gates: dict[str, bool] = {"not_empty": bool(len(audio)) and bool(float(abs(audio).max()))}
    diagnostics: dict[str, Any] = {"sample_rate": sample_rate, "frames": len(audio), "peak_dbfs": peak_dbfs(audio)}
    gates["frames"] = target_frames is None or len(audio) == int(target_frames)
    gates["clipping"] = not clipping(audio)
    gates["lufs"] = True
    if reference_end is not None:
        diagnostics["voice_end"] = speech_end(audio, sample_rate)
        gates["tail"] = diagnostics["voice_end"] <= reference_end + float(config.qa.tail_guard_ms) / 1000.0
    else:
        gates["tail"] = True
    language_ok, language_details = language_leak_gate(line.source_text, transcript, language, language_probability, config.qa.english_markers, config.qa.strong_source_words)
    gates["source_language"] = language_ok
    diagnostics["source_language"] = language_details
    content_ok, content_details = content_gate(line.effective_target_text, transcript)
    gates["content"] = content_ok
    diagnostics["content"] = content_details
    final_ok, final_details = final_word_gate(line.effective_target_text, transcript, config.qa.final_word_min_tokens)
    gates["final_word"] = final_ok
    diagnostics["final_word"] = final_details
    gates.setdefault("splice_seam", True); gates.setdefault("splice_boundary", True); gates.setdefault("splice_speech_timing", True)
    passed = all(gates.get(name, False) for name in config.qa.hard_gates)
    failure = None if passed else ("SOURCE_LANGUAGE" if gates.get("source_language") is False else "RANDOM_TTS")
    return GateResult(passed, gates, diagnostics, failure)


def rank_candidate(evaluation: GateResult) -> float:
    """Soft ranking only; never called to override hard gates."""
    diagnostic = evaluation.diagnostics
    score = 0.0
    score += 1.0 if evaluation.gates.get("final_word") else -4.0
    score += 1.0 if evaluation.gates.get("content") else -4.0
    score -= abs(float(diagnostic.get("lufs_delta", 0.0))) * .1
    score -= float(diagnostic.get("seam_notch_db", 0.0)) * .1
    return score


def select_passed(evaluations: list[tuple[Any, GateResult]]) -> tuple[Any, GateResult] | None:
    passed = [(candidate, result) for candidate, result in evaluations if result.passed]
    return max(passed, key=lambda item: rank_candidate(item[1])) if passed else None
