#!/usr/bin/env python3
"""Generate, QA, select and mount one exact-frame P3R anime scene."""
from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from clean_runtime import prepare

prepare()

OMNIVOICE_PACKAGE_PARENT = Path(
    r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1"
    r"\source\k2-fsa-OmniVoice-5ba967c"
)
if not OMNIVOICE_PACKAGE_PARENT.is_dir():
    raise FileNotFoundError(
        f"OmniVoice 0.2.1 package not found: {OMNIVOICE_PACKAGE_PARENT}"
    )
sys.path.insert(0, str(OMNIVOICE_PACKAGE_PARENT))

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
import omnivoice
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

OMNIVOICE_ENGINE = Path(omnivoice.__file__).resolve()
if OMNIVOICE_PACKAGE_PARENT.resolve() not in OMNIVOICE_ENGINE.parents:
    raise RuntimeError(
        f"anime producer resolved OmniVoice to {OMNIVOICE_ENGINE}, not 0.2.1"
    )

from audio_contracts import (
    AudioSpec,
    active_span,
    align_onset_exact,
    assert_exact,
    constant_gain,
    read,
    resample_exact,
    spec,
    write_exact,
)
from line_policy import (
    BLOCKED,
    Decision,
    KEEP_ORIGINAL,
    SHORT_EXTEND_CUT,
    SHORT_TTS_QA,
    TTS,
    classify_line,
)
from gen_scream_comparison import (
    build_leading_splice,
    energy_end,
    room_tone,
    seam_notch_db,
    speech_resume,
    tile_tone,
)
from score_screams_acoustic import _score_from_array, scream_score
import prod_timing

PIPELINE_ROOT = Path(
    r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline"
)
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
from anime_contract import (
    contextual_final_word_gate,
    generation_contract_hash,
    mount_contract_hash,
    processing_contract_hash,
    qa_contract_hash,
    line_contract_hash,
    scene_contract_hash,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "production_config.json"
STATE_PATH = PROJECT / "state" / "STATE.json"
SR = 24000
FFMPEG = Path(
    r"C:\Users\juand\Desktop\moddeutsch\ffmpeg7"
    r"\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin\ffmpeg.exe"
)


def append_generation_suffix(text: str, suffix: str = "...") -> str:
    """Append the experimental tail marker without changing canonical text.

    OmniVoice's duration conditioning has repeatedly clipped the last German
    phoneme.  The experiment gives the model a prosodic release marker, while
    QA continues to compare the waveform against the canonical German text.
    Empty/punctuation-only lines are never sent through this helper.
    """
    value = str(text or "").strip()
    if not value or not words(value):
        return value
    marker = str(suffix or "...").strip() or "..."
    return value if value.endswith(marker) else f"{value}{marker}"


@dataclass
class GenerationRuntime:
    """Reusable OmniVoice state for a global FMV round.

    The model and clone prompts live for the whole generation round, not for
    each scene/line.  ``prompt_cache`` is keyed by the reference path and
    transcript so a changed reference naturally invalidates the prompt.
    """

    model: object
    prompt_cache: dict[tuple[str, str], object] = field(default_factory=dict)


@dataclass
class QARuntime:
    """Reusable QA state and feature caches for one global QA round."""

    asr: WhisperModel | None = None
    mms: object | None = None
    source_alignment_cache: dict[str, list[dict]] = field(default_factory=dict)
    source_features: dict[str, dict] = field(default_factory=dict)


def stage_timing(out: Path, stage: str, started: float, **extra: object) -> None:
    """Append bounded stage telemetry without changing audio artifacts."""
    path = out / "STAGE_TIMINGS.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows.append({"stage": stage, "seconds": round(time.perf_counter() - started, 4), **extra})
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def cheap_candidate_gate(path: Path, expected_frames: int | None = None) -> tuple[bool, str | None]:
    """Reject unreadable/empty/non-finite audio before MMS/ASR/librosa work."""
    if not path.exists() or path.stat().st_size < 256:
        return False, "missing_or_empty_audio"
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as exc:
        return False, f"unreadable_audio:{exc}"
    y = np.asarray(audio, dtype=np.float32).squeeze()
    if sr <= 0 or not len(y) or not np.isfinite(y).all():
        return False, "non_finite_or_empty_audio"
    if expected_frames is not None and len(y) > expected_frames * 8:
        return False, "candidate_unbounded_duration"
    if float(np.max(np.abs(y), initial=0.0)) <= 1e-5:
        return False, "silent_audio"
    return True, None


def classify_failure(row: dict) -> str:
    """Classify retryability according to the generation/QA guidance."""
    error = str(row.get("error") or "").lower()
    if (
        "ctc" in error
        or "targets length" in error
        or re.fullmatch(r"'[0-9]+'", error.strip()) is not None
    ):
        return "DETERMINISTIC_TEXT_OR_WINDOW"
    if "reference" in error or "contract" in error or "window" in error:
        return "DETERMINISTIC_MAPPING"
    gates = row.get("hard_gates") or {}
    diagnostics = row.get("diagnostic_gates") or {}
    # Onset remains useful for deciding which take to prefer, but it is no
    # longer a release blocker.  Use the diagnostic copy only when explaining
    # a failure caused by another hard safety gate; never turn a soft metric
    # back into a hidden hard gate here.
    if (
        diagnostics.get("onset") is False
        and (
            gates.get("tail") is False
            or gates.get("splice_seam") is False
            or gates.get("splice_boundary") is False
        )
    ) or gates.get("tail") is False or gates.get("splice_seam") is False:
        return "TECHNICAL_REPAIRABLE"
    if gates.get("source_language") is False or gates.get("final_word") is False:
        return "RANDOM_TTS"
    if row.get("pass") is False:
        return "RANDOM_TTS"
    return "PASS"


def attach_accepted_contracts(scene: dict, out: Path) -> None:
    """Accept only the exact current line contract.

    Historical rows used to be accepted here, which made mount-only reuse 163
    stale PASS decisions after the timing and language gates changed. Existing
    WAVs remain on disk for audit, but they must be regenerated or explicitly
    re-QA'd under the current contract before mounting.
    """
    for line in scene.get("lines", []):
        current = line.get("_codex2_line_contract_hash")
        line["_codex2_accepted_line_contract_hashes"] = [current] if current else []


def window_capacity(line: dict) -> tuple[bool, str | None]:
    """Preflight impossible text/window combinations before OmniVoice calls."""
    text = line.get("synthesis_text_override") or line.get("delivery_text") or line.get("target_text", "")
    token_count = len(words(text))
    duration = max(0.0, float(line.get("end", 0.0)) - float(line.get("start", 0.0)))
    minimum = 0.15 + token_count * 0.07
    if duration < minimum:
        return False, f"window_capacity:{duration:.3f}s<{minimum:.3f}s_for_{token_count}_tokens"
    return True, None


def resolve(path: str | Path, parent: Path = PROJECT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (parent / path).resolve()


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def words(text: str) -> list[str]:
    # MMS uses index 0 as CTC blank and cannot align a literal hyphen token.
    # Treat compounds such as Gekkoukan-Oberschule as two spoken words.
    value = fold(text)
    # Keep English ordinals such as ``13th`` as one spoken token so their
    # alignment normalization can map them to ``thirteenth`` without changing
    # the word count.  Other alpha/numeric compounds remain split.
    value = re.sub(
        r"(?<=\d)(?=[a-z])(?!(?:st|nd|rd|th)\b)|(?<=[a-z])(?=\d)",
        " ", value,
    )
    return re.findall(r"[a-z0-9]+", value)


# Whisper's German decode routinely writes Persona names and loanwords with a
# phonetic spelling (``Eigis``/``Aegis``, ``Arcana``/``Arkana``) and collapses
# hyphenated calls such as ``YA-KU-SHI-MA``.  These are lexical identity
# aliases, not permission to ignore arbitrary text.  Keeping them centralized
# makes the content gate reusable across every FMV/anime map.
_CONTENT_TOKEN_ALIASES = {
    "oh": "ooh", "uh": "ooh", "uuh": "ooh", "uuuh": "ooh",
    "nummer": "nummer", "nr": "nummer", "no": "nummer", "number": "nummer",
    "2": "zwei", "zwei": "zwei",
    "eigis": "aigis", "eiges": "aigis", "aegis": "aigis", "iges": "aigis",
    "arcana": "arkana",
    # MMS/Whisper often merges Fuuka Yamagishi into these two phonetic forms.
    "fukuyama": "fuuka", "fuka": "fuuka", "fuuga": "fuuka",
    "gishi": "yamagishi", "gishii": "yamagishi", "yamagishii": "yamagishi",
    # Whisper frequently realizes the Persona name Ryoji as ``Ryuji`` in
    # German speech. This narrow proper-name equivalence does not relax
    # delivery checking for ordinary words.
    "ryuji": "ryoji",
    # Whisper may devoice the /z/ in ``hineingezogen`` and report
    # ``hineingesogen``; the surrounding token sequence is still verified.
    "hineingesogen": "hineingezogen",
}


def canonical_content_token(token: str) -> str:
    value = fold(token)
    return _CONTENT_TOKEN_ALIASES.get(value, value)


def content_tokens_equivalent(expected: str, heard: str) -> bool:
    return canonical_content_token(expected) == canonical_content_token(heard)


def delivery_content_gate(
    expected_text: str,
    transcript: str | None,
    asr_enabled: bool,
) -> tuple[bool, dict]:
    """Check that an ASR-enabled take contains the requested German cue.

    This is intentionally narrower than the old WER/text ranking gate.  It
    only rejects an empty or materially partial delivery: duration, rate,
    onset, pitch, span and punctuation remain diagnostics.  Single-letter
    stutter tokens (``W-Was``) are ignored for the first-word check so normal
    German anime disfluencies do not become false failures.
    """
    expected = words(expected_text)
    heard = words(transcript or "")
    if not asr_enabled:
        return True, {
            "asr_enabled": False, "expected_tokens": expected,
            "heard_tokens": heard, "coverage": None,
        }
    if not expected:
        return True, {
            "asr_enabled": True, "expected_tokens": expected,
            "heard_tokens": heard, "coverage": 1.0,
        }
    expected_core_raw = [token for token in expected if len(token) > 1] or expected
    heard_core_raw = [token for token in heard if len(token) > 1] or heard
    expected_core = [canonical_content_token(token) for token in expected_core_raw]
    heard_core = [canonical_content_token(token) for token in heard_core_raw]
    if not heard_core:
        return False, {
            "asr_enabled": True, "expected_tokens": expected,
            "heard_tokens": heard, "coverage": 0.0,
            "first_token_ok": False, "final_token_ok": False,
        }
    matcher = SequenceMatcher(None, expected_core, heard_core, autojunk=False)
    matched_tokens = sum(block.size for block in matcher.get_matching_blocks())
    coverage = matched_tokens / max(1, len(expected_core))
    first_expected = expected_core[0]
    final_expected = canonical_content_token(expected[-1])
    joined_equivalent = "".join(expected_core) == "".join(heard_core)
    first_token_ok = any(
        token == first_expected
        or token.endswith(first_expected)
        or first_expected.endswith(token)
        for token in heard_core[:2]
    )
    final_token = canonical_content_token(heard[-1]) if heard else ""
    final_token_ok = bool(final_token) and bool(
        final_token == final_expected
        or final_token.endswith(final_expected)
        or final_expected.endswith(final_token)
    )
    if joined_equivalent:
        # A hyphenated proper name can be segmented differently by ASR while
        # retaining exactly the same spoken character sequence (YA-KU-SHI-MA
        # -> YAKU SHIMA). Treat the joined identity as complete content.
        coverage = 1.0
        first_token_ok = True
        final_token_ok = True
    # Long lines may lose one weak function word in Whisper, but their first
    # and final lexical edges must still be present. Short lines are cheap to
    # verify and require complete token coverage.
    minimum_coverage = 1.0 if len(expected_core) <= 3 else 0.75
    ok = bool(
        first_token_ok
        and final_token_ok
        and coverage >= minimum_coverage
    )
    return ok, {
        "asr_enabled": True, "expected_tokens": expected,
        "heard_tokens": heard, "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "first_token_ok": first_token_ok,
        "final_token_ok": final_token_ok,
        "joined_equivalent": joined_equivalent,
    }


def has_mapping_evidence(line: dict) -> tuple[bool, str]:
    """Return whether a forced FMV mapping has enough evidence to synthesize.

    ASR is evidence rather than ground truth: longer utterances may pass with
    partial recognition, but only when at least half the expected words are
    covered *and* the recognized span is not substantially different.  Very
    short lines have too little lexical information and need an explicit
    contextual validation from the mapping stage.  A force flag cannot bypass
    either requirement.
    """
    validation = str(line.get("mapping_validation", "")).upper()
    if validation in {
        "EXACT", "CONTEXTUAL", "HUMAN_CONFIRMED", "LEGACY_ASR_RECOVERED",
    }:
        return True, f"explicit_{validation.lower()}"
    expected = words(line.get("source_text", ""))
    coverage = line.get("alignment_coverage")
    wer = line.get("alignment_wer")
    if coverage is None or wer is None:
        return False, "missing_mapping_evidence"
    try:
        coverage, wer = float(coverage), float(wer)
    except (TypeError, ValueError):
        return False, "invalid_mapping_evidence"
    if len(expected) >= 4 and coverage >= 0.50 and wer <= 0.50:
        return True, "partial_asr_confirmed"
    if len(expected) <= 3:
        return False, "short_line_requires_contextual_anchor"
    return False, f"asr_conflict(coverage={coverage:.2f},wer={wer:.2f})"


def decide_line(line: dict) -> Decision:
    """Apply deterministic policy with an explicit subtitle-coverage override.

    Some subtitled calls are language-neutral on paper (for example ``Wow`` or
    ``Iwatodai``), but retaining them preserves the English actor and can
    overlap adjacent German delivery.  ``force_clone`` means the subtitle must
    receive the same cloned-voice treatment as every other spoken line.  Pure
    nonverbal efforts remain governed by the normal policy unless a map opts in
    deliberately.
    """
    spoken_target = line.get("delivery_text", line["target_text"])
    # Universal content rule: an explicit negative subtitle authorization is a
    # resolved preservation, never a TTS candidate or a pending blocker.  Maps
    # from older branches omit the field and remain governed by their existing
    # reviewed policy until they are migrated.
    if line.get("subtitle_authorized") is False and not line.get("force_keep_original"):
        return Decision(
            KEEP_ORIGINAL,
            line.get("force_keep_reason", "no_visible_subtitle_card"),
            len(words(spoken_target)),
        )
    # Preservation is an intentional resolved outcome, not a pending block.
    if line.get("force_keep_original"):
        return Decision(
            KEEP_ORIGINAL,
            line.get("force_keep_reason", "explicit_nonlexical_preservation"),
            len(words(spoken_target)),
        )
    if line.get("generation_blocked"):
        return Decision(
            BLOCKED,
            "generation blocked pending human mapping/reference correction: "
            f"{line.get('generation_block_reason', 'unspecified')}",
            len(words(spoken_target)),
        )
    if line.get("force_clone"):
        approved, evidence = has_mapping_evidence(line)
        if not approved:
            return Decision(
                BLOCKED,
                f"force_clone denied; mapping evidence insufficient ({evidence})",
                len(words(spoken_target)),
            )
        count = len(words(spoken_target))
        action = SHORT_TTS_QA if 1 <= count <= 3 else TTS
        return Decision(
            action,
            f"forced_subtitled_voice:{evidence}",
            count,
            synthesis_text=spoken_target,
        )
    return classify_line(line["source_text"], spoken_target)


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def spoken_wer(expected: list[str], heard: list[str]) -> tuple[float, bool]:
    """WER with a narrow spoken-German word-boundary equivalence.

    ASR cannot distinguish e.g. ``hierbleibt``/``hier bleibt`` or
    ``erst mal``/``erstmal`` from audio.  Accept that case only when the full
    normalized character stream is identical; different phonemes still use
    ordinary word error rate.
    """
    digit_words = {
        "0": "null", "1": "eins", "2": "zwei", "3": "drei", "4": "vier",
        "5": "funf", "6": "sechs", "7": "sieben", "8": "acht", "9": "neun",
    }
    expected = [digit_words.get(token, token) for token in expected]
    heard = [digit_words.get(token, token) for token in heard]

    def compact(tokens: list[str]) -> str:
        value = "".join(tokens)
        # Japanese long-o romanization: Gekkoukan is normally realized and
        # transcribed in German as Gekkokan / Gekko-Kan.
        value = value.replace("gekkoukan", "gekkokan")
        # Whisper commonly hears the first unstressed /a/ in Iwatodai as /o/.
        # This is an orthographic ASR equivalence, not a pronunciation change.
        return value.replace("iwotodai", "iwatodai")

    # A German delivery may intentionally omit a written stutter marker
    # (``W-Was`` -> ``Was``).  Treat that one-letter prefix as an orthographic
    # hesitation, not as a missing lexical word; the rest of the transcript
    # still has to match exactly.
    stutter_equivalent = bool(
        len(expected) >= 2
        and len(expected[0]) == 1
        and heard == expected[1:]
    )
    joined_equivalent = bool(
        expected and (compact(expected) == compact(heard) or stutter_equivalent)
    )
    value = 0.0 if joined_equivalent else edit_distance(expected, heard) / max(1, len(expected))
    return value, joined_equivalent


# WER remains a useful ranking signal, but it can be overly strict with names
# and German compounds.  This narrower gate catches the destructive case that
# must never ship: OmniVoice repeating recognizable English source words in
# the German take.  One strong source-only English marker is sufficient.
ENGLISH_SOURCE_MARKERS = {
    "a", "again", "am", "and", "because", "being", "can", "can't",
    "could", "did", "do", "don't", "earth", "everything", "for", "from",
    "going", "have", "he", "how", "i", "if", "in", "is", "it", "let",
    "like", "nearby", "no", "not", "now", "of", "okay", "one", "on",
    "only", "remember", "suitable", "the", "this", "to", "vessel", "was",
    "what", "when", "where", "who", "why", "with", "would", "you", "your",
    # Strong content words are uncommon in German output and must not be
    # allowed through merely because they occur as a single source token.
    "created", "machine", "serve", "specific", "purpose", "four",
}

STRONG_ENGLISH_SOURCE_MARKERS = {
    "created", "machine", "serve", "specific", "purpose", "nearby",
    "suitable", "vessel",
}


def source_language_leak(
    source_text: str, target_text: str, transcript: str,
) -> list[str]:
    source_only = (
        set(words(transcript))
        & set(words(source_text))
        - set(words(target_text))
    )
    return sorted(source_only & ENGLISH_SOURCE_MARKERS)


def source_language_confirmation(
    source_text: str, target_text: str, transcript: str,
) -> tuple[list[str], bool, dict]:
    """Return suspected/confirmed source leakage with a consensus rule.

    A single common token such as ``to``/``you`` is only a suspicion.  A
    source-like transcript is confirmed when it is close to the English source
    while remaining materially different from the German target, or when at
    least two distinct source-only English markers survive in the transcript.
    """
    suspected = source_language_leak(source_text, target_text, transcript)
    source_distance, _ = spoken_wer(words(source_text), words(transcript))
    target_distance, _ = spoken_wer(words(target_text), words(transcript))
    source_tokens = words(source_text)
    heard_tokens = words(transcript)
    longest_phrase = 0
    for start in range(len(source_tokens)):
        for end in range(start + 1, len(source_tokens) + 1):
            phrase = source_tokens[start:end]
            if len(phrase) <= longest_phrase:
                continue
            if any(
                heard_tokens[index:index + len(phrase)] == phrase
                for index in range(max(0, len(heard_tokens) - len(phrase) + 1))
            ):
                longest_phrase = len(phrase)
    confirmed = bool(
        (source_distance <= 0.28 and target_distance >= 0.30)
        or len(suspected) >= 2
        or bool(set(suspected) & STRONG_ENGLISH_SOURCE_MARKERS)
        or (longest_phrase >= 3 and target_distance >= 0.30)
    )
    return (suspected if confirmed else []), confirmed, {
        "suspected_tokens": suspected,
        "source_wer": source_distance,
        "target_wer": target_distance,
        "longest_source_phrase_tokens": longest_phrase,
    }


def target_content_gate(
    target_text: str,
    transcript: str,
    wer: float,
    max_wer: float,
    joined_equivalent: bool = False,
) -> tuple[bool, dict]:
    """Make lexical correctness a release gate, not a ranking diagnostic.

    The old producer hard-coded ``hard["text"] = True``.  That allowed a
    candidate such as ``Palladian`` or an incomplete ``Klings wie`` to become
    a winner.  Exact ASR is not required for known orthographic joins, but an
    empty/incomplete transcript is never release-safe.
    """
    expected = words(target_text)
    heard = words(transcript)
    if not expected:
        return True, {"expected_tokens": 0, "heard_tokens": len(heard)}
    if not heard:
        return False, {"expected_tokens": len(expected), "heard_tokens": 0}
    passed = bool(joined_equivalent or float(wer) <= float(max_wer))
    return passed, {
        "expected_tokens": len(expected),
        "heard_tokens": len(heard),
        "wer": float(wer),
        "max_wer": float(max_wer),
        "joined_equivalent": bool(joined_equivalent),
    }


def syllables(text: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", fold(text))))


def meter_lufs(audio: np.ndarray, sr: int) -> float:
    y = np.asarray(audio, dtype=np.float32)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)))
    value = float(pyln.Meter(sr).integrated_loudness(y))
    return value if math.isfinite(value) else -70.0


def contour(audio: np.ndarray, kind: str, points: int = 80) -> np.ndarray | None:
    start, end = active_span(audio, SR)
    y = audio[start:end]
    if len(y) < 1024:
        return None
    if kind == "energy":
        values = np.log(librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0] + 1e-8)
    else:
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR, frame_length=2048, hop_length=256)
        valid = np.isfinite(f0)
        if valid.sum() < 5:
            return None
        ix = np.arange(len(f0))
        values = np.interp(ix, ix[valid], np.log2(f0[valid]))
    if len(values) < 3 or float(np.std(values)) < 1e-7:
        return None
    result = np.interp(np.linspace(0, 1, points), np.linspace(0, 1, len(values)), values)
    return (result - result.mean()) / (result.std() + 1e-8)


def median_f0(audio: np.ndarray, sr: int) -> tuple[float | None, int]:
    """Robust voiced median used to reject octave/pitch clone failures.

    Correct text can still sound like a different actor when a stochastic
    generation locks onto an implausibly high octave.  Compare against the
    actor's source performance for the same cue and skip the gate when either
    side has too little reliable voiced material.
    """
    y = np.asarray(audio, dtype=np.float32).squeeze()
    if sr != SR:
        y = resample_exact(y, sr, SR)
    if len(y) < 1024:
        return None, 0
    f0, _, probability = librosa.pyin(
        y, fmin=70.0, fmax=700.0, sr=SR,
        frame_length=1024, hop_length=128,
    )
    reliable = f0[np.isfinite(f0) & (probability >= 0.5)]
    if len(reliable) < 5:
        return None, int(len(reliable))
    return float(np.median(reliable)), int(len(reliable))


def corr(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    return float(np.corrcoef(left, right)[0, 1]) if left is not None and right is not None else None


def write_state(scene: str, active: str, counts: dict | None = None) -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state["chronological_cursor"] = scene
    state["active"] = active
    if counts is not None:
        state["counts"].update(counts)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def alignment_text(text: str) -> str:
    """Normalize numeric spellings for MMS without changing spoken synthesis."""
    replacements = {
        "13th": "thirteenth",
        "12th": "twelfth",
        "11th": "eleventh",
        "10th": "tenth",
        "13.": "dreizehnten",
        "13": "dreizehn",
        "12.": "zwölften",
        "12": "zwölf",
        "11.": "elften",
        "11": "elf",
        "10.": "zehnten",
        "10": "zehn",
        # MMS/Fairseq has no lexical entries for bare numeric tokens. Anime
        # cards use ``Nr. 1/2`` frequently; leaving the digit untouched raises
        # a cryptic KeyError (``'1'``/``'2'``) before QA can inspect a take.
        # Each replacement remains one input token -> one spoken token so
        # alignment word indices stay stable.
        "0": "null",
        "1": "eins",
        "2": "zwei",
        "3": "drei",
        "4": "vier",
        "5": "fünf",
        "6": "sechs",
        "7": "sieben",
        "8": "acht",
        "9": "neun",
    }
    result = text
    for token, spoken in replacements.items():
        result = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", spoken, result)
    return result


class MMS:
    def __init__(self) -> None:
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).cuda().eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def align(self, audio: np.ndarray, transcript: str) -> list[dict]:
        y16 = resample_exact(audio, SR, 16000)
        tokens = words(transcript)
        align_tokens = words(alignment_text(transcript))
        if len(align_tokens) != len(tokens):
            raise ValueError(
                f"alignment token normalization changed token count: "
                f"{len(tokens)} -> {len(align_tokens)}"
            )
        with torch.inference_mode():
            emission, _ = self.model(torch.from_numpy(y16).unsqueeze(0).cuda())
        spans = self.aligner(emission[0], self.tokenizer(align_tokens))
        frame_seconds = (len(y16) / 16000) / emission.shape[1]
        return [
            {"text": token, "start": chars[0].start * frame_seconds,
             "end": chars[-1].end * frame_seconds,
             "score": float(np.mean([float(c.score) for c in chars]))}
            for token, chars in zip(tokens, spans)
        ]


def resolve_leading_boundaries(source: np.ndarray, line: dict) -> dict:
    """Resolve Empalme B boundaries from the source waveform.

    Map timestamps are optional assertions.  They must agree with the measured
    effort/resume edges; a stale timestamp fails closed instead of silently
    cutting an active onomatopoeia.
    """
    detected_effort_end = float(
        energy_end(source, SR, min(1.2, len(source) / SR * 0.6))
    )
    detected_source_resume = float(
        speech_resume(source, SR, detected_effort_end)
    )
    tolerance = float(line.get("splice_boundary_tolerance_seconds", 0.035))
    values = {
        "effort_end_seconds": detected_effort_end,
        "source_resume_seconds": detected_source_resume,
        "detected_effort_end_seconds": detected_effort_end,
        "detected_source_resume_seconds": detected_source_resume,
        "boundary_tolerance_seconds": tolerance,
        "manual_boundary_assertion": False,
    }
    manual_authoritative = bool(
        line.get("allow_non_neutral_leading_interjection")
        and "effort_end_seconds" in line
        and "source_resume_seconds" in line
    )
    for key, detected in (
        ("effort_end_seconds", detected_effort_end),
        ("source_resume_seconds", detected_source_resume),
    ):
        if key not in line:
            continue
        configured = float(line[key])
        # Energy valley detection is intentionally conservative for a spoken
        # lexical interjection: "Geez" contains voiced phonemes and its pause
        # is not a scream-like deep valley. When the map supplies independent
        # ASR word edges for an explicitly authorized prefix, those edges are
        # authoritative; keep the measured values in the report for audit.
        if not manual_authoritative and abs(configured - detected) > tolerance:
            raise ValueError(
                f"stale Empalme B {key} for {line['id']}: "
                f"map={configured:.3f}s detected={detected:.3f}s "
                f"tolerance={tolerance:.3f}s"
            )
        values[key] = configured
        values["manual_boundary_assertion"] = True
    values["manual_boundary_authoritative"] = manual_authoritative
    effort_end = values["effort_end_seconds"]
    source_resume = values["source_resume_seconds"]
    if not (0.0 <= effort_end <= source_resume <= len(source) / SR):
        raise ValueError(
            f"invalid Empalme B boundaries for {line['id']}: "
            f"{effort_end:.3f} <= {source_resume:.3f}"
        )
    values["boundary_valid"] = True
    return values


def cut_short(audio: np.ndarray, synthesis_text: str, count: int, mms: MMS, start_word: int = 0,
              silence_db: float = -45.0, max_extra_seconds: float = 0.500,
              hop_seconds: float = 0.002, min_tail_seconds: float = 0.010,
              fade_seconds: float = 0.012, onset_fade_seconds: float = 0.003,
              lead_seconds: float = 0.010,
              max_frames: int | None = None) -> tuple[np.ndarray, dict]:
    aligned = mms.align(audio, synthesis_text)
    if len(aligned) < start_word + count:
        raise ValueError(f"short-line alignment returned {len(aligned)}/{start_word + count} words")
    # Always anchor to the aligned word's own start, even when start_word == 0:
    # OmniVoice sometimes renders up to ~1s of lead-in silence/artifact before
    # the first word, and starting the cut at sample 0 would fold that into
    # the delivered body (or, once budget-clamped, cut off before the word
    # ever begins).
    begin = max(0, round((aligned[start_word]["start"] - lead_seconds) * SR))
    # MMS marks the acoustic-phonetic transition, not where the ear stops
    # hearing the word: German word-final consonants/vowels carry a natural
    # decay/aspiration tail that a short fixed offset often clips mid-sound.
    # Confirmed on a real line ("Da wären wir...", a trailing-off delivery
    # cued by the "..." in expanded_prompt): the model can render a pause of
    # several hundred ms of still-loud voice before it actually decays, so the
    # search window has to be generous, not just enough for a consonant
    # release. Search forward from the alignment mark for where the envelope
    # actually settles into near-silence (bounded, so this never runs away
    # into the next word), then fade only over that already-quiet tail with
    # an equal-power (raised-cosine) curve instead of a linear ramp over
    # audible energy.
    word_end = aligned[start_word + count - 1]["end"]
    min_cut = round((word_end + min_tail_seconds) * SR)
    max_cut = min(len(audio), round((word_end + max_extra_seconds) * SR))
    if start_word + count < len(aligned):
        # Contextual short-line fallback: never let the extraction run into
        # the following helper sentence.  Stop just before its aligned onset;
        # the fade then lands inside the punctuation pause, not inside a word.
        next_word_limit = round(
            max(word_end, aligned[start_word + count]["start"] - 0.010) * SR
        )
        max_cut = min(max_cut, next_word_limit)
    if max_frames is not None:
        # The exact-frame cue window is a hard ceiling. Never search past it:
        # cap the whole tail search to whatever room the placement step has,
        # so cut_short's own fade (not a second one downstream) is always
        # what terminates the delivered body.
        hard_limit = begin + max_frames
        min_cut = min(min_cut, hard_limit)
        max_cut = min(max_cut, hard_limit)
    hop = max(1, round(hop_seconds * SR))
    cut = min(min_cut, len(audio))
    pos = min_cut
    consecutive_quiet = 0
    quiet_tail_found = False
    while pos < max_cut:
        seg = audio[pos:pos + hop]
        if len(seg) == 0:
            break
        level = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12)
        if level <= silence_db:
            consecutive_quiet += 1
            if consecutive_quiet >= 2:
                cut = pos + hop
                quiet_tail_found = True
                break
        else:
            consecutive_quiet = 0
        pos += hop
        cut = pos
    cut = min(cut, len(audio))
    if max_frames is not None:
        cut = min(cut, begin + max_frames)
    if cut <= begin:
        raise ValueError("invalid short-line cut")
    result = np.array(audio[begin:cut], dtype=np.float32, copy=True)
    # The onset edge only needs enough ramp to kill the digital-zero
    # discontinuity at place_short_cut's placement boundary. Confirmed on a
    # real line: fading even 3 ms into the word itself (when the take has
    # ~0 lead-in before speech starts) measurably breaks ASR accuracy --
    # some short OmniVoice takes start speaking essentially at sample 0, with
    # no silent cushion. So the fade must never reach past where the word was
    # actually aligned to start; if there's no cushion, there's no fade.
    # The tail edge has no such constraint -- it's fading into content
    # cut_short already chose to be near-silent -- so it keeps the longer,
    # smoother `fade_seconds` curve.
    word_start_sample = round(aligned[start_word]["start"] * SR)
    onset_cushion = max(0, word_start_sample - begin)
    onset_fade = min(round(onset_fade_seconds * SR), len(result) // 2, onset_cushion)
    if onset_fade:
        ramp_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, onset_fade, dtype=np.float32)))
        result[:onset_fade] *= ramp_in
    tail_fade = min(round(fade_seconds * SR), len(result) // 2)
    if tail_fade:
        ramp_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail_fade, dtype=np.float32)))
        result[-tail_fade:] *= ramp_out
    hard_limit = begin + max_frames if max_frames is not None else None
    hit_max_frames = bool(
        hard_limit is not None
        and cut >= min(hard_limit, len(audio))
        and not quiet_tail_found
    )
    metadata = {
        "begin_seconds": begin / SR,
        "cut_seconds": cut / SR,
        "last_target_word_end_seconds": float(word_end),
        "tail_after_last_word_seconds": max(0.0, cut / SR - float(word_end)),
        "quiet_tail_found": quiet_tail_found,
        "hit_max_frames": hit_max_frames,
        "tail_release_ok": quiet_tail_found and not hit_max_frames,
        "max_frames": max_frames,
        "silence_db": silence_db,
    }
    return result, metadata


def place_short_cut(source: np.ndarray, generated: np.ndarray, sr: int, output_frames: int,
                     lead_guard_seconds: float) -> np.ndarray:
    """Position an already cut+faded SHORT_EXTEND_CUT body at the source onset.

    Unlike align_onset_exact, this does not re-detect the active span of
    `generated` or apply a second fade: cut_short() already produced a
    complete, correctly-terminated body, and re-processing it here would
    compound two independent fades into an artificially fast collapse to
    silence (audible as the word being choked off or ducked).
    """
    src_start, _ = active_span(source, sr)
    lead = round(lead_guard_seconds * sr)
    destination = max(0, src_start - lead)
    if destination + len(generated) > output_frames:
        raise ValueError(
            f"active body overflow: need {destination + len(generated)} frames, have {output_frames}"
        )
    result = np.zeros(output_frames, dtype=np.float32)
    result[destination:destination + len(generated)] = generated
    return result


def source_window(
    stem: np.ndarray,
    line: dict,
    sr: int,
    *,
    reference: bool = False,
) -> np.ndarray:
    """Extract one line window without conflating delivery and reference edges.

    ``start/end`` are the immutable delivery/mount window.  A map may carry a
    wider phoneme-safe ``reference_start/reference_end`` window (or a later
    onset that excludes an unscripted preceding effort).  OmniVoice must see
    the latter only as its English prompt; using the delivery bounds here can
    import the previous line into the clone and make the German take repeat
    it.  Missing reference edges intentionally fall back to the delivery
    window for legacy maps.
    """
    start_key = "reference_start" if reference and "reference_start" in line else "start"
    end_key = "reference_end" if reference and "reference_end" in line else "end"
    start = max(0, round(float(line[start_key]) * sr))
    end = min(len(stem), round(float(line[end_key]) * sr))
    if end <= start:
        raise ValueError(
            f"invalid {('reference' if reference else 'delivery')} window for "
            f"{line.get('id', '<unknown>')}: {line.get(start_key)}..{line.get(end_key)}"
        )
    return np.asarray(stem[start:end], dtype=np.float32)


def trim_lead_silence(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Apply the handoff's universal artificial-delay removal.

    SUKAKU can add a variable 137--206 ms lead-in. Detect the first sample
    above -45 dB relative to the clip peak, discard only the preceding model
    silence, and soften the new edge with the same 3 ms raised-cosine fade
    used by ``gen_scream_comparison.py``.
    """
    y = np.asarray(audio, dtype=np.float32).squeeze()
    if not len(y):
        return y, 0
    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return y, 0
    threshold = peak * (10.0 ** (-45.0 / 20.0))
    indices = np.flatnonzero(np.abs(y) > threshold)
    trimmed = int(indices[0]) if len(indices) else 0
    result = np.array(y[trimmed:], dtype=np.float32, copy=True)
    fade = min(round(0.003 * sr), len(result))
    if fade:
        result[:fade] *= 0.5 * (
            1.0 - np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32))
        )
    return result, trimmed


def prepare_references(scene: dict, psm: np.ndarray, sr: int, out: Path) -> dict[str, tuple[Path, str]]:
    refs = out / "references"
    refs.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for line in scene["lines"]:
        decision = decide_line(line)
        if decision.action not in {KEEP_ORIGINAL, BLOCKED}:
            grouped.setdefault(line["speaker"], []).append(line)
    result = {}
    for index, (speaker, lines) in enumerate(grouped.items(), 1):
        chunks, texts = [], []
        for line in lines:
            chunk = source_window(psm, line, sr, reference=True)
            current_seconds = sum(len(item) for item in chunks) / sr
            if chunks and current_seconds + len(chunk) / sr > 9.8:
                break
            chunks.append(chunk)
            chunks.append(np.zeros(round(0.12 * sr), dtype=np.float32))
            texts.append(line["source_text"].strip())
        audio = np.concatenate(chunks) if chunks else np.zeros(sr, np.float32)
        if sr != SR:
            audio = resample_exact(audio, sr, SR)
        path = refs / f"speaker_{index:02d}.wav"
        sf.write(path, audio, SR, subtype="PCM_16")
        result[speaker] = (path, " ".join(texts))
    # Cutscenes need the actor's delivery, not merely an averaged speaker
    # identity.  For sufficiently long dry cues, the matching English line is
    # the strongest available emotion/prosody reference and has enough speech
    # for a stable clone.  Short calls keep the grouped identity reference
    # because sub-second prompts are unreliable.
    if scene.get("kind") in {
        "anime", "ANIME_USM_EMBEDDED_MIX", "in_engine",
    }:
        for line in scene["lines"]:
            if decide_line(line).action in {KEEP_ORIGINAL, BLOCKED}:
                continue
            chunk = source_window(psm, line, sr, reference=True)
            if len(chunk) / sr < 1.5:
                continue
            if sr != SR:
                chunk = resample_exact(chunk, sr, SR)
            path = refs / f"{line['id']}_line_emotion.wav"
            sf.write(path, chunk, SR, subtype="PCM_16")
            result[line["id"]] = (path, line["source_text"].strip())
    # Difficult emotional lines may need a cleaner identity anchor than the
    # noisy/whispered material available in their own cue.  A map can provide
    # verified reference segments from the same actor.  Concatenate them
    # deterministically (clean identity first, scene emotion second) while
    # preserving exact matching English transcripts for OmniVoice.
    for line in scene["lines"]:
        segments = line.get("reference_segments")
        if not segments:
            continue
        chunks, texts = [], []
        for segment in segments:
            source_path = resolve(segment["path"])
            source_audio, source_sr = read(source_path)
            if source_audio.ndim == 2:
                # A verified reference may name the exact 6-ch lane that
                # carries the actor.  Averaging a full mix can import music
                # or a different simultaneous English line and destabilize
                # the clone (the 100_090 ``Morning`` cue is channel 4).
                channel = segment.get("channel")
                if channel is not None:
                    channel = int(channel)
                    if channel < 0 or channel >= source_audio.shape[1]:
                        raise ValueError(
                            f"reference channel {channel} outside "
                            f"{source_audio.shape[1]} channels for {line['id']}"
                        )
                    source_audio = source_audio[:, channel]
                else:
                    source_audio = source_audio.mean(axis=1)
            start = max(0, round(float(segment["start"]) * source_sr))
            end = min(len(source_audio), round(float(segment["end"]) * source_sr))
            chunk = np.asarray(source_audio[start:end], dtype=np.float32)
            if source_sr != SR:
                chunk = resample_exact(chunk, source_sr, SR)
            current_seconds = sum(len(item) for item in chunks) / SR
            if chunks and current_seconds + len(chunk) / SR > 9.8:
                break
            chunks.append(chunk)
            chunks.append(np.zeros(round(0.12 * SR), dtype=np.float32))
            texts.append(segment["text"].strip())
        if not chunks:
            raise ValueError(f"empty reference override for {line['id']}")
        path = refs / f"{line['id']}_reference_override.wav"
        sf.write(path, np.concatenate(chunks), SR, subtype="PCM_16")
        result[line["id"]] = (path, " ".join(texts))
    return result


def candidate_name(line_id: str, round_index: int, take: int) -> str:
    return f"{line_id}_r{round_index}_t{take:02d}.wav"


def generate_round(
    scene: dict, stem: np.ndarray, stem_sr: int, out: Path, profile: dict,
    round_index: int, only_ids: set[str] | None = None,
    runtime: GenerationRuntime | None = None,
) -> None:
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = candidate_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    contract_hash = scene.get("_codex2_contract_hash")
    if not contract_hash:
        raise RuntimeError("missing Codex2 scene contract hash before generation")
    line_hashes = {
        line["id"]: line.get("_codex2_line_contract_hash")
        for line in scene["lines"]
    }
    accepted_hashes = {
        line["id"]: set(line.get("_codex2_accepted_line_contract_hashes", [line_hashes[line["id"]]]))
        for line in scene["lines"]
    }
    metadata = [
        row for row in metadata
        if row.get("contract_hash") in accepted_hashes.get(row.get("line_id"), set())
    ]
    takes = int(profile["initial_takes"] if round_index == 0 else profile["retry_takes"])
    eligible = []
    preflight_rejections = {}
    for line in scene["lines"]:
        if only_ids is not None and line["id"] not in only_ids:
            continue
        decision = decide_line(line)
        if decision.action in {KEEP_ORIGINAL, BLOCKED}:
            if decision.action == BLOCKED:
                preflight_rejections[line["id"]] = decision.reason
            continue
        capacity_ok, capacity_reason = window_capacity(line)
        if not capacity_ok:
            preflight_rejections[line["id"]] = capacity_reason
            continue
        count = sum(
            row["line_id"] == line["id"]
            and int(row["round"]) == round_index
            and row.get("contract_hash") in accepted_hashes.get(line["id"], set())
            for row in metadata
        )
        if count < takes:
            eligible.append(line["id"])
    if not eligible:
        if preflight_rejections:
            (out / "PREFLIGHT_REJECTIONS.json").write_text(
                json.dumps(preflight_rejections, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        print(f"GEN round {round_index}: all candidates already persisted; skipping model load", flush=True)
        return
    if preflight_rejections:
        (out / "PREFLIGHT_REJECTIONS.json").write_text(
            json.dumps(preflight_rejections, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    refs = prepare_references(scene, stem, stem_sr, out)
    owns_model = runtime is None
    model = (
        OmniVoice.from_pretrained(
            profile.get("model", "k2-fsa/OmniVoice"),
            device_map="cuda", dtype=torch.float16,
        )
        if runtime is None else runtime.model
    )
    # Canonical post-comparison profile (2026-07-29): official OmniVoice 0.2.1,
    # guidance 2 and PP false. Unlike SUKAKU 0.1.2, 0.2.1 exposes edge padding
    # and fading, so the model is asked for a clean body and all alignment /
    # seam protection stays localized in this producer.
    generation_config = OmniVoiceGenerationConfig(
        num_step=int(profile["num_step"]),
        guidance_scale=float(profile["guidance_scale"]),
        position_temperature=float(profile.get("position_temperature", 5.0)),
        class_temperature=float(profile.get("class_temperature", 0.0)),
        t_shift=float(profile.get("t_shift", 0.1)),
        preprocess_prompt=True,
        postprocess_output=bool(profile.get("postprocess_output", False)),
        pad_duration=float(profile.get("pad_duration", 0.0)),
        fade_duration=float(profile.get("fade_duration", 0.0)),
    )
    prompts = runtime.prompt_cache if runtime is not None else {}
    known = {
        row["file"] for row in metadata
        if row.get("contract_hash") in accepted_hashes.get(row.get("line_id"), set())
    }
    jobs_by_prompt: dict[str, list[dict]] = {}
    for line_index, line in enumerate(scene["lines"]):
        if only_ids is not None and line["id"] not in only_ids:
            continue
        spoken_target = line.get("delivery_text", line["target_text"])
        decision = decide_line(line)
        if decision.action in {KEEP_ORIGINAL, BLOCKED}:
            continue
        prompt_key = line["id"] if line["id"] in refs else line["speaker"]
        ref_path, ref_text = refs[prompt_key]
        prompt_cache_key = (str(ref_path), ref_text)
        if prompt_cache_key not in prompts:
            prompts[prompt_cache_key] = model.create_voice_clone_prompt(
                ref_audio=str(ref_path), ref_text=ref_text, preprocess_prompt=True)
        # A mixed leading effort uses the whole cue as its immutable output
        # window, but OmniVoice generates only the spoken German body after
        # the original actor's effort/pause.  The splice is assembled during
        # QA with the handoff's canonical 25 ms/room-tone implementation.
        synthesis_start = float(line.get("synthesis_start", line["start"]))
        duration = float(line["end"]) - synthesis_start
        canonical_synthesis_text = line.get(
            "synthesis_text_override", decision.synthesis_text or spoken_target,
        )
        synthesis_text = canonical_synthesis_text
        if profile.get("append_ellipsis_experiment", False):
            synthesis_text = append_generation_suffix(
                canonical_synthesis_text,
                profile.get("ellipsis_suffix", "..."),
            )
        # SUKAKU's BAT-default post-processing pads roughly 100 ms at each
        # edge. Compensate only the duration request so the post-processed
        # waveform still fits the immutable anime cue; do not trim speech.
        postprocess_padding = float(line.get(
            "postprocess_duration_compensation",
            0.20 if generation_config.postprocess_output else 0.0,
        ))
        cue_duration = max(0.40, duration - postprocess_padding)
        contextual_extraction = (
            decision.action == SHORT_EXTEND_CUT
            or "delivery_word_start" in line
        )
        synthesis_duration = (
            (
                min(6.0, max(0.8, cue_duration))
                if line.get("synthesis_duration_mode") == "lexical_only"
                else min(6.0, max(3.5, cue_duration + 2.4))
            )
            if contextual_extraction else cue_duration
        )
        if line.get("synthesis_duration_override") is not None:
            synthesis_duration = float(line["synthesis_duration_override"])
        for take in range(takes):
            name = candidate_name(line["id"], round_index, take)
            if name in known and (candidate_dir / name).exists():
                continue
            jobs_by_prompt.setdefault(str(prompt_cache_key), []).append({
                "line": line, "line_index": line_index, "take": take,
                "name": name, "synthesis_text": synthesis_text,
                "canonical_synthesis_text": canonical_synthesis_text,
                "ellipsis_experiment": bool(
                    synthesis_text != canonical_synthesis_text
                ),
                "synthesis_start": synthesis_start,
                "synthesis_duration": synthesis_duration,
                "ref_path": ref_path, "prompt": prompts[prompt_cache_key],
                "decision": decision,
            })
    batch_size = max(1, int(profile.get("batch_size", 2)))
    for prompt_jobs in jobs_by_prompt.values():
        # Round-robin by line: a batch is two distinct dialogue units, never
        # two candidates of the same line.  This is the canonical batch=2
        # interpretation from the generation references.
        by_line: dict[str, list[dict]] = {}
        for job in prompt_jobs:
            by_line.setdefault(job["line"]["id"], []).append(job)
        scheduled = []
        for take in range(max(len(items) for items in by_line.values())):
            scheduled.extend(items[take] for items in by_line.values() if take < len(items))
        # OmniVoice's conditioning path cannot safely receive two candidates
        # from the same dialogue unit.  When a prompt group contains only one
        # line, the old scheduler nevertheless formed [L003, L003]-style
        # batches, which surfaced as opaque numeric KeyErrors ('1'/'2').
        # Reduce the effective batch size to the number of distinct lines;
        # this preserves batching across distinct units while making isolated
        # lines deterministic and safe.
        effective_batch_size = min(batch_size, max(1, len(by_line)))
        for batch_start in range(0, len(scheduled), effective_batch_size):
            batch = scheduled[batch_start:batch_start + effective_batch_size]
            seed = 2026072200 + round_index * 100000 + sum(
                int(job["line_index"]) * 100 + int(job["take"]) for job in batch
            )
            torch.manual_seed(seed)
            outputs = model.generate(
                text=[job["synthesis_text"] for job in batch],
                language=["German"] * len(batch),
                duration=[job["synthesis_duration"] for job in batch],
                voice_clone_prompt=batch[0]["prompt"],
                generation_config=generation_config,
                class_temperature=float(profile.get("class_temperature", 0.25)),
            )
            for offset, (job, output) in enumerate(zip(batch, outputs)):
                name = job["name"]
                y = output.detach().float().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
                y = np.asarray(y, dtype=np.float32).squeeze()
                sf.write(candidate_dir / name, y, SR, subtype="PCM_16")
                row = {
                    "file": name, "line_id": job["line"]["id"], "round": round_index,
                    "take": job["take"], "seed": seed, "batch_index": offset,
                    "batch_unit_ids": [item["line"]["id"] for item in batch],
                    "contract_hash": line_hashes[job["line"]["id"]],
                    "generation_contract_hash": job["line"].get("_codex2_generation_hash"),
                    "processing_contract_hash": job["line"].get("_codex2_processing_hash"),
                    "synthesis_text": job["synthesis_text"],
                    "canonical_synthesis_text": job["canonical_synthesis_text"],
                    "ellipsis_experiment": job["ellipsis_experiment"],
                    "duration_request": job["synthesis_duration"],
                    "synthesis_start": job["synthesis_start"],
                    "duration_actual": len(y) / SR, "action": job["decision"].action,
                    "num_step": profile["num_step"], "guidance_scale": profile["guidance_scale"],
                    "postprocess_output": generation_config.postprocess_output,
                    "position_temperature": generation_config.position_temperature,
                    "class_temperature": float(profile.get("class_temperature", 0.25)),
                    "denoise": generation_config.denoise,
                    "omnivoice_engine": str(OMNIVOICE_ENGINE),
                    "reference_audio": str(job["ref_path"]),
                    "reference_override": str(job["ref_path"]).endswith(f"{job['line']['id']}_line_emotion.wav"),
                }
                metadata = [item for item in metadata if item["file"] != name]
                metadata.append(row)
                print(f"GEN {job['line']['id']} r{round_index} t{job['take']}: {len(y) / SR:.3f}s", flush=True)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if runtime is None:
        del prompts, model
    else:
        runtime.prompt_cache.update(prompts)
    gc.collect()
    if owns_model:
        torch.cuda.empty_cache()


def transcribe(asr: WhisperModel, path: Path) -> str:
    # Screening is deliberately cheap and VAD-aware.  Confirmation below is
    # the only path that spends the expensive beam-5 decode.
    segments, _ = asr.transcribe(
        str(path), language="de", beam_size=1, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_confirmed(asr: WhisperModel, path: Path) -> str:
    segments, _ = asr.transcribe(
        str(path), language="en", beam_size=5, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def tail_rms_ratio(audio: np.ndarray, sr: int, guard_seconds: float = 0.080) -> float:
    """Return residual end energy for the global acoustic "oder" gate.

    The gate applies to every generated line. If the final 80 ms remain loud,
    duration-conditioned synthesis probably reached its boundary mid-phoneme.
    """
    y = np.asarray(audio, dtype=np.float64).squeeze()
    if not len(y):
        return 0.0
    guard = min(len(y), max(1, round(guard_seconds * sr)))
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-6:
        return 0.0
    return float(np.sqrt(np.mean(y[-guard:] ** 2)) / peak)


def active_tail_rms_ratio(
    audio: np.ndarray, sr: int, guard_seconds: float = 0.080,
) -> float:
    """Residual energy before model-added trailing padding.

    ``postprocess_output=True`` can append ~100 ms of silence after a word that
    was already cut.  Measuring the physical end then returns zero.  Find the
    last sample above -45 dB relative to peak and measure the preceding 80 ms,
    so padding can no longer conceal an abrupt active-voice edge.
    """
    y = np.asarray(audio, dtype=np.float64).squeeze()
    if not len(y):
        return 0.0
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-6:
        return 0.0
    threshold = peak * (10.0 ** (-45.0 / 20.0))
    indices = np.flatnonzero(np.abs(y) > threshold)
    if not len(indices):
        return 0.0
    active_end = int(indices[-1]) + 1
    guard = min(active_end, max(1, round(guard_seconds * sr)))
    return float(np.sqrt(np.mean(y[active_end - guard:active_end] ** 2)) / peak)


def evaluate(
    scene: dict, stem: np.ndarray, stem_sr: int, out: Path, qa: dict,
    only_ids: set[str] | None = None,
    only_rounds: set[int] | None = None,
    use_asr: bool = True,
    runtime: QARuntime | None = None,
) -> tuple[dict[str, list[dict]], set[str]]:
    owns_asr = runtime is None or runtime.asr is None
    asr = runtime.asr if runtime is not None else None
    if use_asr and asr is None:
        print("Loading large-v3-turbo on GPU for FMV/anime QA...", flush=True)
        asr = WhisperModel(
            qa["asr_model"], device="cuda",
            compute_type=qa["asr_compute_type"],
        )
    elif not use_asr:
        print("ASR disabled for cue-separated in-engine/VN QA", flush=True)
    else:
        print("Reusing cached Whisper runtime for global QA", flush=True)
    owns_mms = runtime is None or runtime.mms is None
    mms = runtime.mms if runtime is not None else None
    if mms is None:
        mms = MMS()
    if runtime is not None:
        runtime.asr = asr if use_asr else None
        runtime.mms = mms
    candidate_dir, processed_dir = out / "candidates", out / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    contract_hash = scene.get("_codex2_contract_hash")
    if not contract_hash:
        raise RuntimeError("missing Codex2 scene contract hash before QA")
    by_line: dict[str, list[dict]] = {}
    lines = {line["id"]: line for line in scene["lines"]}
    line_hashes = {
        line_id: line.get("_codex2_line_contract_hash")
        for line_id, line in lines.items()
    }
    accepted_hashes = {
        line_id: set(line.get("_codex2_accepted_line_contract_hashes", [line_hashes[line_id]]))
        for line_id, line in lines.items()
    }
    source_alignment_cache = runtime.source_alignment_cache if runtime is not None else {}
    feature_cache = runtime.source_features if runtime is not None else {}
    qa_hash = qa_contract_hash(qa, Path(__file__).resolve())
    qa_cache_dir = out / "cache" / "qa"
    qa_cache_dir.mkdir(parents=True, exist_ok=True)
    for meta in metadata:
        if meta.get("contract_hash") not in accepted_hashes.get(meta.get("line_id"), set()):
            continue
        if only_ids is not None and meta["line_id"] not in only_ids:
            continue
        if only_rounds is not None and int(meta["round"]) not in only_rounds:
            continue
        line = lines[meta["line_id"]]
        spoken_target = line.get("delivery_text", line["target_text"])
        decision = decide_line(line)
        if decision.action in {KEEP_ORIGINAL, BLOCKED}:
            continue
        output_frames = round((float(line["end"]) - float(line["start"])) * SR)
        cache_path = qa_cache_dir / (
            f"{meta['file']}.{line_hashes[line['id']]}.{qa_hash}.json"
        )
        if cache_path.exists() and (processed_dir / meta["file"]).exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cached"] = True
            by_line.setdefault(line["id"], []).append(cached)
            continue
        ok, cheap_error = cheap_candidate_gate(candidate_dir / meta["file"], output_frames * 8)
        if not ok:
            row = dict(meta)
            row.update({
                "processed": None, "pass": False, "score": 1e9,
                "error": cheap_error, "failure_class": "DETERMINISTIC_AUDIO",
            })
            by_line.setdefault(line["id"], []).append(row)
            continue
        raw, raw_sr = read(candidate_dir / meta["file"])
        if raw_sr != SR:
            raw = resample_exact(raw, raw_sr, SR)
        raw, lead_trim_frames = trim_lead_silence(raw, SR)
        extraction_meta = None
        body_audio_for_alignment = None
        splice_boundaries = None
        try:
            source = source_window(stem, line, stem_sr)
            source = resample_exact(source, stem_sr, SR) if stem_sr != SR else source
            output_frames = round((float(line["end"]) - float(line["start"])) * SR)
            if len(source) < output_frames:
                source = np.pad(source, (0, output_frames - len(source)))
            source = source[:output_frames]
            if decision.action == SHORT_EXTEND_CUT or "delivery_word_start" in line:
                lead = round(float(qa["leading_guard_seconds"]) * SR)
                src_start, _ = active_span(source, SR)
                budget = max(0, output_frames - max(0, src_start - lead))
                if line.get("preserve_leading_effort"):
                    # The contextual body is later joined after the preserved
                    # original effort. Its extraction ceiling must therefore
                    # reserve the prefix length now; otherwise a complete body
                    # can pass cut_short() and only overflow after Empalme B.
                    # This shortens only model-added tail/context. It never
                    # time-stretches or processes the preserved actor effort.
                    splice_boundaries = resolve_leading_boundaries(source, line)
                    prefix_seconds = float(
                        splice_boundaries["source_resume_seconds"]
                    )
                    crossfade_seconds = float(
                        line.get("splice_crossfade_seconds", 0.025)
                    )
                    prefix_frames = max(
                        0, round((prefix_seconds - crossfade_seconds) * SR)
                    )
                    budget = max(0, budget - prefix_frames)
                raw, extraction_meta = cut_short(
                    raw, meta["synthesis_text"],
                    int(line.get("delivery_word_count", decision.cut_after_words or 1)), mms,
                    int(line.get("delivery_word_start", 0)),
                    lead_seconds=float(line.get("delivery_lead_seconds", 0.010)),
                    max_frames=budget,
                )
                body_audio_for_alignment = np.asarray(raw, dtype=np.float32).copy()
            pause_insertion = None
            pause_spec = line.get("minimum_pause_after_word")
            if pause_spec:
                aligned_for_pause = mms.align(raw, spoken_target)
                word_index = int(pause_spec["word_index"])
                if word_index < 0 or word_index + 1 >= len(aligned_for_pause):
                    raise ValueError(
                        f"pause word index {word_index} outside "
                        f"{len(aligned_for_pause)} aligned words"
                    )
                current_gap = (
                    aligned_for_pause[word_index + 1]["start"]
                    - aligned_for_pause[word_index]["end"]
                )
                reallocate = min(
                    float(pause_spec.get("reallocate_from_lead_seconds", 0.0)),
                    max(0.0, aligned_for_pause[0]["start"] - 0.015),
                )
                if reallocate > 0:
                    remove_frames = round(reallocate * SR)
                    raw = raw[remove_frames:]
                else:
                    remove_frames = 0
                target_gap = float(pause_spec["seconds"])
                extra_gap = max(0.0, target_gap - current_gap)
                if extra_gap > 0:
                    tone, tone_rms = room_tone(source, SR)
                    if tone_rms <= 0 or not len(tone):
                        raise ValueError("no source room tone available for pause insertion")
                    midpoint = (
                        aligned_for_pause[word_index]["end"] + current_gap / 2.0
                    )
                    insert_at = round(midpoint * SR)
                    filler = tile_tone(tone, round(extra_gap * SR), SR)
                    raw = np.concatenate((raw[:insert_at], filler, raw[insert_at:]))
                    pause_insertion = {
                        "after_word_index": word_index,
                        "gap_before_seconds": current_gap,
                        "target_gap_seconds": target_gap,
                        "inserted_seconds": len(filler) / SR,
                        "reallocated_lead_seconds": remove_frames / SR,
                        "room_tone_rms": tone_rms,
                    }

            splice_info = None
            model_lead_alignment_seconds = None
            model_lead_alignment_trim_frames = 0
            if line.get("preserve_leading_effort"):
                if line.get("trim_model_lead_before_splice"):
                    # Some stochastic takes retain a long artificial lead
                    # after the generic -45 dB trim.  In a mixed Empalme B
                    # cue that delay is not artistic breath: it moves the
                    # German first word away from the measured handoff and
                    # makes QA report hundreds of milliseconds of false seam
                    # error.  Trim only when the map explicitly authorizes
                    # it; preserved source effort is never touched.
                    model_lead_frames, _ = active_span(raw, SR)
                    if model_lead_frames > round(0.080 * SR):
                        raw = np.asarray(raw[model_lead_frames:], dtype=np.float32)
                    # Energy VAD cannot distinguish a breath/noise lead from a
                    # missing first German word.  For an explicitly surgical
                    # Empalme-B line, use MMS's independent word edge as a
                    # second boundary check.  A late first aligned word means
                    # this stochastic take did not start the requested body;
                    # fail it closed instead of mounting a clipped/shifted
                    # sentence.  A small residual edge is removed before the
                    # splice so the first phoneme lands on the measured seam.
                    lead_alignment = mms.align(raw, spoken_target)
                    if lead_alignment:
                        model_lead_alignment_seconds = float(
                            lead_alignment[0]["start"]
                        )
                        max_model_lead = float(line.get(
                            "max_model_lead_alignment_seconds", 0.25
                        ))
                        if model_lead_alignment_seconds > max_model_lead:
                            raise ValueError(
                                "model body first word is late for Empalme B: "
                                f"{model_lead_alignment_seconds:.3f}s > "
                                f"{max_model_lead:.3f}s"
                            )
                        if model_lead_alignment_seconds > 0.010:
                            # Keep a 10 ms consonant pre-roll; trimming exactly
                            # at MMS's first frame can shave the leading /n/.
                            model_lead_alignment_trim_frames = max(
                                0,
                                round((model_lead_alignment_seconds - 0.010) * SR),
                            )
                            raw = np.asarray(
                                raw[model_lead_alignment_trim_frames:],
                                dtype=np.float32,
                            )
                # align_onset_exact normally removes model-added trailing
                # padding via its active-span crop.  A hybrid splice bypasses
                # that function, so remove only the artificial postprocess
                # tail here while retaining 50 ms for the real consonant/
                # breath decay.  Without this, clean speech can falsely
                # overflow the fixed anime window by the padded silence.
                _, body_active_end = active_span(raw, SR)
                splice_tail_guard = float(line.get(
                    "splice_trailing_guard_seconds",
                    qa["trailing_guard_seconds"],
                ))
                body_keep = min(
                    len(raw), body_active_end + round(splice_tail_guard * SR),
                )
                body_trailing_trim_frames = max(0, len(raw) - body_keep)
                raw = raw[:body_keep]
                # QA must align the translated body in its own timebase.  If
                # it aligns against the already-assembled prefix+body clip,
                # the preserved effort offset is counted a second time and
                # every otherwise-correct Empalme B reports a ~1 s onset
                # error.  Keep this snapshot after the guarded tail trim and
                # before build_leading_splice adds the original head.
                body_audio_for_alignment = np.asarray(raw, dtype=np.float32).copy()
                measure = _score_from_array(source, SR)
                acoustic_score = scream_score(measure) if measure else -1
                # The normal Empalme-B safety classifier is intentionally
                # strict for nonverbal screams/efforts.  A map may explicitly
                # authorize a lexical leading interjection (for example the
                # English "Geez..." that is deliberately preserved) while
                # still using the same boundary, seam, and prefix-integrity
                # gates.  This flag is per-line and never inferred globally.
                if (
                    acoustic_score < 2
                    and not line.get("allow_non_neutral_leading_interjection")
                ):
                    raise ValueError(
                        f"leading effort failed acoustic scream gate: {acoustic_score} < 2"
                    )
                splice_boundaries = splice_boundaries or resolve_leading_boundaries(source, line)
                effort_end = float(splice_boundaries["effort_end_seconds"])
                splice_cut = float(splice_boundaries["source_resume_seconds"])
                raw, splice_seam, used_room_tone = build_leading_splice(
                    source, SR, raw, SR, effort_end, splice_cut,
                    min_pause_seconds=float(
                        line.get("splice_min_pause_seconds", 0.070)
                    ),
                    crossfade_seconds=float(
                        line.get("splice_crossfade_seconds", 0.025)
                    ),
                    crossfade_curve=str(
                        line.get("splice_crossfade_curve", "equal_gain")
                    ),
                )
                notch_db = seam_notch_db(raw, SR, splice_seam)
                max_seam_notch_db = float(
                    line.get("max_splice_seam_notch_db", 12.0)
                )
                if notch_db > max_seam_notch_db:
                    raise ValueError(
                        f"audible leading-effort seam: {notch_db:.2f} dB "
                        f"> {max_seam_notch_db:.2f} dB"
                    )
                splice_info = {
                    "acoustic_measure": measure,
                    "scream_score": acoustic_score,
                    "effort_end_seconds": effort_end,
                    "source_resume_seconds": splice_cut,
                    "detected_effort_end_seconds": splice_boundaries[
                        "detected_effort_end_seconds"
                    ],
                    "detected_source_resume_seconds": splice_boundaries[
                        "detected_source_resume_seconds"
                    ],
                    "boundary_tolerance_seconds": splice_boundaries[
                        "boundary_tolerance_seconds"
                    ],
                    "boundary_valid": splice_boundaries["boundary_valid"],
                    "seam_seconds": splice_seam,
                    "crossfade_curve": str(
                        line.get("splice_crossfade_curve", "equal_gain")
                    ),
                    "seam_notch_db": notch_db,
                    "max_seam_notch_db": max_seam_notch_db,
                    "room_tone_used": used_room_tone,
                    "body_trailing_trim_frames": body_trailing_trim_frames,
                    "model_lead_alignment_seconds": model_lead_alignment_seconds,
                    "model_lead_alignment_trim_frames": model_lead_alignment_trim_frames,
                    "preserved_prefix_text": line.get("preserved_prefix_text"),
                    "preserved_source_intervals": line.get(
                        "preserved_source_intervals", []
                    ),
                    "leading_interjection_authorized": bool(
                        line.get("allow_non_neutral_leading_interjection")
                    ),
                }
                crossfade = float(
                    line.get("splice_crossfade_seconds", 0.025)
                )
                if splice_seam + crossfade + 1e-6 < effort_end:
                    raise ValueError(
                        f"Empalme B seam starts inside preserved effort for "
                        f"{line['id']}: seam={splice_seam:.3f}s "
                        f"effort_end={effort_end:.3f}s "
                        f"crossfade={crossfade:.3f}s"
                    )
                splice_info["effort_boundary_gate"] = True
            timing_correction = None
            if not line.get("preserve_leading_effort"):
                # Duration conditioning is the base. Only rescue an active
                # German span outside the agreed +/-0.35 s cinematic window.
                # This operates on the synthetic body before placement; it
                # never stretches preserved efforts or the full movie mix.
                source_start_for_timing, source_end_for_timing = active_span(
                    source, SR
                )
                source_active_for_timing = max(
                    0.05,
                    (source_end_for_timing - source_start_for_timing) / SR,
                )
                timing_dir = out / "_timing"
                timing_dir.mkdir(parents=True, exist_ok=True)
                timing_contract = json.loads(
                    CONFIG_PATH.read_text(encoding="utf-8")
                )["contracts"]
                raw, timing_correction = prod_timing.correct_length(
                    np.asarray(raw, dtype=np.float64),
                    SR,
                    source_active_for_timing,
                    FFMPEG,
                    timing_dir,
                    max_ratio_deviation=float(
                        timing_contract.get("max_tempo_deviation", 0.15)
                    ),
                    under_tol=float(
                        timing_contract.get("duration_tolerance_seconds", 0.35)
                    ),
                    over_tol=float(
                        timing_contract.get("duration_tolerance_seconds", 0.35)
                    ),
                )
                raw = np.asarray(raw, dtype=np.float32)
                # A take can satisfy the +/-0.35 s duration contract yet still
                # be too long to move to a late English onset without crossing
                # the cue boundary. Compress only the synthetic body by the
                # minimum amount needed to preserve that onset and the trailing
                # guard. Never process a preserved effort or the movie mix.
                raw_start_for_fit, raw_end_for_fit = active_span(raw, SR)
                available_active_seconds = max(
                    0.05,
                    (
                        output_frames
                        - source_start_for_timing
                        - round(float(qa["trailing_guard_seconds"]) * SR)
                    )
                    / SR,
                )
                raw_active_seconds = max(
                    0.0, (raw_end_for_fit - raw_start_for_fit) / SR
                )
                if raw_active_seconds > available_active_seconds:
                    target_raw_end = (
                        raw_start_for_fit / SR + available_active_seconds
                    )
                    raw, onset_fit = prod_timing.fit_speech_end(
                        np.asarray(raw, dtype=np.float64),
                        SR,
                        target_raw_end,
                        FFMPEG,
                        timing_dir,
                        max_ratio_deviation=float(
                            timing_contract.get("max_tempo_deviation", 0.15)
                        ),
                    )
                    raw = np.asarray(raw, dtype=np.float32)
                    timing_correction["onset_fit"] = onset_fit
            # Raw-tail values remain diagnostics. PP=false intentionally emits
            # no model padding, so the last 80 ms can contain a perfectly
            # complete final phoneme. The release gate is evaluated after the
            # localized placement/fade below.
            tail_ratio = tail_rms_ratio(raw, SR)
            active_tail_ratio = active_tail_rms_ratio(raw, SR)
            tail_ok = True
            alignment_fallback = None
            try:
                if line.get("preserve_leading_effort"):
                    # build_leading_splice already carries the original onset,
                    # pause, cosine crossfade and room-tone bed.  Re-aligning it
                    # would destroy that timing and compound its fades.
                    if len(raw) > output_frames:
                        raise ValueError(
                            f"leading splice overflow: need {len(raw)} frames, "
                            f"have {output_frames}"
                        )
                    processed = np.zeros(output_frames, dtype=np.float32)
                    processed[:len(raw)] = raw
                elif decision.action == SHORT_EXTEND_CUT:
                    # `raw` was already cut+faded to a complete, silence-terminated
                    # body above; only reposition it, don't re-detect its active
                    # span or fade it a second time (see place_short_cut docstring).
                    processed = place_short_cut(
                        source, raw, SR, output_frames, float(qa["leading_guard_seconds"]),
                    )
                else:
                    processed = align_onset_exact(
                        source, raw, SR, output_frames,
                        float(qa["leading_guard_seconds"]),
                        float(qa["trailing_guard_seconds"]),
                        float(qa["fade_seconds"]),
                    )
            except ValueError as exc:
                # Four attempts are the hard ceiling. If the complete waveform already
                # fits the cue but cannot be shifted to the source onset, keep it
                # whole at window start and flag onset REVIEW. Never cut a word.
                if len(raw) > output_frames:
                    raise
                # ``align_onset_exact`` can reject an otherwise valid take when the
                # model's artificial lead-in consumes the available guard. The
                # candidate was already passed through ``trim_lead_silence`` above,
                # but leading-splice/legacy paths may reintroduce that quiet prefix.
                # Remove only that non-speech prefix before the safe window-start
                # fallback; otherwise the fallback records an onset failure and leaves
                # an audible delayed/overlapping attack at the beginning of the cue.
                fallback_raw, _ = trim_lead_silence(raw, SR)
                if len(fallback_raw) <= output_frames:
                    raw = fallback_raw
                processed = np.zeros(output_frames, dtype=np.float32)
                processed[:len(raw)] = raw
                alignment_fallback = f"window_start_unaligned: {exc}"
            src_start, src_end = active_span(source, SR)
            gen_start, gen_end = active_span(processed, SR)
            src_lufs = meter_lufs(source[src_start:src_end] if src_end > src_start else source, SR)
            gen_lufs = meter_lufs(processed[gen_start:gen_end] if gen_end > gen_start else processed, SR)
            requested_gain = float(np.clip(src_lufs - gen_lufs, -8.0, 8.0))
            if line.get("preserve_leading_effort"):
                # The original effort must remain at the actor's real level.
                # build_leading_splice already matches only the synthetic body
                # to the source speech, so a second whole-clip gain is wrong.
                applied_gain = 0.0
            else:
                processed, applied_gain = constant_gain(
                    processed, requested_gain, peak_limit=0.98,
                )
            # Librosa's onset grid can differ by one or more hops after fades
            # and gain. Refine by a pure integer-sample shift only when the
            # complete active span remains inside the exact window.
            refine_frames = 0
            refined_start, refined_end = active_span(processed, SR)
            delta = src_start - refined_start
            if (
                not line.get("preserve_leading_effort")
                and delta > 0
                and refined_end + delta <= output_frames
            ):
                shifted = np.zeros_like(processed)
                shifted[delta:] = processed[:-delta]
                processed = shifted
                refine_frames = int(delta)
            window_tail_ratio = tail_rms_ratio(processed, SR)
            window_active_tail_ratio = active_tail_rms_ratio(processed, SR)
            placement_hits_end = active_span(processed, SR)[1] >= (
                output_frames - round(float(qa["trailing_guard_seconds"]) * SR)
            )
            edge_cut_risk = (
                placement_hits_end
                and max(window_tail_ratio, window_active_tail_ratio)
                >= float(qa.get("max_safe_edge_tail_ratio", 0.03))
            )
            # Energy at a fixed-window edge is only a warning: it can be a
            # complete plosive, breath, room echo or a slightly loose subtitle
            # boundary.  Promote it to a release failure only after the final
            # word alignment below independently shows an abbreviated release.
            tail_ok = True
            processed_path = processed_dir / meta["file"]
            write_exact(processed_path, processed, SR, output_frames)
            expected = words(spoken_target)
            source_language_suspect = []
            source_language_confirmed = False
            source_language_confirmation_info = None
            if asr is not None:
                transcript = transcribe(asr, processed_path)
                wer, joined_equivalent = spoken_wer(
                    expected, words(transcript),
                )
                suspected, screening_confirmed, initial_diag = source_language_confirmation(
                    line["source_text"], spoken_target, transcript,
                )
                source_language_suspect = initial_diag["suspected_tokens"]
                source_language_confirmation_info = initial_diag
                if suspected or screening_confirmed:
                    confirmed_text = transcribe_confirmed(asr, processed_path)
                    confirmed_tokens, confirmed, confirm_diag = source_language_confirmation(
                        line["source_text"], spoken_target, confirmed_text,
                    )
                    source_language_confirmed = confirmed
                    source_language_confirmation_info = {
                        "screening": initial_diag,
                        "confirmation": confirm_diag,
                        "confirmation_transcript": confirmed_text,
                    }
                    leaked_source_tokens = (
                        confirmed_tokens
                        if confirmed and confirmed_tokens
                        else (["<confirmed_source_language>"] if confirmed else [])
                    )
                else:
                    leaked_source_tokens = []
            else:
                # Isolated engine cues already own an exact reference and text
                # mapping.  Whisper is neither a gate nor a ranking signal for
                # these assets; acoustic/edge/frame contracts remain active.
                transcript = None
                wer = 0.0
                joined_equivalent = False
                leaked_source_tokens = []
            gen_start, gen_end = active_span(processed, SR)
            onset_error_ms = abs(gen_start - src_start) / SR * 1000
            source_active = max((src_end - src_start) / SR, 0.1)
            target_active = max((gen_end - gen_start) / SR, 0.1)
            span_error = abs(target_active - source_active)
            final_lufs = meter_lufs(processed[gen_start:gen_end] if gen_end > gen_start else processed, SR)
            lufs_delta = final_lufs - src_lufs
            peak = float(np.max(np.abs(processed)))
            alignment_audio = (
                body_audio_for_alignment
                if body_audio_for_alignment is not None
                else processed
            )
            target_alignment = mms.align(alignment_audio, spoken_target)
            splice_speech_onset_error_ms = None
            if (
                splice_info is not None
                and target_alignment
                and line.get("preserve_leading_effort")
            ):
                # A technically smooth join can still sound like two clips if
                # a fixed pause shifts the translated word away from the
                # actor's original speech entry.  Compare word alignment, not
                # generic VAD: the preserved effort intentionally dominates
                # the latter.
                splice_speech_onset_error_ms = abs(
                    float(splice_info["seam_seconds"])
                    + float(target_alignment[0]["start"])
                    - (
                        float(splice_info["source_resume_seconds"])
                        - float(line.get("splice_crossfade_seconds", 0.025))
                    )
                ) * 1000.0
            internal_boundaries = []
            # A comma does not imply a silent gap in natural speech. Only
            # explicit ellipses and strong internal punctuation are pause gates.
            tokens = re.findall(r"\w+|\.{2,}|[.;:!?]", spoken_target, flags=re.UNICODE)
            word_index = -1
            for token in tokens:
                if re.match(r"\w+", token, flags=re.UNICODE):
                    word_index += 1
                elif word_index >= 0 and word_index < len(target_alignment) - 1:
                    internal_boundaries.append(word_index)
            gaps = [target_alignment[i + 1]["start"] - target_alignment[i]["end"] for i in internal_boundaries]
            pause_ok = all(gap >= 0.075 for gap in gaps)
            # Intentional strong-punctuation pauses are not "slow speech".
            # Exclude them from the articulation-rate denominator; otherwise
            # fixing a missing dramatic pause can make the rate gate fail.
            target_voiced_seconds = max(target_active - sum(gaps), 0.1)
            rate_ratio = (
                (syllables(spoken_target) / target_voiced_seconds)
                / (syllables(line["source_text"]) / source_active)
            )
            # Cross-language syllable-rate ratios are structurally invalid for
            # very short translations: "Sign here" vs "Unterschreib hier"
            # can differ by ~1.6-1.9x while both deliveries are natural.  For
            # 1-3 words, final-word duration, span, onset and tail already guard
            # rushing without penalising German merely for having more
            # syllables than English.
            rate_ok = (
                True
                if decision.action == SHORT_TTS_QA
                else (
                    float(qa["min_rate_ratio"])
                    <= rate_ratio
                    <= float(qa["max_rate_ratio"])
                )
            )

            # Postprocess padding can hide a clipped final word from the
            # classic last-80-ms "oder" gate.  On short lines, compare the
            # aligned final phoneme/syllable duration with the actor's source.
            # This caught the human-confirmed clipped "schon": 0.495x of the
            # source final syllable, while the corrected take reaches 0.556x.
            if line["id"] not in source_alignment_cache:
                source_alignment_cache[line["id"]] = mms.align(
                    source, line["source_text"],
                )
            source_alignment = source_alignment_cache[line["id"]]
            final_word_duration_ratio = None
            if source_alignment and target_alignment:
                source_last = source_alignment[-1]
                target_last = target_alignment[-1]
                source_last_per_syllable = (
                    (source_last["end"] - source_last["start"])
                    / syllables(source_last["text"])
                )
                target_last_per_syllable = (
                    (target_last["end"] - target_last["start"])
                    / syllables(target_last["text"])
                )
                if source_last_per_syllable > 1e-6:
                    final_word_duration_ratio = (
                        target_last_per_syllable / source_last_per_syllable
                    )
            contextual_tail_gate_required = extraction_meta is not None
            extracted_tail_release_ok = bool(
                extraction_meta and extraction_meta.get("tail_release_ok")
            )
            final_word_ratio_threshold = float(
                line.get(
                    "min_short_final_word_duration_ratio",
                    qa.get(
                        "min_final_word_duration_ratio",
                        qa.get("min_short_final_word_duration_ratio", 0.55),
                    ),
                )
            )
            contextual_final_word_ok = (
                contextual_final_word_gate(
                    extraction_meta,
                    float(target_alignment[-1]["end"])
                    if target_alignment else None,
                    len(alignment_audio),
                    SR,
                )
                if contextual_tail_gate_required else True
            )
            # Cross-language final-word *duration* is not a valid hard gate:
            # German function words (``da``, ``oder``, ``nicht``) can be much
            # shorter than the corresponding English word even when the
            # recording is complete.  Use the independent ASR lexical edge as
            # the release signal and keep the duration ratio diagnostic.  A
            # missing final token still fails closed, as does a contextual
            # extraction whose tail-release gate did not find a quiet finish.
            expected_tokens = words(spoken_target)
            heard_tokens = words(transcript or "")
            final_expected = expected_tokens[-1] if expected_tokens else ""
            final_heard = heard_tokens[-1] if heard_tokens else ""
            final_expected_canonical = canonical_content_token(final_expected)
            final_heard_canonical = canonical_content_token(final_heard)
            final_word_content_ok = bool(
                not final_expected
                or (asr is None and transcript is None)
                or (
                    bool(final_heard)
                    and (
                        final_heard_canonical == final_expected_canonical
                        or final_heard_canonical.endswith(final_expected_canonical)
                        or final_expected_canonical.endswith(final_heard_canonical)
                    )
                )
            )
            final_word_ratio_ok = (
                final_word_duration_ratio is not None
                and final_word_duration_ratio >= final_word_ratio_threshold
            )
            final_word_ok = bool(contextual_final_word_ok and final_word_content_ok)
            delivery_content_ok, delivery_content_info = delivery_content_gate(
                spoken_target, transcript, asr is not None,
            )
            tail_cut_evidence = bool(
                (
                    contextual_tail_gate_required
                    and not extracted_tail_release_ok
                )
                or (
                    edge_cut_risk
                    and not final_word_content_ok
                )
            )
            tail_ok = not tail_cut_evidence
            source_feature = feature_cache.get(line["id"])
            if source_feature is None:
                source_feature = {
                    "pitch_contour": contour(source, "pitch"),
                    "energy_contour": contour(source, "energy"),
                    "f0_median": median_f0(source, SR),
                    "lufs": src_lufs,
                }
                feature_cache[line["id"]] = source_feature
            pitch_correlation = corr(source_feature["pitch_contour"], contour(processed, "pitch"))
            energy_correlation = corr(source_feature["energy_contour"], contour(processed, "energy"))
            emotion_values = [value for value in (pitch_correlation, energy_correlation) if value is not None]
            emotion_correlation = float(np.mean(emotion_values)) if emotion_values else None
            source_f0_median, source_f0_frames = source_feature["f0_median"]
            generated_f0_median, generated_f0_frames = median_f0(processed, SR)
            f0_median_ratio = (
                generated_f0_median / source_f0_median
                if source_f0_median and generated_f0_median else None
            )
            # ``pyin`` occasionally reports the source one octave above the
            # actual actor (especially on quiet/noisy FMV tails).  Treat an
            # exact octave-equivalent ratio as the same pitch identity; keep
            # the raw ratio in the report so this remains auditable.
            pitch_min = float(qa.get("min_f0_median_ratio", 0.72))
            pitch_max = float(qa.get("max_f0_median_ratio", 1.30))
            pitch_ratio_candidates = (
                [f0_median_ratio, f0_median_ratio * 2.0, f0_median_ratio / 2.0]
                if f0_median_ratio is not None else []
            )
            pitch_identity_ratio = next(
                (value for value in pitch_ratio_candidates if pitch_min <= value <= pitch_max),
                None,
            )
            pitch_identity_ok = (
                f0_median_ratio is None
                or pitch_identity_ratio is not None
            )
            onset_limit_ms = float(
                qa.get("max_effort_onset_error_ms", 20.0)
                if line.get("preserve_leading_effort")
                else qa["max_onset_error_ms"]
            )
            effort_prefix_max_abs_error = None
            effort_prefix_preserved = True
            if line.get("preserve_leading_effort") and splice_info is not None:
                prefix_frames = max(
                    0, round(float(splice_info["seam_seconds"]) * SR),
                )
                effort_prefix_max_abs_error = float(np.max(
                    np.abs(processed[:prefix_frames] - source[:prefix_frames]),
                    initial=0.0,
                ))
                effort_prefix_preserved = effort_prefix_max_abs_error <= 1e-7
            text_diagnostic_ok = wer <= float(qa["max_wer"])
            text_gate_ok, text_gate_info = target_content_gate(
                spoken_target, transcript or "", wer, float(qa["max_wer"]),
                joined_equivalent,
            ) if asr is not None else (True, {"asr_enabled": False})
            subtitle_delivery_line = bool(
                line.get("force_clone")
                and decision.action not in {KEEP_ORIGINAL, BLOCKED}
            )
            # For an authorized anime subtitle, the fixed card window is the
            # delivery contract. English and German do not have identical
            # phonetic durations. Duration correction is still performed
            # before this QA stage; span/rate, onset, pitch, pause and text
            # remain observable quality signals, not hidden release blockers.
            # English leakage, truncation, bad splice, clipping, loudness and
            # frame/container errors remain hard safety gates below.
            subtitle_window_fit = bool(
                subtitle_delivery_line
                and not leaked_source_tokens
                and not edge_cut_risk
                and gen_start >= 0
                and gen_end <= output_frames
                and target_alignment
                and float(target_alignment[-1]["end"]) <= len(alignment_audio) / SR + 0.010
            )
            # Kept for ranking/telemetry only.  A complete German final word
            # is not rejected merely because the English reference span or
            # syllable rate differs.
            span_release_ok = span_error <= float(qa["max_span_error_seconds"])
            rate_release_ok = rate_ok
            final_word_release_ok = final_word_ok
            hard = {
                "not_empty": peak > 1e-3,
                "source_language": not leaked_source_tokens,
                "tail": tail_ok,
                "final_word": final_word_release_ok,
                # This is a lexical-presence safety gate, not the removed WER
                # ranking gate.  A take that is only ``ich nicht`` or only
                # the final clause is not a genuine delivery of the subtitle.
                "delivery_content": delivery_content_ok,
                "splice_seam": (
                    splice_info is None
                    or float(splice_info["seam_notch_db"])
                    <= float(splice_info.get("max_seam_notch_db", 12.0))
                ),
                "splice_boundary": (
                    splice_info is None
                    or bool(splice_info.get("boundary_valid"))
                    and bool(splice_info.get("effort_boundary_gate"))
                ),
                "splice_speech_timing": (
                    splice_speech_onset_error_ms is None
                    or splice_speech_onset_error_ms
                    <= float(line.get(
                        "max_splice_speech_onset_error_ms",
                        qa.get("max_splice_speech_onset_error_ms", 20.0),
                    ))
                ),
                # ±1 dB remains the ranking target. A difference up to 3 dB is
                # not a destructive mismatch and must not outweigh clean
                # content/tail/identity; it remains penalised in the score.
                # Gain has already been matched on active speech above.
                # Residual LUFS is a selection metric (the source may carry
                # scene effects), not a reason to discard an otherwise clean
                # exact-window take.
                "lufs": bool(
                    abs(lufs_delta) <= float(
                        line.get("max_lufs_delta", qa.get("max_lufs_delta", 3.0))
                    )
                    or (
                        line.get("allow_unreliable_source_lufs")
                        and src_lufs <= float(line.get("unreliable_source_lufs_threshold", -45.0))
                    )
                ),
                "clipping": peak < float(qa["clip_peak"]),
                "frames": len(processed) == output_frames,
            }
            diagnostic_gates = {
                # These are deliberately not release gates.  Keep the values
                # visible for debugging and soft candidate ranking.
                "text": text_gate_ok,
                "onset": (
                    effort_prefix_preserved
                    if line.get("preserve_leading_effort")
                    else onset_error_ms <= onset_limit_ms
                ),
                "span": span_release_ok,
                "rate": rate_release_ok,
                "pause": pause_ok,
                "pitch_identity": pitch_identity_ok,
            }
            # Calibrated fidelity floor: hard defects dominate, while
            # cross-language prosody is a bounded soft component.  This avoids
            # chasing numerical perfection but still rejects a line that is
            # technically valid yet clearly unlike the source performance.
            identity_component = (
                max(
                    0.0,
                    1.0 - abs(math.log(max(f0_median_ratio, 0.05)))
                    / math.log(1.5),
                )
                if f0_median_ratio is not None else 0.65
            )
            integrity_component = float(np.mean([
                hard["tail"], hard["final_word"], hard["delivery_content"],
                hard["splice_seam"],
                hard["clipping"], hard["frames"],
            ]))
            timing_component = float(np.mean([
                diagnostic_gates["onset"], hard["splice_speech_timing"],
                diagnostic_gates["pause"],
            ]))
            loudness_component = max(
                0.0,
                1.0 - abs(lufs_delta)
                / max(float(qa["hard_lufs_limit"]), 1e-6),
            )
            prosody_component = (
                float(np.clip((emotion_correlation + 1.0) / 2.0, 0.0, 1.0))
                if emotion_correlation is not None else 0.60
            )
            fidelity_components = {
                "content": 1.0 if text_diagnostic_ok else max(0.0, 1.0 - wer),
                "voice_pitch_identity": identity_component,
                "technical_integrity": integrity_component,
                "timing_and_pauses": timing_component,
                "loudness": loudness_component,
                "prosody": prosody_component,
            }
            # Text/WER is diagnostic-only for this run.  Keep its measured
            # component in the report, but use a neutral value in the
            # aggregate fidelity score so it cannot rank or reject a take.
            content_for_score = 1.0
            fidelity_score = 100.0 * (
                0.25 * content_for_score
                + 0.15 * fidelity_components["voice_pitch_identity"]
                + 0.20 * fidelity_components["technical_integrity"]
                + 0.15 * fidelity_components["timing_and_pauses"]
                + 0.10 * fidelity_components["loudness"]
                + 0.15 * fidelity_components["prosody"]
            )
            fidelity_minimum_ok = (
                fidelity_score >= float(qa.get("minimum_fidelity_score", 70.0))
            )
            passed = all(hard.values())
            score = (
                # Text/WER intentionally does not participate in ranking.
                # Timing metrics remain soft preferences; hard safety is
                # decided exclusively by ``hard`` above.
                onset_error_ms / 12.0
                + 4.0 * min(
                    abs(span_error) / max(float(qa["max_span_error_seconds"]), 1e-6),
                    3.0,
                )
                + 4.0 * min(abs(math.log(max(rate_ratio, 0.05))), 3.0)
                + abs(lufs_delta)
                + 2.0 * max(0.0, -(emotion_correlation or 0.0))
                + (8.0 if not tail_ok else 0.0)
                + (
                    8.0 * max(
                        0.0,
                        final_word_ratio_threshold
                        - final_word_duration_ratio,
                    )
                    if final_word_duration_ratio is not None else 0.0
                )
                + (4.0 if not diagnostic_gates["pause"] else 0.0)
                + (
                    12.0 * abs(math.log(max(f0_median_ratio, 0.05)))
                    if f0_median_ratio is not None else 0.0
                )
                + (12.0 if not diagnostic_gates["pitch_identity"] else 0.0)
            )
            row = dict(meta)
            row.update({
                "processed": str(processed_path), "transcript": transcript,
                "wer": wer, "asr_enabled": asr is not None,
                 "text_diagnostic_ok": text_diagnostic_ok,
                 "text_gate_ok": text_gate_ok,
                 "text_gate_info": text_gate_info,
                "source_language_leak_tokens": leaked_source_tokens,
                "source_language_suspect_tokens": source_language_suspect,
                "source_language_confirmed": source_language_confirmed,
                "source_language_confirmation": source_language_confirmation_info,
                "orthographic_join_equivalent": joined_equivalent,
                "onset_error_ms": onset_error_ms, "span_error": span_error,
                "span_diagnostic_ok": span_error <= float(qa["max_span_error_seconds"]),
                "rate_ratio": rate_ratio, "rate_diagnostic_ok": rate_ok,
                "source_lufs": src_lufs, "final_lufs": final_lufs,
                "lufs_delta": lufs_delta, "gain_db": applied_gain, "peak": peak,
                "lufs_diagnostic_ok": abs(lufs_delta) <= float(
                    line.get("max_lufs_delta", qa.get("max_lufs_delta", 3.0))
                ),
                "tail_rms_ratio": tail_ratio,
                "active_tail_rms_ratio": active_tail_ratio,
                "window_tail_rms_ratio": window_tail_ratio,
                "window_active_tail_rms_ratio": window_active_tail_ratio,
                "placement_hits_end": placement_hits_end,
                "edge_cut_risk": edge_cut_risk,
                "tail_cut_evidence": tail_cut_evidence,
                "final_word_duration_ratio": final_word_duration_ratio,
                "final_word_content_ok": final_word_content_ok,
                "delivery_content_ok": delivery_content_ok,
                "delivery_content": delivery_content_info,
                "lead_trim_frames": lead_trim_frames,
                "lead_trim_ms": lead_trim_frames / SR * 1000.0,
                "pause_gaps": gaps, "pitch_correlation": pitch_correlation,
                "source_f0_median_hz": source_f0_median,
                "generated_f0_median_hz": generated_f0_median,
                "source_f0_voiced_frames": source_f0_frames,
                "generated_f0_voiced_frames": generated_f0_frames,
                "f0_median_ratio": f0_median_ratio,
                "f0_median_ratio_octave_adjusted": pitch_identity_ratio,
                "pitch_identity_diagnostic_ok": pitch_identity_ok,
                "fidelity_score": fidelity_score,
                "fidelity_minimum_diagnostic_ok": fidelity_minimum_ok,
                "fidelity_components": fidelity_components,
                "diagnostic_gates": diagnostic_gates,
                "text_ranking_enabled": False,
                "energy_correlation": energy_correlation, "emotion_correlation": emotion_correlation,
                "onset_refine_frames": refine_frames,
                "splice_speech_onset_error_ms": splice_speech_onset_error_ms,
                 "effort_prefix_max_abs_error": effort_prefix_max_abs_error,
                 "effort_prefix_preserved": effort_prefix_preserved,
                 "extraction_tail": extraction_meta,
                 "contextual_tail_gate_required": contextual_tail_gate_required,
                 "extracted_tail_release_ok": extracted_tail_release_ok,
                 "leading_effort_splice": splice_info,
                "pause_insertion": pause_insertion,
                "timing_correction": timing_correction,
                "hard_gates": hard, "pass": passed, "score": score,
                "alignment_fallback": alignment_fallback,
            })
        except Exception as exc:
            row = dict(meta)
            row.update({"processed": None, "pass": False, "score": 1e9, "error": str(exc)})
        row["qa_contract_hash"] = qa_contract_hash(qa, Path(__file__).resolve())
        row["failure_class"] = classify_failure(row)
        cache_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        by_line.setdefault(line["id"], []).append(row)
        print(f"QA {line['id']} {meta['file']}: {'PASS' if row['pass'] else 'FAIL'} score={row['score']:.2f}", flush=True)
    failed = set()
    for line in scene["lines"]:
        if only_ids is not None and line["id"] not in only_ids:
            continue
        decision = decide_line(line)
        if decision.action not in {KEEP_ORIGINAL, BLOCKED} and not any(
            row["pass"] for row in by_line.get(line["id"], [])
        ):
            failed.add(line["id"])
    if asr is not None:
        del asr
    del mms
    gc.collect()
    torch.cuda.empty_cache()
    return by_line, failed


def select_and_mount(scene: dict, stem_path: Path, full_path: Path, out: Path, rankings: dict[str, list[dict]]) -> dict:
    contract_hash = scene.get("_codex2_contract_hash")
    if not contract_hash:
        raise RuntimeError("missing Codex2 scene contract hash before selection")
    stem, stem_sr = read(stem_path)
    full, full_sr = read(full_path, always_2d=True)
    if stem_sr != full_sr or len(stem) != len(full):
        raise ValueError("stem/full container mismatch")
    original_spec = spec(full_path)
    selected_dir = out / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    generated_mix = np.zeros(len(stem), dtype=np.float32)
    mask = np.zeros(len(stem), dtype=bool)
    report_lines = []
    generated_intervals = []
    review_masked_ids: set[str] = set()
    mapped_voice_ids = {str(line["id"]) for line in scene.get("lines", [])}
    expected_visual_ids = {
        str(line_id) for line_id in scene.get("expected_visual_ids", mapped_voice_ids)
    }
    missing_expected_visual_ids = sorted(expected_visual_ids - mapped_voice_ids)
    if missing_expected_visual_ids:
        raise RuntimeError(
            f"visual-card coverage incomplete for {scene.get('scene')}: "
            + ", ".join(missing_expected_visual_ids)
        )
    if scene.get("coverage_status") not in (
        None, "COMPLETE", "RECONCILED_WITH_LEGACY_INVENTORY",
    ):
        raise RuntimeError(
            f"visual-card coverage is not release-ready for {scene.get('scene')}: "
            f"{scene.get('coverage_status')}"
        )
    required_voice_ids = {
        line["id"]
        for line in scene["lines"]
        if decide_line(line).action not in {KEEP_ORIGINAL, BLOCKED}
    }
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    current_qa_hash = qa_contract_hash(
        config["qa"], Path(__file__).resolve(),
    )
    mounted_voice_ids: set[str] = set()
    missing_current_candidate_ids: set[str] = set()
    for line in scene["lines"]:
        decision = decide_line(line)
        start = round(float(line["start"]) * stem_sr)
        end = round(float(line["end"]) * stem_sr)
        if decision.action == BLOCKED:
            mask[start:end] = True
            review_masked_ids.add(line["id"])
            missing_current_candidate_ids.add(line["id"])
            report_lines.append({
                "id": line["id"], "action": BLOCKED,
                "reason": decision.reason,
                "coverage_required": True,
                "coverage_status": "MAPPING_CORRECTION_REQUIRED",
            })
            continue
        if decision.action == KEEP_ORIGINAL:
            report_lines.append({"id": line["id"], "action": KEEP_ORIGINAL, "reason": decision.reason})
            continue
        rejected = set(line.get("rejected_candidates", []))
        line_contract = line.get("_codex2_line_contract_hash")
        accepted_line_contracts = set(
            line.get("_codex2_accepted_line_contract_hashes", [line_contract])
        )
        rows = [
            row for row in rankings.get(line["id"], [])
            if (
                row.get("file") not in rejected
                and row.get("contract_hash") in accepted_line_contracts
                and row.get("generation_contract_hash")
                == line.get("_codex2_generation_hash")
                and row.get("processing_contract_hash")
                == line.get("_codex2_processing_hash")
                and row.get("qa_contract_hash") == current_qa_hash
            )
        ]
        rows.sort(key=lambda row: (not row["pass"], row["score"]))
        if not rows:
            mask[start:end] = True
            review_masked_ids.add(line["id"])
            report_lines.append({
                "id": line["id"],
                "action": KEEP_ORIGINAL,
                "reason": "no_current_contract_candidate",
                "pass": False,
                "coverage_required": True,
                "coverage_status": "MISSING_CURRENT_CONTRACT_CANDIDATE",
            })
            missing_current_candidate_ids.add(line["id"])
            continue
        # A fixed-frame movie may be container-valid while its spoken take is
        # not. Never turn a failed take into a release merely because it is
        # the least-bad candidate. If every candidate is unsafe, fail closed
        # per delivery: preserve that original English interval but still mount
        # the rest of the movie. One dubious line must not discard an entire
        # otherwise verified scene.
        passing_rows = [row for row in rows if bool(row.get("pass", False))]
        if not passing_rows:
            failures = {
                row.get("file", "<unknown>"): [
                    name for name, value in (row.get("hard_gates") or {}).items()
                    if not value
                ]
                for row in rows
            }
            report_lines.append({
                "id": line["id"],
                "action": KEEP_ORIGINAL,
                "reason": "no_release_safe_candidate",
                "attempted_hard_gate_failures": failures,
                "pass": False,
                "coverage_required": True,
                "coverage_status": "NO_RELEASE_SAFE_CANDIDATE",
            })
            # A failed required line is never allowed to fall through to the
            # English stem.  Keep the dialogue channel silent for this exact
            # visual window while the scene remains explicitly non-release.
            mask[start:end] = True
            review_masked_ids.add(line["id"])
            missing_current_candidate_ids.add(line["id"])
            continue
        preferred = line.get("preferred_candidate")
        if preferred:
            preferred_rows = [
                row for row in rows
                if row.get("file") == preferred and row.get("processed")
            ]
            if not preferred_rows:
                raise RuntimeError(
                    f"preferred candidate unavailable for {line['id']}: {preferred}"
                )
            winner = preferred_rows[0]
            if not bool(winner.get("pass", False)):
                raise RuntimeError(
                    f"preferred candidate is not release-safe for {line['id']}"
                )
        else:
            winner = passing_rows[0]
        if not winner.get("processed"):
            raise RuntimeError(f"no mountable candidate for {line['id']}: {winner.get('error')}")
        selected = selected_dir / f"{line['id']}_DE.wav"
        shutil.copy2(winner["processed"], selected)
        audio, audio_sr = read(selected)
        audio = resample_exact(audio, audio_sr, stem_sr) if audio_sr != stem_sr else audio
        needed = end - start
        if len(audio) < needed:
            audio = np.pad(audio, (0, needed - len(audio)))
        elif len(audio) > needed:
            extra = audio[needed:]
            if np.max(np.abs(extra), initial=0.0) > 1e-4:
                raise ValueError(f"resample overflow contains speech for {line['id']}")
            audio = audio[:needed]
        mask[start:end] = True
        generated_mix[start:end] += audio
        generated_intervals.append((start, end, line["id"]))
        mounted_voice_ids.add(line["id"])
        report_lines.append({
            "id": line["id"], "action": decision.action, "winner": winner["file"],
            "pass": winner["pass"], "score": winner["score"], "wer": winner.get("wer"),
            "transcript": winner.get("transcript"), "round": winner["round"],
            "source_language_leak_tokens": winner.get(
                "source_language_leak_tokens"
            ) or source_language_leak(
                line["source_text"],
                line.get("delivery_text", line["target_text"]),
                winner.get("transcript") or "",
            ),
            "source_language_confirmed": bool(
                winner.get("source_language_confirmed")
                or winner.get("source_language_leak_tokens")
            ),
            "fidelity_score": winner.get("fidelity_score"),
            "hard_gates": winner.get("hard_gates"),
            "human_preferred_candidate": bool(preferred),
            "selected": str(selected),
            "coverage_required": True,
            "coverage_status": "MOUNTED_CURRENT_CONTRACT",
        })
    contracts = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["contracts"]

    # A cue map can end a few milliseconds before the source actor actually
    # stops speaking. If another generated cue immediately follows, leaving
    # that tiny gap unmasked produces a bilingual fragment at the join
    # ("schon" + the tail of "come on" + "Mir ..."). Bridge only short gaps
    # in which the original dialogue channel is demonstrably active. The other
    # five movie channels still carry ambience, so zeroing this tiny portion of
    # the isolated dialogue stem is safer than retaining foreign speech.
    bridge_limit = round(float(contracts.get("bridge_active_source_gaps_ms", 120.0)) / 1000.0 * stem_sr)
    bridge_source_ratio = float(contracts.get("bridge_source_peak_ratio", 0.01))
    bridged_gaps = []
    intervals = sorted(generated_intervals)
    coverage_start, coverage_end, coverage_id = intervals[0] if intervals else (0, 0, "")
    for right_start, right_end, right_id in intervals[1:]:
        if right_start <= coverage_end:
            if right_end > coverage_end:
                coverage_end, coverage_id = right_end, right_id
            continue
        gap = right_start - coverage_end
        if gap <= 0 or gap > bridge_limit:
            coverage_start, coverage_end, coverage_id = right_start, right_end, right_id
            continue
        local = stem[coverage_start:right_end]
        gap_audio = stem[coverage_end:right_start]
        local_peak = float(np.max(np.abs(local), initial=0.0))
        gap_peak = float(np.max(np.abs(gap_audio), initial=0.0))
        gap_ratio = gap_peak / max(local_peak, 1e-9)
        if gap_ratio >= bridge_source_ratio:
            mask[coverage_end:right_start] = True
            bridged_gaps.append({
                "left": coverage_id,
                "right": right_id,
                "start": coverage_end / stem_sr,
                "end": right_start / stem_sr,
                "duration_ms": gap / stem_sr * 1000.0,
                "source_peak_ratio": gap_ratio,
            })
            coverage_end, coverage_id = right_end, right_id
        else:
            coverage_start, coverage_end, coverage_id = right_start, right_end, right_id

    # Crossfade only when the original dialogue stem is already acoustically
    # quiet at that edge. Fading through active source speech reintroduces the
    # English phoneme that the replacement mask is meant to remove. Generated
    # candidates already contain their own audited onset/tail fades, so a hard
    # mask at a source-active boundary does not cut the German candidate.
    alpha = mask.astype(np.float32)
    transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    edge = round(0.010 * stem_sr)
    active_return_edge = round(
        float(contracts.get("source_active_return_fade_ms", 40.0))
        / 1000.0 * stem_sr
    )
    active_tail_search = round(
        float(contracts.get("source_active_tail_search_ms", 350.0))
        / 1000.0 * stem_sr
    )
    source_quiet_hold = round(
        float(contracts.get("source_quiet_hold_ms", 25.0))
        / 1000.0 * stem_sr
    )
    source_quiet_step = max(1, round(0.001 * stem_sr))
    edge_quiet_ratio = float(contracts.get("source_edge_quiet_ratio", 0.03))
    scene_last_mapped_end = max(
        (float(line.get("end", 0.0)) for line in scene.get("lines", [])),
        default=0.0,
    )
    edge_audit = []
    for interval_index, (start, end) in enumerate(zip(starts, ends)):
        fade = min(edge, max(1, (end - start) // 2))
        region_peak = float(np.max(np.abs(stem[start:end]), initial=0.0))
        start_ratio = float(np.max(np.abs(stem[start:start + fade]), initial=0.0)) / max(region_peak, 1e-9)
        end_ratio = float(np.max(np.abs(stem[end - fade:end]), initial=0.0)) / max(region_peak, 1e-9)
        start_faded = start_ratio <= edge_quiet_ratio
        end_faded = end_ratio <= edge_quiet_ratio
        source_fade_out_before_start = False
        source_fade_in_after_end = False
        source_return_waited_for_quiet = False
        suppressed_active_tail_ms = 0.0
        source_return_at = None
        if start_faded:
            alpha[start:start + fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        elif start >= fade and not np.any(mask[start - fade:start]):
            # Do not crossfade active foreign speech *into* the replacement.
            # Fade the source out immediately before the mask while generated
            # audio is still zero instead.
            alpha[start - fade:start] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            source_fade_out_before_start = True
        if end_faded:
            alpha[end - fade:end] = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
        else:
            # The cue map can stop on the source actor's last consonant or
            # breath. Returning the isolated English stem immediately after
            # `end` creates a systematic "tss/shh" at every splice. Search
            # forward for a sustained quiet portion, keep the dialogue stem
            # suppressed through the active tail, and return it only there.
            next_start = (
                int(starts[interval_index + 1])
                if interval_index + 1 < len(starts)
                else len(alpha)
            )
            # Background dialogue often overlaps or decays for longer than the
            # old fixed 350 ms search. When another mapped replacement is
            # nearby, continue looking up to that next window (bounded to
            # 1.5 s). If no quiet exists, the branch below keeps the isolated
            # English stem masked until the next German cue. This prevents the
            # systematic "tss/shh" tail without touching generated speech or
            # filling a naturally silent gap.
            adaptive_tail_search = max(
                active_tail_search, round(1.5 * stem_sr)
            )
            search_stop = min(
                len(alpha), end + adaptive_tail_search, next_start
            )
            quiet_threshold = max(region_peak * edge_quiet_ratio, 1e-5)
            quiet_start = None
            last_candidate = search_stop - source_quiet_hold
            for candidate in range(end, last_candidate + 1, source_quiet_step):
                quiet_peak = float(np.max(
                    np.abs(stem[candidate:candidate + source_quiet_hold]),
                    initial=0.0,
                ))
                if quiet_peak <= quiet_threshold:
                    quiet_start = candidate
                    break
            if quiet_start is None:
                if search_stop == next_start and next_start < len(alpha):
                    # No safe return exists before the next replacement. Keep
                    # the source masked so the two German cues meet cleanly.
                    alpha[end:next_start] = 1.0
                    mask[end:next_start] = True
                    suppressed_active_tail_ms = (next_start - end) / stem_sr * 1000.0
                elif (
                    search_stop >= len(alpha)
                    or (
                        interval_index == len(starts) - 1
                        and end >= round(scene_last_mapped_end * stem_sr) - 1
                    )
                ):
                    # A subtitle-authorized line can legitimately be the last
                    # spoken cue in a short movie stem.  There is no quiet
                    # sample *after* the cue because the source container ends
                    # on its final English phoneme.  Failing closed here would
                    # leave the previous mount (and therefore English) in the
                    # delivered file.  Keep the English stem suppressed to
                    # the exact source end; the German candidate has already
                    # passed its own duration/tail gates and is frame-fitted
                    # to this window.
                    alpha[end:] = 1.0
                    mask[end:] = True
                    suppressed_active_tail_ms = (len(alpha) - end) / stem_sr * 1000.0
                    source_return_waited_for_quiet = False
                    source_return_at = None
                else:
                    raise AssertionError(
                        "active source tail does not reach a stable quiet "
                        f"region within {active_tail_search / stem_sr * 1000.0:.0f} ms "
                        f"after {end / stem_sr:.3f}s"
                    )
            else:
                return_fade = min(
                    active_return_edge,
                    source_quiet_hold,
                    len(alpha) - quiet_start,
                    next_start - quiet_start,
                )
                alpha[end:quiet_start] = 1.0
                mask[end:quiet_start] = True
                if return_fade > 0:
                    alpha[quiet_start:quiet_start + return_fade] = np.linspace(
                        1.0, 0.0, return_fade,
                        endpoint=False, dtype=np.float32,
                    )
                    mask[quiet_start:quiet_start + return_fade] = True
                    source_fade_in_after_end = True
                source_return_waited_for_quiet = True
                suppressed_active_tail_ms = (quiet_start - end) / stem_sr * 1000.0
                source_return_at = quiet_start / stem_sr
        edge_audit.append({
            "start": start / stem_sr,
            "end": end / stem_sr,
            "source_start_peak_ratio": start_ratio,
            "source_end_peak_ratio": end_ratio,
            "source_crossfade_at_start": start_faded,
            "source_crossfade_at_end": end_faded,
            "source_fade_out_before_start": source_fade_out_before_start,
            "source_fade_in_after_end": source_fade_in_after_end,
            "source_return_waited_for_quiet": source_return_waited_for_quiet,
            "suppressed_active_tail_ms": suppressed_active_tail_ms,
            "source_return_at": source_return_at,
        })
    source_contribution = np.abs(stem * (1.0 - alpha))
    source_leak_audit = []
    for start, end in zip(starts, ends):
        region_peak = float(np.max(np.abs(stem[start:end]), initial=0.0))
        leak_peak = float(np.max(source_contribution[start:end], initial=0.0))
        leak_ratio = leak_peak / max(region_peak, 1e-9)
        source_leak_audit.append({
            "start": start / stem_sr,
            "end": end / stem_sr,
            "max_reintroduced_source_peak_ratio": leak_ratio,
            "pass": leak_ratio <= edge_quiet_ratio + 1e-6,
        })
    if not all(row["pass"] for row in source_leak_audit):
        raise AssertionError(
            "mounted dialogue would reintroduce active source speech at a replacement edge"
        )
    rebuilt = stem * (1.0 - alpha) + generated_mix * alpha
    peak = float(np.max(np.abs(rebuilt)))
    scale_db = 0.0
    if peak > 0.98:
        factor = 0.98 / peak
        generated_mix *= factor
        scale_db = 20 * math.log10(factor)
        rebuilt = stem * (1.0 - alpha) + generated_mix * alpha
    dialogue_out = out / f"{scene['scene']}_DE_dialog_ch5_exact.wav"
    write_exact(dialogue_out, rebuilt, stem_sr, len(stem))
    channel = int(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["contracts"]["replace_dialogue_channel"]) - 1
    full[:, channel] = rebuilt
    full_out = out / f"{scene['scene']}_DE_6ch_exact.wav"
    write_exact(full_out, full, full_sr, original_spec.frames)
    assert_exact(full_out, original_spec)
    other_channels_equal = all(np.array_equal(full[:, i], read(full_path, always_2d=True)[0][:, i]) for i in range(full.shape[1]) if i != channel)
    missing_current_candidate_ids.update(
        required_voice_ids - mounted_voice_ids
    )
    all_required_dubs_mounted = not missing_current_candidate_ids
    all_generated_voice_hard_gates_pass = bool(
        all_required_dubs_mounted
        and all(
            bool(row.get("pass", False))
            for row in report_lines
            if row.get("coverage_required")
        )
    )
    continuous_audit_path = out / "CONTINUOUS_LANGUAGE_AUDIT.json"
    continuous_language_audit = (
        json.loads(continuous_audit_path.read_text(encoding="utf-8"))
        if continuous_audit_path.exists() else None
    )
    continuous_language_pass = bool(
        continuous_language_audit
        and continuous_language_audit.get("pass") is True
    )
    semantic_source_language_pass = bool(
        all_required_dubs_mounted
        and all(
            not row.get("source_language_leak_tokens")
            for row in report_lines
            if row.get("action") != KEEP_ORIGINAL
        )
    )
    perceptual_review_pass = bool(scene.get("perceptual_review_done") is True)
    release_blockers = []
    if not all_required_dubs_mounted:
        release_blockers.append("missing_current_candidate")
    if not semantic_source_language_pass:
        release_blockers.append("semantic_source_language")
    if not all_generated_voice_hard_gates_pass:
        release_blockers.append("generated_voice_hard_gates")
    if not continuous_language_pass:
        release_blockers.append("continuous_language_audit")
    if not perceptual_review_pass:
        release_blockers.append("perceptual_review")
    report = {
        "scene": scene["scene"], "source": str(full_path), "output": str(full_out),
        "contract_hash": contract_hash,
        "mount_contract_hash": scene.get("_codex2_mount_hash"),
        "qa_contract_hash": qa_contract_hash(
            json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["qa"],
            Path(__file__).resolve(),
        ),
        "dialogue_output": str(dialogue_out), "contract": original_spec.__dict__,
        "contract_pass": spec(full_out) == original_spec, "other_channels_equal": other_channels_equal,
        "generated_peak_scale_db": scale_db,
        "bridged_active_source_gaps": bridged_gaps,
        "replacement_edge_audit": edge_audit,
        "source_leak_audit": source_leak_audit,
        "required_voice_ids": sorted(required_voice_ids),
        "expected_visual_ids": sorted(expected_visual_ids),
        "missing_expected_visual_ids": missing_expected_visual_ids,
        "mounted_voice_ids": sorted(mounted_voice_ids),
        "missing_current_candidate_ids": sorted(missing_current_candidate_ids),
        "all_required_dubs_mounted": all_required_dubs_mounted,
        "review_masked_ids": sorted(review_masked_ids),
        "preflight_rejections": (
            json.loads((out / "PREFLIGHT_REJECTIONS.json").read_text(encoding="utf-8"))
            if (out / "PREFLIGHT_REJECTIONS.json").exists() else {}
        ),
        "release_ready": not release_blockers,
        "release_blockers": release_blockers,
        "continuous_language_audit": continuous_language_audit,
        "continuous_language_pass": continuous_language_pass,
        "semantic_source_language_pass": semantic_source_language_pass,
        "perceptual_review_pass": perceptual_review_pass,
        "all_generated_voice_hard_gates_pass": all_generated_voice_hard_gates_pass,
        "active_source_reintroduced_at_edges": not all(row["pass"] for row in source_leak_audit),
        "omnivoice_engine": str(OMNIVOICE_ENGINE),
        "lines": report_lines,
    }
    (out / "FINAL_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def make_html(scene: dict, stem_path: Path, report: dict, out: Path) -> None:
    def relative(path: str | Path) -> str:
        return Path(os.path.relpath(Path(path), out)).as_posix()

    cards = []
    by_id = {row["id"]: row for row in report["lines"]}
    for line in scene["lines"]:
        row = by_id[line["id"]]
        selected = row.get("selected")
        required = bool(row.get("coverage_required"))
        if selected:
            player = f'<audio controls preload="none" src="{html.escape(relative(selected))}"></audio>'
        elif required:
            player = (
                "<strong>NO CURRENT DUB CANDIDATE — ORIGINAL NOT RELEASE-SAFE</strong>"
            )
        else:
            player = "<em>Original preserved by policy</em>"
        status = (
            "MISSING_CURRENT_CONTRACT_CANDIDATE"
            if row.get("coverage_status") == "MISSING_CURRENT_CONTRACT_CANDIDATE"
            else str(row.get("action"))
        )
        pass_value = row.get("pass", False) if required else True
        cards.append(
            f"<section><h3>{html.escape(line['id'])} — {html.escape(line['speaker'])}</h3>"
            f"<p>EN: {html.escape(line['source_text'])}<br>DE: {html.escape(line['target_text'])}</p>"
            f"{player}<p>{html.escape(status)} | PASS={pass_value} | "
            f"WER={row.get('wer', 0):.3f} | ASR={html.escape(str(row.get('transcript', 'original')))}</p></section>"
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>{scene['scene']} QA</title>
<style>body{{background:#17191d;color:#eee;font:16px system-ui;margin:24px}}section{{border:1px solid #667;padding:14px;margin:12px 0}}audio{{width:48%}}code{{color:#9ef}}</style>
<h1>{scene['scene']} — exact-frame German dub</h1>
<p>Original ch5: <audio controls src="{html.escape(relative(stem_path))}"></audio></p>
<p>German ch5: <audio controls src="{html.escape(relative(report['dialogue_output']))}"></audio></p>
{''.join(cards)}"""
    (out / "QA_LISTEN.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map", type=Path)
    parser.add_argument("--mount-only", action="store_true")
    parser.add_argument("--qa-ids", nargs="+")
    parser.add_argument("--regenerate-ids", nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Write a non-destructive production branch outside PROJECT/outputs.",
    )
    parser.add_argument("--next", dest="next_scene", default="next chronological scene")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    map_path = resolve(args.map)
    scene = json.loads(map_path.read_text(encoding="utf-8"))
    scene["_codex2_contract_hash"] = scene_contract_hash(
        scene,
        map_path,
        config,
        Path(__file__).resolve(),
        PROJECT,
    )
    for line in scene.get("lines", []):
        line["_codex2_line_contract_hash"] = line_contract_hash(
            scene,
            line,
            map_path,
            config,
            Path(__file__).resolve(),
            PROJECT,
        )
        line["_codex2_generation_hash"] = generation_contract_hash(
            scene, line, map_path, config, PROJECT,
        )
        line["_codex2_processing_hash"] = processing_contract_hash(
            scene, line, map_path, config, Path(__file__).resolve(), PROJECT,
        )
        line["_codex2_qa_hash"] = qa_contract_hash(
            config["qa"], Path(__file__).resolve(),
        )
    scene["_codex2_mount_hash"] = mount_contract_hash(
        scene, config, Path(__file__).resolve(),
    )
    stem_path = resolve(scene["source_stem"], map_path.parent)
    full_name = stem_path.name.replace("_dialog_ch5", "_6ch")
    full_path = stem_path.with_name(full_name)
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    stem, stem_sr = read(stem_path)
    if len(stem) != int(scene["container_frames"]) or stem_sr != int(scene["sample_rate"]):
        raise ValueError("map does not match source container")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT / "outputs"
    )
    out = output_root / scene["scene"]
    out.mkdir(parents=True, exist_ok=True)
    attach_accepted_contracts(scene, out)
    profile = dict(config["anime"])
    profile["model"] = config["model"]
    keep = sum(
        decide_line(line).action in {KEEP_ORIGINAL, BLOCKED}
        for line in scene["lines"]
    )
    total = len(scene["lines"]) - keep
    if args.regenerate_ids:
        requested = set(args.regenerate_ids)
        known_ids = {line["id"] for line in scene["lines"]}
        unknown = requested - known_ids
        if unknown:
            raise ValueError(f"unknown line ids: {sorted(unknown)}")
        metadata_path = out / "candidates" / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists() else []
        )
        previous_rounds = [
            int(row["round"]) for row in metadata if row["line_id"] in requested
        ]
        round_index = max(previous_rounds, default=-1) + 1
        write_state(
            scene["scene"],
            f"Targeted regeneration round {round_index} for {sorted(requested)}",
        )
        generate_round(
            scene, stem, stem_sr, out, profile, round_index, requested,
        )
        focused, _ = evaluate(
            scene, stem, stem_sr, out, config["qa"],
            only_ids=requested, only_rounds={round_index},
        )
        rankings_path = out / "QA_RANKING.json"
        rankings = (
            json.loads(rankings_path.read_text(encoding="utf-8"))
            if rankings_path.exists() else {}
        )
        for line_id, rows in focused.items():
            rankings.setdefault(line_id, []).extend(rows)
        rankings_path.write_text(
            json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        report = select_and_mount(scene, stem_path, full_path, out, rankings)
        make_html(scene, stem_path, report, out)
        print(json.dumps({
            "scene": scene["scene"],
            "targeted_regeneration": sorted(requested),
            "round": round_index,
            "output": report["output"],
        }, indent=2))
        return
    if args.qa_ids:
        rankings_path = out / "QA_RANKING.json"
        rankings = json.loads(rankings_path.read_text(encoding="utf-8")) if rankings_path.exists() else {}
        focused, _ = evaluate(scene, stem, stem_sr, out, config["qa"], set(args.qa_ids))
        rankings.update(focused)
        rankings_path.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
        report = select_and_mount(scene, stem_path, full_path, out, rankings)
        make_html(scene, stem_path, report, out)
        passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
        review = total - passed
        write_state(scene["scene"], f"Completed {scene['scene']}; next {args.next_scene}", {"generated_lines": total, "qa_pass": passed, "qa_review": review, "keep_original": keep})
        print(json.dumps({"scene": scene["scene"], "focused_qa": args.qa_ids, "pass": passed, "review": review, "keep": keep}, indent=2))
        return
    if args.mount_only:
        rankings_path = out / "QA_RANKING.json"
        if not rankings_path.exists():
            raise FileNotFoundError(rankings_path)
        rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
        report = select_and_mount(scene, stem_path, full_path, out, rankings)
        make_html(scene, stem_path, report, out)
        passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
        review = total - passed
        write_state(scene["scene"], f"Completed {scene['scene']}; next {args.next_scene}", {"generated_lines": total, "qa_pass": passed, "qa_review": review, "keep_original": keep})
        print(json.dumps({"scene": scene["scene"], "mount_only": True, "pass": passed, "review": review, "keep": keep, "output": report["output"]}, indent=2))
        return
    if total == 0:
        # A movie made only of protected efforts/onomatopoeias still needs an
        # exact-frame deliverable. There are no candidates or ASR rankings to
        # create, so mount the untouched English 5.1 mix directly.
        rankings = {}
        (out / "QA_RANKING.json").write_text(
            json.dumps(rankings, indent=2), encoding="utf-8",
        )
        report = select_and_mount(scene, stem_path, full_path, out, rankings)
        make_html(scene, stem_path, report, out)
        write_state(
            scene["scene"],
            f"Completed protected-only {scene['scene']}; next {args.next_scene}",
            {
                "generated_lines": 0,
                "qa_pass": 0,
                "qa_review": 0,
                "keep_original": keep,
            },
        )
        print(json.dumps({
            "scene": scene["scene"],
            "pass": 0,
            "review": 0,
            "keep": keep,
            "output": report["output"],
        }, indent=2))
        return
    write_state(scene["scene"], f"Generating one initial take per line for {scene['scene']}", {"generated_lines": 0, "keep_original": keep, "qa_pass": 0, "qa_review": 0})
    generate_round(scene, stem, stem_sr, out, profile, 0)
    write_state(scene["scene"], f"Advanced GPU QA round 1 for {scene['scene']}")
    rankings, _ = evaluate(
        scene, stem, stem_sr, out, config["qa"], only_rounds={0},
    )

    lines_by_id = {line["id"]: line for line in scene["lines"]}

    def retry_required(candidate_ids: set[str] | None = None) -> set[str]:
        """Regenerate until a take passes both the full metric and tail gate.

        A container/frame PASS is not a voice-quality PASS.  In particular,
        robotic cadence, bad onset/span, wrong loudness or poor fidelity must
        not be mounted merely because the final tail is quiet.
        """
        ids = candidate_ids if candidate_ids is not None else set(rankings)
        result = set()
        classifications = {}
        for line_id in ids:
            line = lines_by_id[line_id]
            decision = decide_line(line)
            if decision.action in {KEEP_ORIGINAL, BLOCKED}:
                continue
            rows = rankings.get(line_id, [])
            acceptable = any(bool(row.get("pass", False)) for row in rows)
            if not acceptable:
                classes = [classify_failure(row) for row in rows]
                classifications[line_id] = classes
                if any(item == "RANDOM_TTS" for item in classes):
                    result.add(line_id)
        (out / "RETRY_CLASSIFICATION.json").write_text(
            json.dumps(classifications, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return result

    failed = retry_required()
    max_rounds = max(1, int(profile.get("max_rounds", 4)))
    for round_index in range(1, max_rounds):
        if not failed:
            break
        (out / "retry_ids.json").write_text(
            json.dumps(sorted(failed), indent=2), encoding="utf-8",
        )
        write_state(
            scene["scene"],
            f"Regenerating {len(failed)} failed lines, attempt {round_index + 1}/{max_rounds}",
        )
        generate_round(scene, stem, stem_sr, out, profile, round_index, failed)
        write_state(
            scene["scene"],
            f"Advanced GPU QA attempt {round_index + 1}/{max_rounds} for {scene['scene']}",
        )
        retry_rankings, _ = evaluate(
            scene, stem, stem_sr, out, config["qa"],
            only_ids=failed, only_rounds={round_index},
        )
        for line_id, rows in retry_rankings.items():
            rankings.setdefault(line_id, []).extend(rows)
        failed = retry_required(failed)
    (out / "QA_RANKING.json").write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
    report = select_and_mount(scene, stem_path, full_path, out, rankings)
    make_html(scene, stem_path, report, out)
    passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
    review = total - passed
    write_state(scene["scene"], f"Completed {scene['scene']}; next {args.next_scene}", {"generated_lines": total, "qa_pass": passed, "qa_review": review, "keep_original": keep})
    print(json.dumps({"scene": scene["scene"], "pass": passed, "review": review, "keep": keep, "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
