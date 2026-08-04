#!/usr/bin/env python3
"""Continuous final-scene language audit for anime/FMV dialogue stems.

The per-candidate source-language gate cannot see English left in an unmapped
gap or in a preserved lexical line.  This pass decodes the mounted dialogue
stem in short windows and fails closed on sentence-level English.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


ENGLISH_MARKERS = {
    "a", "am", "and", "are", "because", "being", "can", "can't", "did",
    "do", "don't", "for", "from", "have", "he", "how", "i", "if", "in",
    "is", "it", "like", "nearby", "no", "not", "now", "of", "okay", "one", "yes",
    "only", "remember", "suitable", "the", "this", "to", "vessel", "was",
    "what", "when", "where", "who", "why", "with", "would", "you", "your",
}
APPROVED_TERMS = {"dark", "hour", "tartarus", "palladion", "persona", "arcana"}
STRONG_SOURCE_WORDS = {
    "created", "machine", "serve", "specific", "purpose", "nearby",
    "suitable", "vessel", "destroy", "existing", "reason",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def _is_english_sentence(text: str, language: str | None, probability: float) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 2:
        return False
    residual = [token for token in tokens if token not in APPROVED_TERMS]
    marker_count = sum(token in ENGLISH_MARKERS for token in residual)
    if language == "en" and probability >= 0.70:
        # A two-token Whisper hallucination such as ``to full`` is a common
        # false alarm at a German cue boundary. Require two English markers
        # for short text; a longer sentence still needs only one marker.
        if marker_count >= 2 or (marker_count >= 1 and len(residual) >= 3):
            return True
    # A high-confidence German decode containing common English markers is
    # not, by itself, an English leak: ``was``, ``in`` and ``oh`` are normal
    # German words and previously produced false alarms in 140_170/190_140.
    if language in {"de", "deu"}:
        return False
    return marker_count >= 2 and len(residual) >= 3


def _overlapping_lines(
    scene: dict | None, start: float, end: float,
) -> list[dict]:
    if scene is None:
        return []
    return [
        line for line in scene.get("lines", [])
        if line.get("force_clone") is True
        and line.get("subtitle_authorized", True) is not False
        and float(line.get("start", 0.0) or 0.0) < end
        and float(line.get("end", 0.0) or 0.0) > start
    ]


def _subtitle_required_intervals(scene: dict | None) -> list[tuple[float, float]] | None:
    """Return only intervals whose text is expected to be German.

    ``None`` keeps the legacy whole-stem behavior for callers without a map.
    A visible subtitle that was deliberately classified KEEP_ORIGINAL is
    reported by the mapping audit, but it is not a leak in the mounted German
    replacement.  Only ``force_clone`` rows therefore enter this continuous
    language gate; audio-only/background rows never do.
    """
    if scene is None:
        return None
    intervals: list[tuple[float, float]] = []
    preserved: list[tuple[float, float]] = []
    for line in scene.get("lines", []):
        if line.get("force_clone") is not True:
            continue
        if line.get("subtitle_authorized", True) is False:
            continue
        if line.get("force_keep_original") and line.get("mapping_validation") == "NO_VISIBLE_SUBTITLE_CARD":
            continue
        # Audit the actual replacement speech span, not the full subtitle
        # card.  A subtitle window can deliberately extend past the spoken
        # German edge into an adjacent KEEP_ORIGINAL line; auditing that
        # whole card would misattribute the preserved English to the dub
        # (notably 200_130_M_L025 immediately before legacy L026).  Maps
        # without independent speech edges still fall back to their cue
        # bounds.
        start = float(line.get("speech_start", line.get("start", 0.0)) or 0.0)
        end = float(line.get("speech_end", line.get("end", 0.0)) or 0.0)
        if end > start:
            intervals.append((start, end))
        for item in line.get("preserved_source_intervals", []) or []:
            try:
                left = max(start, float(item.get("start", start)))
                right = min(end, float(item.get("end", end)))
            except (TypeError, ValueError):
                continue
            if right > left:
                preserved.append((left, right))
    # Preserve-original Empalme-B prefixes are intentionally English source
    # audio. Remove only those exact authorized intervals from the German
    # language audit; the rest of the subtitle window remains audited.
    for left, right in preserved:
        remainder: list[tuple[float, float]] = []
        for start, end in intervals:
            if right <= start or left >= end:
                remainder.append((start, end))
                continue
            if start < left:
                remainder.append((start, left))
            if right < end:
                remainder.append((right, end))
        intervals = [(start, end) for start, end in remainder if end > start]
    return intervals


def audit_scene(
    output_dir: Path,
    model: WhisperModel,
    window_seconds: float = 5.0,
    scene: dict | None = None,
) -> dict:
    report_path = output_dir / "FINAL_REPORT.json"
    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    dialogue_path = Path(existing.get("dialogue_output", ""))
    if not dialogue_path.exists():
        matches = sorted(output_dir.glob("*_DE_dialog_ch5_exact.wav"))
        dialogue_path = matches[-1] if matches else Path()
    if not dialogue_path.exists():
        result = {"scene": output_dir.name, "pass": False, "error": "missing_dialogue_output"}
        (output_dir / "CONTINUOUS_LANGUAGE_AUDIT.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    audio, sr = sf.read(dialogue_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    rows = []
    required_intervals = _subtitle_required_intervals(scene)
    window_frames = max(1, round(window_seconds * sr))
    audit_dir = output_dir / "_continuous_audit_windows"
    audit_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, start in enumerate(range(0, len(audio), window_frames)):
            end = min(len(audio), start + window_frames)
            raw_clip = audio[start:end]
            window_start = start / sr
            window_end = end / sr
            audited = (
                required_intervals is None
                or any(window_start < interval_end and window_end > interval_start
                       for interval_start, interval_end in required_intervals)
            )
            # A fixed five-second Whisper window can straddle an authorized
            # subtitle cue and an adjacent KEEP_ORIGINAL/background cue.  If
            # we transcribe that raw window, the English outside the subtitle
            # interval is falsely attributed to the German replacement.  Keep
            # the scan cadence, but zero every sample outside the authoritative
            # subtitle intervals before language classification.
            if required_intervals is not None:
                clip = np.zeros_like(raw_clip)
                for interval_start, interval_end in required_intervals:
                    left = max(start, round(interval_start * sr))
                    right = min(end, round(interval_end * sr))
                    if right > left:
                        clip[left - start:right - start] = audio[left:right]
            else:
                clip = raw_clip
            if float(np.max(np.abs(clip), initial=0.0)) <= 1e-5:
                continue
            clip_path = audit_dir / f"window_{index:05d}.wav"
            sf.write(clip_path, clip, sr, subtype="PCM_16")
            segments, info = model.transcribe(
                str(clip_path), language=None, beam_size=1,
                vad_filter=True, condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            language = getattr(info, "language", None)
            probability = float(getattr(info, "language_probability", 0.0) or 0.0)
            overlapping = _overlapping_lines(scene, window_start, window_end)
            expected_text = " ".join(
                str(line.get("delivery_text", line.get("target_text", "")))
                for line in overlapping
            ).strip()
            source_text = " ".join(
                str(line.get("source_text", "")) for line in overlapping
            ).strip()
            suspicious = audited and _is_english_sentence(
                text, language, probability,
            )
            confirmation_text = None
            confirmation_language = None
            source_only_strong_tokens: list[str] = []
            target_overlap_tokens: list[str] = []
            # Language identification can call a German cue English when a
            # proper name is adjacent to a clipped word (e.g. Yakushima +
            # ``Du klingst``). Confirm every suspect with a forced German
            # decode and the authoritative target/source text for the window.
            # Conversely, a strong source-only English content word such as
            # ``created`` remains a leak even when Whisper labels the window
            # German.
            if audited and suspicious:
                confirm_segments, _ = model.transcribe(
                    str(clip_path), language="de", beam_size=5,
                    vad_filter=True, condition_on_previous_text=False,
                )
                confirmation_text = " ".join(
                    segment.text.strip() for segment in confirm_segments
                ).strip()
                confirmation_language = "de"
            elif audited and language in {"de", "deu"}:
                # A German-labelled window does not need a second expensive
                # decode. Its first transcript is sufficient for the
                # source-only strong-word check below.
                confirmation_text = text
                confirmation_language = language
            if confirmation_text is not None:
                confirmed_tokens = set(_tokens(confirmation_text))
                expected_tokens = set(_tokens(expected_text))
                source_tokens = set(_tokens(source_text))
                target_overlap_tokens = sorted(
                    confirmed_tokens & expected_tokens
                )
                source_only_strong_tokens = sorted(
                    (confirmed_tokens & source_tokens & STRONG_SOURCE_WORDS)
                    - APPROVED_TERMS
                )
                if source_only_strong_tokens:
                    suspicious = True
                elif suspicious and target_overlap_tokens:
                    # The forced German decode agrees with at least one
                    # authoritative target token; short five-second windows
                    # often contain only a proper name plus one clipped
                    # function word (for example ``Yukari ... Bitte``).  The
                    # first language-agnostic result is then a mixed-name
                    # false positive, not an English leak.  Strong
                    # source-only words above still fail closed regardless of
                    # this relaxed short-window confirmation.
                    suspicious = False
            rows.append({
                "start": round(window_start, 3),
                "end": round(window_end, 3),
                "language": language,
                "language_probability": probability,
                "text": text,
                "audited": audited,
                "ignored_reason": None if audited else "no_subtitle_authorized_line_in_window",
                "english_suspect": suspicious,
                "confirmation_language": confirmation_language,
                "confirmation_text": confirmation_text,
                "target_overlap_tokens": target_overlap_tokens,
                "source_only_strong_tokens": source_only_strong_tokens,
            })
    finally:
        for path in audit_dir.glob("*.wav"):
            path.unlink(missing_ok=True)
        audit_dir.rmdir()
    suspects = [row for row in rows if row["english_suspect"]]
    result = {
        "scene": output_dir.name,
        "source": str(dialogue_path),
        "window_seconds": window_seconds,
        "subtitle_policy": "SUBTITLE_AUTHORIZED_ONLY" if scene is not None else "UNSCOPED_LEGACY",
        "subtitle_required_intervals": required_intervals,
        "preserved_source_intervals": [
            item
            for line in (scene or {}).get("lines", [])
            for item in (line.get("preserved_source_intervals", []) or [])
        ],
        "windows": rows,
        "english_suspects": suspects,
        "pass": not suspects,
    }
    (output_dir / "CONTINUOUS_LANGUAGE_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def audit_output_root(output_root: Path, config: dict) -> list[dict]:
    qa = config["qa"]
    model = WhisperModel(
        qa.get("asr_model", "large-v3-turbo"),
        device=qa.get("asr_device", "cuda"),
        compute_type=qa.get("asr_compute_type", "float16"),
    )
    return [
        audit_scene(path, model)
        for path in sorted(output_root.iterdir())
        if path.is_dir() and (path / "FINAL_REPORT.json").exists()
    ]


__all__ = ["audit_scene", "audit_output_root"]
