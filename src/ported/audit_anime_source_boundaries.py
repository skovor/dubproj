"""Audit anime/FMVs without treating subtitle cards as delivery boundaries.

This is a read-only evidence report.  It classifies non-lexical effort by
position and checks generated (force_clone) windows against the dry English
stem.  A post-card rise is treated as a new event, not as the previous line's
tail; only a decaying release is proposed for extension.
"""
from __future__ import annotations

import csv
import json
import math
import re
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAPS = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v2"
SOURCE_DIR = ROOT / "anime_all_20260725" / "inventory" / "movie_audio"
ASR_DIR = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "ANIME_LARGE_V3_ASR_EVIDENCE"
OUT_DIR = ROOT / "P3R_ANIME_DELIVERY_ALIGNED_V2_20260801" / "boundary_audit"

EFFORT_WORDS = {
    "ah", "agh", "argh", "augh", "ugh", "uh", "urgh", "ngh", "hng",
    "nng", "gah", "gugh", "grr", "grrr", "hmpf", "ow", "oof", "oww",
    "aua", "autsch", "keuch", "stöhn", "stohn", "ächz", "achz", "seufz",
    "schluchz", "winsel", "pff", "pfff", "puh",
}
TOKEN = re.compile(r"[^\W\d_]+(?:[.!?…]+)?", re.UNICODE)


def rms(values: list[int | float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values) / max(1, len(values)))


def read_mono(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as w:
        rate, channels, frames = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(frames)
    import struct
    vals = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 1:
        return rate, [float(x) for x in vals]
    return rate, [float(max(abs(vals[i + c]) for c in range(channels))) for i in range(0, len(vals), channels)]


def window_rms(samples: list[float], rate: int, start: float, end: float, step: float = 0.02) -> list[tuple[float, float]]:
    a, b = max(0, int(start * rate)), min(len(samples), int(end * rate))
    n = max(1, int(step * rate))
    return [(start + i / rate, rms(samples[i:i + n])) for i in range(a, b, n) if samples[i:i + n]]


def token_texts(text: str) -> list[str]:
    return [re.sub(r"[^\wäöüÄÖÜß]", "", t, flags=re.UNICODE).casefold() for t in TOKEN.findall(text)]


def is_effort_token(token: str) -> bool:
    collapsed = re.sub(r"(.)\1+", r"\1", token.casefold())
    if collapsed in EFFORT_WORDS:
        return True
    # Do not use a broad vowel-only heuristic here: it misclassifies normal
    # German words such as "einer", "ihre" and "ehren".  Elongated tokens
    # are already reduced by ``collapsed`` and still match the lexicon.
    return False


def classify_text(source_text: str, target_text: str, force_keep: bool) -> dict:
    source_tokens, target_tokens = token_texts(source_text), token_texts(target_text)
    flags = [is_effort_token(t) for t in target_tokens]
    if not flags or not any(flags):
        content_class, position = "LEXICAL", "none"
    elif all(flags):
        content_class, position = "NONLEXICAL_PURE", "pure"
    elif flags[0]:
        content_class, position = "MIXED_LEADING_EFFORT", "leading"
    elif flags[-1]:
        content_class, position = "MIXED_TRAILING_EFFORT", "trailing"
    else:
        content_class, position = "MIXED_INTERNAL_EFFORT", "internal"
    source_leading = bool(source_tokens and is_effort_token(source_tokens[0]))
    preserve = bool(
        content_class == "NONLEXICAL_PURE"
        or (content_class == "MIXED_LEADING_EFFORT" and source_leading)
        or force_keep
    )
    tts_text = target_text
    if content_class == "MIXED_LEADING_EFFORT" and preserve:
        match = re.match(r"\s*[^\s]+\s*", target_text)
        if match:
            tts_text = target_text[match.end():].lstrip(" ,;:—–-…")
    return {
        "content_class": content_class,
        "onomatopoeia_position": position,
        "source_effort_token": source_tokens[0] if source_leading else "",
        "target_effort_token": target_tokens[0] if flags and flags[0] else "",
        "preserve_original_effort": preserve,
        "tts_text_without_effort": tts_text,
        "classification_confidence": "text_plus_source" if preserve else "text_candidate",
    }


def norm_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def asr_tail_evidence(scene: str, source_text: str, card_end: float) -> tuple[float, str]:
    """Use ASR only as corroboration when it matches this line's text.

    A neighbouring segment is not allowed to extend a card merely because it
    is loud.  At least half of the source words must occur in the segment, and
    the segment must overlap the card's end.  The waveform remains the final
    boundary authority; this field only says that a human-safe window review
    is warranted.
    """
    path = ASR_DIR / f"{scene}_large_v3.json"
    if not path.is_file():
        return 0.0, "asr_unavailable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0, "asr_unreadable"
    expected = norm_words(source_text)
    if not expected:
        return 0.0, "no_source_words"
    required = max(1, math.ceil(len(expected) * 0.5))
    matches = []
    for segment in payload.get("segments", []):
        heard = norm_words(str(segment.get("text", "")))
        overlap = len(expected & heard)
        start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        if overlap >= required and start < card_end + 0.10 and end > card_end + 0.03:
            matches.append(end)
    if not matches:
        return 0.0, "no_matching_segment_tail"
    return round(max(0.0, (max(matches) - card_end) * 1000.0), 1), "asr_matching_segment_tail"


def extension_after(samples: list[float], rate: int, card_start: float, card_end: float, next_start: float | None) -> dict:
    limit = min(card_end + 0.75, next_start if next_start is not None else card_end + 0.75)
    local = [v for _, v in window_rms(samples, rate, max(0, card_start), card_end)]
    after = window_rms(samples, rate, card_end, limit)
    peak = max(local or [0.0])
    threshold = max(18.0, peak * 0.01)
    values = [value for _, value in after]
    if len(values) < 3 or max(values[:3], default=0.0) < threshold:
        return {"source_tail_after_card_ms": 0.0, "threshold": threshold, "evidence": "none"}
    peak_index = max(range(min(len(values), 12)), key=lambda i: values[i])
    if peak_index >= 2 and values[peak_index] > max(values[0], 1.0) * 3.5:
        return {"source_tail_after_card_ms": 0.0, "threshold": threshold, "evidence": "next_event_rise"}
    tail = [idx for idx, value in enumerate(values) if idx >= peak_index and value >= threshold]
    if not tail:
        return {"source_tail_after_card_ms": 0.0, "threshold": threshold, "evidence": "none"}
    return {"source_tail_after_card_ms": round((max(tail) + 1) * 20.0, 1), "threshold": threshold, "evidence": "release_tail_candidate"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for map_path in sorted(MAPS.glob("*_map.json")):
        scene = json.loads(map_path.read_text(encoding="utf-8"))
        source = SOURCE_DIR / f"{scene['scene']}_EN_dialog_ch5.wav"
        if not source.is_file():
            continue
        rate, samples = read_mono(source)
        lines = scene.get("lines", [])
        for index, line in enumerate(lines):
            target, source_text = str(line.get("target_text", "")), str(line.get("source_text", ""))
            classification = classify_text(source_text, target, bool(line.get("force_keep_original")))
            next_start = float(lines[index + 1]["start"]) if index + 1 < len(lines) else None
            tail = extension_after(samples, rate, float(line["start"]), float(line["end"]), next_start) if line.get("force_clone") else {"source_tail_after_card_ms": 0.0, "threshold": 0.0, "evidence": "not_generated_or_keep_original"}
            asr_tail_ms, asr_evidence = asr_tail_evidence(scene["scene"], source_text, float(line["end"])) if line.get("force_clone") else (0.0, "not_generated_or_keep_original")
            rows.append({
                "scene": scene["scene"], "line_id": line["id"],
                "card_start_s": line["start"], "card_end_s": line["end"],
                "source_text": source_text, "target_text": target,
                "force_clone": bool(line.get("force_clone")),
                "force_keep_original": bool(line.get("force_keep_original")),
                **classification,
                "nonlexical_candidate": classification["content_class"] != "LEXICAL",
                "nonlexical_token": classification["target_effort_token"],
                **tail,
                "waveform_tail_candidate_ms": tail["source_tail_after_card_ms"],
                "asr_matching_segment_tail_ms": asr_tail_ms,
                "asr_boundary_evidence": asr_evidence,
                "needs_window_repair": asr_tail_ms >= 35.0,
                "needs_context_review": classification["content_class"] != "LEXICAL" or asr_tail_ms >= 35.0,
            })
    fields = list(rows[0]) if rows else []
    csv_path = OUT_DIR / "ANIME_BOUNDARY_AND_NONLEXICAL_AUDIT.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    nonlexical = [r for r in rows if r["nonlexical_candidate"]]
    repairs = [r for r in rows if r["needs_window_repair"]]
    summary = {
        "rows": len(rows), "scenes": len({r["scene"] for r in rows}),
        "nonlexical_candidates": len(nonlexical),
        "by_content_class": {k: sum(r["content_class"] == k for r in rows) for k in sorted({r["content_class"] for r in rows})},
        "window_repair_candidates": len(repairs),
        "waveform_only_candidates": sum(r["waveform_tail_candidate_ms"] >= 35.0 and not r["needs_window_repair"] for r in rows),
        "window_repair_examples": repairs,
        "rule": "subtitle cards authorize content; source waveform and next-event evidence decide delivery edges; preserve source efforts and synthesize only lexical remainder",
    }
    (OUT_DIR / "ANIME_BOUNDARY_AND_NONLEXICAL_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT_DIR / "NONLEXICAL_REVIEW.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(nonlexical)
    with (OUT_DIR / "WINDOW_REPAIR_CANDIDATES.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(repairs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
