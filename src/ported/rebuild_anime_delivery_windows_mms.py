#!/usr/bin/env python3
"""Propose true anime delivery windows from constrained phonetic alignment.

Subtitle cards authorize text and provide a coarse neighborhood. They are not
speech boundaries. This script aligns each official English line against a
broad slice of the dry dialogue stem, then refines the first/last aligned word
to stable acoustic silence. It only writes a proposal report; maps remain
untouched until confidence and ordering gates pass.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
OUT = ROOT / "ANIME_FORCED_ALIGNMENT_WINDOW_PROPOSALS.json"
DEVICE = "cuda"
TARGET_SR = 16000
CONTEXT_BEFORE = 1.0
CONTEXT_AFTER = 3.0
QUIET_HOLD = 0.025
ENERGY_HOP = 0.005
EDGE_SEARCH = 1.50


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def words(text: str) -> list[str]:
    value = fold(text)
    spoken_numbers = {
        "13th": "thirteenth",
    }
    for written, spoken in spoken_numbers.items():
        value = re.sub(rf"\b{re.escape(written)}\b", spoken, value)
    value = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", value)
    return re.findall(r"[a-z0-9]+", value)


def rms_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    hop = max(1, round(ENERGY_HOP * sr))
    values = np.array([
        float(np.sqrt(np.mean(audio[pos:pos + hop] ** 2)))
        for pos in range(0, len(audio), hop)
    ], dtype=np.float32)
    return values, hop


def refine_edges(
    audio: np.ndarray,
    sr: int,
    aligned_start: float,
    aligned_end: float,
) -> tuple[float, float, dict]:
    envelope, hop = rms_envelope(audio, sr)
    peak = float(envelope.max(initial=0.0))
    threshold = max(1e-4, peak * 10 ** (-40.0 / 20.0))
    quiet_frames = max(1, round(QUIET_HOLD * sr / hop))
    start_frame = max(0, min(len(envelope) - 1, int(aligned_start * sr / hop)))
    end_frame = max(0, min(len(envelope), int(np.ceil(aligned_end * sr / hop))))
    search_frames = max(1, round(EDGE_SEARCH * sr / hop))

    onset = start_frame
    start_valley_found = False
    lower = max(0, start_frame - search_frames)
    for pos in range(start_frame, lower - 1, -1):
        q0 = max(lower, pos - quiet_frames)
        if pos - q0 >= quiet_frames and np.all(envelope[q0:pos] < threshold):
            onset = pos
            start_valley_found = True
            break

    release = end_frame
    end_valley_found = False
    upper = min(len(envelope), end_frame + search_frames)
    for pos in range(end_frame, max(end_frame, upper - quiet_frames) + 1):
        if pos + quiet_frames <= len(envelope) and np.all(envelope[pos:pos + quiet_frames] < threshold):
            release = pos
            end_valley_found = True
            break
    return (
        onset * hop / sr,
        release * hop / sr,
        {
            "local_peak": peak,
            "quiet_threshold": threshold,
            "start_valley_found": start_valley_found,
            "end_valley_found": end_valley_found,
            "quiet_hold_seconds": QUIET_HOLD,
            "edge_search_seconds": EDGE_SEARCH,
        },
    )


def main() -> int:
    torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 1))))
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(DEVICE).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    rows: list[dict] = []
    for map_path in sorted((ROOT / "maps").glob("*_map.json")):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
        if source.ndim > 1:
            source = source.mean(axis=1)
        duration = len(source) / sr
        lines = sorted(mapping["lines"], key=lambda row: float(row["start"]))
        for line_index, line in enumerate(lines):
            tokens = words(line.get("source_text", ""))
            base = {
                "scene": mapping["scene"], "line_id": line["id"],
                "text_en": line.get("source_text", ""),
                "speaker": line.get("speaker"),
                "card_start": float(line["start"]), "card_end": float(line["end"]),
            }
            if not tokens:
                rows.append({**base, "status": "NO_LEXICAL_TOKENS"})
                continue
            context_start = max(0.0, float(line["start"]) - CONTEXT_BEFORE)
            context_end = min(duration, float(line["end"]) + CONTEXT_AFTER)
            clip = source[round(context_start * sr):round(context_end * sr)]
            waveform = torch.from_numpy(np.asarray(clip, dtype=np.float32))
            waveform16 = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            try:
                with torch.inference_mode():
                    emission, _ = model(waveform16.unsqueeze(0).to(DEVICE))
                spans = aligner(emission[0], tokenizer(tokens))
                frame_seconds = (len(waveform16) / TARGET_SR) / emission.shape[1]
                aligned = [
                    {
                        "text": token,
                        "start": chars[0].start * frame_seconds,
                        "end": chars[-1].end * frame_seconds,
                        "score": float(np.mean([float(char.score) for char in chars])),
                    }
                    for token, chars in zip(tokens, spans)
                ]
            except Exception as exc:
                rows.append({**base, "status": "ALIGNMENT_ERROR", "error": str(exc)})
                continue
            if not aligned:
                rows.append({**base, "status": "ALIGNMENT_EMPTY"})
                continue
            local_start, local_end, acoustic = refine_edges(
                clip, sr, aligned[0]["start"], aligned[-1]["end"],
            )
            proposed_start = context_start + local_start
            proposed_end = context_start + local_end
            next_card_start = (
                float(lines[line_index + 1]["start"])
                if line_index + 1 < len(lines) else duration
            )
            scores = [word["score"] for word in aligned]
            mean_score = float(np.mean(scores))
            min_score = float(np.min(scores))
            ordering_ok = proposed_start < proposed_end
            confidence_ok = mean_score >= 0.30 and min_score >= 0.05
            start_boundary_ok = bool(
                acoustic["start_valley_found"] or proposed_start <= 0.05
            )
            end_boundary_ok = bool(
                acoustic["end_valley_found"] or proposed_end >= duration - 0.05
            )
            crosses_next_card = proposed_end > next_card_start + 0.02
            proposal_ok = bool(
                confidence_ok
                and ordering_ok
                and start_boundary_ok
                and end_boundary_ok
                and not crosses_next_card
            )
            rows.append({
                **base,
                "status": "PROPOSED" if proposal_ok else "REVIEW",
                "context_start": context_start, "context_end": context_end,
                "aligned_words": aligned,
                "alignment_mean_score": mean_score,
                "alignment_min_score": min_score,
                "proposed_start": round(proposed_start, 3),
                "proposed_end": round(proposed_end, 3),
                "start_delta": round(proposed_start - float(line["start"]), 3),
                "end_delta": round(proposed_end - float(line["end"]), 3),
                "next_card_start": next_card_start,
                "crosses_next_card": crosses_next_card,
                "requires_delivery_grouping": crosses_next_card or not end_boundary_ok,
                "acoustic_refinement": acoustic,
            })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "MMS constrained official-EN forced alignment plus stable-quiet edge refinement",
        "device": DEVICE,
        "maps_modified": False,
        "rows": rows,
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT), "counts": payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
