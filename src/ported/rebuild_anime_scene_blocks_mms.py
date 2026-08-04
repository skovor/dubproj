#!/usr/bin/env python3
"""Align official subtitle sequences against broad dry-stem scene blocks.

This second pass is independent of each card's individual end. It preserves
subtitle order, assigns the resulting word spans back to their original cards,
and uses AC-29 valley refinement for the acoustic boundaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from rebuild_anime_delivery_windows_mms import refine_edges, words


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
OUT = ROOT / "ANIME_SCENE_BLOCK_ALIGNMENT_PROPOSALS.json"
TARGET_SR = 16000
BLOCK_BREAK_SECONDS = 6.0
CONTEXT_BEFORE = 1.5
CONTEXT_AFTER = 3.0


def make_blocks(lines: list[dict]) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    current: list[dict] = []
    for line in lines:
        if current and float(line["start"]) - float(current[-1]["end"]) > BLOCK_BREAK_SECONDS:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def main() -> int:
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).cuda().eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    rows: list[dict] = []
    for map_path in sorted((ROOT / "maps").glob("*_map.json")):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
        if source.ndim > 1:
            source = source.mean(axis=1)
        duration = len(source) / sr
        ordered = sorted(mapping["lines"], key=lambda row: float(row["start"]))
        next_start_by_id = {
            line["id"]: (float(ordered[index + 1]["start"]) if index + 1 < len(ordered) else duration)
            for index, line in enumerate(ordered)
        }
        for block_index, block in enumerate(make_blocks(ordered)):
            lexical = [(line, words(line.get("source_text", ""))) for line in block]
            flat = [token for _, tokens in lexical for token in tokens]
            if not flat:
                for line, _ in lexical:
                    rows.append({"scene": mapping["scene"], "line_id": line["id"], "status": "NO_LEXICAL_TOKENS"})
                continue
            context_start = max(0.0, float(block[0]["start"]) - CONTEXT_BEFORE)
            context_end = min(duration, float(block[-1]["end"]) + CONTEXT_AFTER)
            clip = source[round(context_start * sr):round(context_end * sr)]
            waveform = torch.from_numpy(np.asarray(clip, dtype=np.float32))
            waveform16 = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            try:
                with torch.inference_mode():
                    emission, _ = model(waveform16.unsqueeze(0).cuda())
                spans = aligner(emission[0], tokenizer(flat))
                frame_seconds = (len(waveform16) / TARGET_SR) / emission.shape[1]
                aligned = [
                    {
                        "text": token,
                        "start": chars[0].start * frame_seconds,
                        "end": chars[-1].end * frame_seconds,
                        "score": float(np.mean([float(char.score) for char in chars])),
                    }
                    for token, chars in zip(flat, spans)
                ]
            except Exception as exc:
                for line, _ in lexical:
                    rows.append({
                        "scene": mapping["scene"], "line_id": line["id"],
                        "status": "ALIGNMENT_ERROR", "error": str(exc),
                    })
                continue
            cursor = 0
            for line, tokens in lexical:
                base = {
                    "scene": mapping["scene"], "line_id": line["id"],
                    "speaker": line.get("speaker"), "text_en": line.get("source_text", ""),
                    "card_start": float(line["start"]), "card_end": float(line["end"]),
                    "block_index": block_index,
                    "block_context_start": context_start,
                    "block_context_end": context_end,
                }
                if not tokens:
                    rows.append({**base, "status": "NO_LEXICAL_TOKENS"})
                    continue
                line_words = aligned[cursor:cursor + len(tokens)]
                cursor += len(tokens)
                local_start, local_end, acoustic = refine_edges(
                    clip, sr, line_words[0]["start"], line_words[-1]["end"],
                )
                proposed_start = context_start + local_start
                proposed_end = context_start + local_end
                scores = [word["score"] for word in line_words]
                mean_score = float(np.mean(scores))
                min_score = float(np.min(scores))
                next_card_start = next_start_by_id[line["id"]]
                crosses = proposed_end > next_card_start + 0.02
                start_ok = acoustic["start_valley_found"] or proposed_start <= 0.05
                end_ok = acoustic["end_valley_found"] or proposed_end >= duration - 0.05
                confidence_ok = mean_score >= 0.30 and min_score >= 0.05
                proposal_ok = confidence_ok and start_ok and end_ok and proposed_start < proposed_end and not crosses
                rows.append({
                    **base,
                    "status": "PROPOSED" if proposal_ok else "REVIEW",
                    "aligned_words": line_words,
                    "aligned_words_global": [
                        {
                            **word,
                            "start": context_start + float(word["start"]),
                            "end": context_start + float(word["end"]),
                        }
                        for word in line_words
                    ],
                    "alignment_mean_score": mean_score,
                    "alignment_min_score": min_score,
                    "proposed_start": round(proposed_start, 3),
                    "proposed_end": round(proposed_end, 3),
                    "start_delta": round(proposed_start - float(line["start"]), 3),
                    "end_delta": round(proposed_end - float(line["end"]), 3),
                    "next_card_start": next_card_start,
                    "crosses_next_card": crosses,
                    "requires_delivery_grouping": crosses or not end_ok,
                    "acoustic_refinement": acoustic,
                })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "scene-block MMS official-EN alignment plus AC-29 first sustained valleys",
        "maps_modified": False,
        "rows": rows,
        "counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT), "counts": payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
