#!/usr/bin/env python3
"""Rescue blocked anime lines using dual-ASR anchors and fresh MMS alignment.

Only lines whose two free ASR passes independently locate enough official
tokens are eligible.  Their consensus span seeds a new local phonetic search;
ASR itself never becomes the final acoustic boundary authority.
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
CONSENSUS = ROOT / "ANIME_TIMING_CONSENSUS.json"
OUT = ROOT / "ANIME_ASR_SEEDED_MMS_RESCUE.json"
TARGET_SR = 16_000


def main() -> int:
    consensus = json.loads(CONSENSUS.read_text(encoding="utf-8"))
    blocked = [row for row in consensus["rows"] if row["status"] == "TIMING_EVIDENCE_BLOCK"]
    maps = {
        path.stem.removesuffix("_map"): json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "maps").glob("*_map.json")
    }
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).cuda().eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    source_cache: dict[str, tuple[np.ndarray, int]] = {}
    rows: list[dict] = []
    for row in blocked:
        tokens = words(row.get("text_en", ""))
        selected_pass_names = row.get("anchor_agreement_passes", [])
        selected_passes = [row["passes"][name] for name in selected_pass_names]
        common = sorted(set(row.get("common_exact_indices", [])))
        required = float(row["required_coverage"])
        eligible = bool(
            tokens
            and common
            and len(selected_passes) == 2
            and all(source_pass["coverage"] >= required for source_pass in selected_passes)
            and row.get("anchor_spread_start") is not None
            and row.get("anchor_spread_end") is not None
            and float(row["anchor_spread_start"]) <= 0.75
            and float(row["anchor_spread_end"]) <= 0.75
            and (
                len(tokens) > 3
                or "vad_beam5" in selected_pass_names
                or float(row.get("mms_mean_score") or 0.0) >= 0.75
            )
        )
        if not eligible:
            rows.append({"scene": row["scene"], "line_id": row["line_id"], "status": "NOT_ELIGIBLE"})
            continue
        pairs = [pair for source_pass in selected_passes for pair in source_pass["pairs"]]
        anchor_start = min(float(pair["start"]) for pair in pairs)
        anchor_end = max(float(pair["end"]) for pair in pairs)
        context_start = max(0.0, anchor_start - 1.5)
        mapping = maps[row["scene"]]
        if row["scene"] not in source_cache:
            source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
            if source.ndim > 1:
                source = source.mean(axis=1)
            source_cache[row["scene"]] = (np.asarray(source, dtype=np.float32), sr)
        source, sr = source_cache[row["scene"]]
        context_end = min(len(source) / sr, anchor_end + 1.5)
        clip = source[round(context_start * sr):round(context_end * sr)]
        waveform = torch.from_numpy(clip)
        waveform16 = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        try:
            with torch.inference_mode():
                emission, _ = model(waveform16.unsqueeze(0).cuda())
            spans = aligner(emission[0], tokenizer(tokens))
            frame_seconds = (len(waveform16) / TARGET_SR) / emission.shape[1]
            aligned = [
                {
                    "text": token,
                    "start": context_start + chars[0].start * frame_seconds,
                    "end": context_start + chars[-1].end * frame_seconds,
                    "score": float(np.mean([float(char.score) for char in chars])),
                }
                for token, chars in zip(tokens, spans)
            ]
            local_start, local_end, acoustic = refine_edges(
                clip,
                sr,
                aligned[0]["start"] - context_start,
                aligned[-1]["end"] - context_start,
            )
            proposed_start = context_start + local_start
            proposed_end = context_start + local_end
            scores = [word["score"] for word in aligned]
            anchor_deltas = []
            for pair in pairs:
                index = int(pair["official_index"])
                if index < len(aligned):
                    anchor_deltas.extend([
                        abs(float(pair["start"]) - aligned[index]["start"]),
                        abs(float(pair["end"]) - aligned[index]["end"]),
                    ])
            median_delta = float(np.median(anchor_deltas)) if anchor_deltas else None
            passed = bool(
                np.mean(scores) >= 0.30
                and median_delta is not None
                and median_delta <= 0.30
                and acoustic["start_valley_found"]
                and acoustic["end_valley_found"]
                and proposed_start < proposed_end
            )
            rows.append({
                "scene": row["scene"],
                "line_id": row["line_id"],
                "status": "RESCUED" if passed else "RESCUE_BLOCK",
                "text_en": row["text_en"],
                "asr_anchor_start": anchor_start,
                "asr_anchor_end": anchor_end,
                "context_start": context_start,
                "context_end": context_end,
                "aligned_words_global": aligned,
                "alignment_mean_score": float(np.mean(scores)),
                "alignment_min_score": float(np.min(scores)),
                "mms_asr_median_delta": median_delta,
                "proposed_start": round(proposed_start, 3),
                "proposed_end": round(proposed_end, 3),
                "acoustic": acoustic,
            })
        except Exception as exc:
            rows.append({"scene": row["scene"], "line_id": row["line_id"], "status": "ERROR", "error": str(exc)})
    payload = {
        "schema": "p3r_anime_asr_seeded_mms_rescue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
