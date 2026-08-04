#!/usr/bin/env python3
"""Final conservative rescue using coarse subtitle-card neighbourhoods.

This is used only after the normal scene-block and ASR-seeded MMS passes.  It
searches all three free ASR streams near a card, requires two independent
timestamp-agreeing matches, and then runs MMS again in that discovered span.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from build_anime_timing_consensus import exact_sequence_alignment, flatten_asr
from rebuild_anime_delivery_windows_mms import refine_edges, words


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
CONSENSUS = ROOT / "ANIME_TIMING_CONSENSUS.json"
FIRST_RESCUE = ROOT / "ANIME_ASR_SEEDED_MMS_RESCUE.json"
OUT = ROOT / "ANIME_CARD_SEEDED_MMS_RESCUE.json"
TARGET_SR = 16_000


def local_match(tokens: list[str], stream: list[dict], lo: float, hi: float) -> dict:
    local = [item for item in stream if item["end"] >= lo and item["start"] <= hi]
    mapping = exact_sequence_alignment(tokens, [item["token"] for item in local])
    pairs = sorted([(index, local[target]) for index, target in mapping.items()])
    return {
        "coverage": len(pairs) / max(1, len(tokens)),
        "pairs": [{"official_index": index, **item} for index, item in pairs],
    }


def main() -> int:
    consensus = json.loads(CONSENSUS.read_text(encoding="utf-8"))
    first = json.loads(FIRST_RESCUE.read_text(encoding="utf-8"))
    already = {row["line_id"] for row in first["rows"] if row["status"] == "RESCUED"}
    maps = {
        path.stem.removesuffix("_map"): json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "maps").glob("*_map.json")
    }
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).cuda().eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    source_cache: dict[str, tuple[np.ndarray, int]] = {}
    asr_cache: dict[str, dict[str, list[dict]]] = {}
    rows: list[dict] = []
    for row in consensus["rows"]:
        if row["status"] != "TIMING_EVIDENCE_BLOCK" or row["line_id"] in already:
            continue
        scene = row["scene"]
        tokens = words(row.get("text_en", ""))
        if scene not in asr_cache:
            dual = json.loads((ROOT / "ANIME_DUAL_ASR_EVIDENCE" / f"{scene}_dual_asr.json").read_text(encoding="utf-8"))
            large = json.loads((ROOT / "ANIME_LARGE_V3_ASR_EVIDENCE" / f"{scene}_large_v3.json").read_text(encoding="utf-8"))
            asr_cache[scene] = {
                "vad_beam5": flatten_asr(dual["passes"]["vad_beam5"]),
                "full_beam10": flatten_asr(dual["passes"]["full_beam10"]),
                "large_v3_full": flatten_asr(large["segments"]),
            }
        search_lo = max(0.0, float(row["card_start"]) - 2.5)
        search_hi = float(row["card_end"]) + 4.0
        matches = {
            name: local_match(tokens, stream, search_lo, search_hi)
            for name, stream in asr_cache[scene].items()
        }
        required = 1.0 if len(tokens) <= 3 else (0.60 if len(tokens) <= 7 else 0.50)
        qualified = [name for name, match in matches.items() if match["coverage"] >= required]
        best = None
        for index, first_name in enumerate(qualified):
            for second_name in qualified[index + 1:]:
                first_pairs = {p["official_index"]: p for p in matches[first_name]["pairs"]}
                second_pairs = {p["official_index"]: p for p in matches[second_name]["pairs"]}
                shared = sorted(set(first_pairs) & set(second_pairs))
                if not shared:
                    continue
                ds = abs(first_pairs[shared[0]]["start"] - second_pairs[shared[0]]["start"])
                de = abs(first_pairs[shared[-1]]["end"] - second_pairs[shared[-1]]["end"])
                score = max(ds, de)
                if score <= 0.75 and (best is None or score < best[0]):
                    best = (score, first_name, second_name, shared)
        if best is None:
            rows.append({"scene": scene, "line_id": row["line_id"], "status": "NOT_ELIGIBLE", "matches": matches})
            continue
        _, first_name, second_name, _ = best
        if (
            len(tokens) <= 3
            and "vad_beam5" not in {first_name, second_name}
            and float(row.get("mms_mean_score") or 0.0) < 0.75
        ):
            rows.append({
                "scene": scene, "line_id": row["line_id"],
                "status": "NOT_ELIGIBLE", "matches": matches,
                "reason": "short_nonvad_hallucination_guard",
            })
            continue
        pairs = matches[first_name]["pairs"] + matches[second_name]["pairs"]
        anchor_start = min(float(pair["start"]) for pair in pairs)
        anchor_end = max(float(pair["end"]) for pair in pairs)
        context_start = max(0.0, anchor_start - 1.5)
        mapping = maps[scene]
        if scene not in source_cache:
            source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
            if source.ndim > 1:
                source = source.mean(axis=1)
            source_cache[scene] = (np.asarray(source, dtype=np.float32), sr)
        source, sr = source_cache[scene]
        context_end = min(len(source) / sr, anchor_end + 1.5)
        clip = source[round(context_start * sr):round(context_end * sr)]
        waveform16 = torchaudio.functional.resample(torch.from_numpy(clip), sr, TARGET_SR)
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
                clip, sr, aligned[0]["start"] - context_start, aligned[-1]["end"] - context_start
            )
            deltas = []
            for pair in pairs:
                token_index = int(pair["official_index"])
                if token_index < len(aligned):
                    deltas += [
                        abs(float(pair["start"]) - aligned[token_index]["start"]),
                        abs(float(pair["end"]) - aligned[token_index]["end"]),
                    ]
            scores = [item["score"] for item in aligned]
            median_delta = float(np.median(deltas)) if deltas else None
            passed = bool(
                np.mean(scores) >= 0.30 and median_delta is not None and median_delta <= 0.30
                and acoustic["start_valley_found"] and acoustic["end_valley_found"]
            )
            rows.append({
                "scene": scene, "line_id": row["line_id"],
                "status": "RESCUED" if passed else "RESCUE_BLOCK",
                "text_en": row["text_en"], "matches": matches,
                "anchor_passes": [first_name, second_name],
                "asr_anchor_start": anchor_start, "asr_anchor_end": anchor_end,
                "proposed_start": round(context_start + local_start, 3),
                "proposed_end": round(context_start + local_end, 3),
                "alignment_mean_score": float(np.mean(scores)),
                "alignment_min_score": float(np.min(scores)),
                "mms_asr_median_delta": median_delta,
                "aligned_words_global": aligned, "acoustic": acoustic,
            })
        except Exception as exc:
            rows.append({"scene": scene, "line_id": row["line_id"], "status": "ERROR", "error": str(exc)})
    payload = {
        "schema": "p3r_anime_card_seeded_mms_rescue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "maps_modified": False,
        "rows": rows,
        "counts": {status: sum(x["status"] == status for x in rows) for status in sorted({x["status"] for x in rows})},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT), "counts": payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
