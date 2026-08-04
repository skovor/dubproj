#!/usr/bin/env python3
"""Third independent ASR judge for P3R anime timing validation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
OUT = ROOT / "ANIME_LARGE_V3_ASR_EVIDENCE"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    maps = sorted((ROOT / "maps").glob("*_map.json"))
    for index, map_path in enumerate(maps, 1):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
        if source.ndim > 1:
            source = source.mean(axis=1)
        if sr != 16_000:
            source = librosa.resample(source, orig_sr=sr, target_sr=16_000)
        segments, _ = model.transcribe(
            np.asarray(source, dtype=np.float32),
            language="en",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        rows = []
        for segment in segments:
            rows.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
                "words": [
                    {
                        "start": float(word.start), "end": float(word.end),
                        "word": word.word.strip(), "probability": float(word.probability),
                    }
                    for word in (segment.words or [])
                    if word.start is not None and word.end is not None
                ],
            })
        payload = {
            "schema": "p3r_anime_large_v3_asr_evidence_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scene": mapping["scene"],
            "audio": mapping["source_stem"],
            "model": "faster-whisper large-v3 CUDA FP16",
            "authority": "corroboration_only_not_boundary_authority",
            "segments": rows,
        }
        target = OUT / f"{mapping['scene']}_large_v3.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        print(f"[{index}/{len(maps)}] {mapping['scene']} words={sum(len(x['words']) for x in rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
