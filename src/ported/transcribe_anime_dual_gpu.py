#!/usr/bin/env python3
"""Create two independent GPU ASR views of every mapped P3R anime stem.

These transcripts are evidence only.  They never define a delivery boundary
by themselves; the consensus stage combines them with official text, MMS
forced alignment and AC-29 acoustic valleys.
"""
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
MAPS = ROOT / "maps"
OUT = ROOT / "ANIME_DUAL_ASR_EVIDENCE"


def transcribe(model: WhisperModel, audio: np.ndarray, *, vad: bool, beam: int) -> list[dict]:
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=beam,
        vad_filter=vad,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "avg_logprob": float(segment.avg_logprob),
            "no_speech_prob": float(segment.no_speech_prob),
            "words": [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word.strip(),
                    "probability": float(word.probability),
                }
                for word in (segment.words or [])
                if word.start is not None and word.end is not None
            ],
        }
        for segment in segments
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    maps = sorted(MAPS.glob("*_map.json"))
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    for index, map_path in enumerate(maps, 1):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        target = OUT / f"{mapping['scene']}_dual_asr.json"
        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
        if source.ndim > 1:
            source = source.mean(axis=1)
        duration = len(source) / sr
        if sr != 16_000:
            source = librosa.resample(source, orig_sr=sr, target_sr=16_000)
        audio = np.asarray(source, dtype=np.float32)
        payload = {
            "schema": "p3r_anime_dual_asr_evidence_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scene": mapping["scene"],
            "audio": mapping["source_stem"],
            "duration_seconds": duration,
            "model": "faster-whisper large-v3-turbo CUDA FP16",
            "authority": "corroboration_only_not_boundary_authority",
            "passes": {
                "vad_beam5": transcribe(model, audio, vad=True, beam=5),
                "full_beam10": transcribe(model, audio, vad=False, beam=10),
            },
        }
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        counts = {
            name: sum(len(segment["words"]) for segment in segments)
            for name, segments in payload["passes"].items()
        }
        print(f"[{index}/{len(maps)}] {mapping['scene']} {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
