"""Align each P3R anime USM to a subtitle-burned reference compilation.

This intentionally does *not* use ASR or BMD order as timing evidence.  It
matches the original mix to the reference video's audio through a robust RMS
envelope correlation, then writes the per-scene video timebase needed by the
subtitle-card detector.  Input and derived artifacts live on E: so the source
project remains compact.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
VIDEO_AUDIO = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\reference_audio_8k.wav")
INVENTORY = ROOT / "anime_all_20260725" / "anime_dialogue_inventory.json"
OUT = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\scene_video_alignment.json")

RATE = 8000
HOP = 800  # 10 Hz: enough to locate a scene before frame-level card detection
ENVELOPE_RATE = RATE / HOP


def mono(path: Path) -> np.ndarray:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != RATE:
        # All currently supplied sources are 48 kHz.  Decimation after a
        # low-cost anti-aliasing average keeps this script dependency-light.
        ratio = sr // RATE
        if sr % RATE:
            raise ValueError(f"unsupported sample rate {sr} for {path}")
        usable = (len(data) // ratio) * ratio
        data = data[:usable].reshape(-1, ratio, data.shape[1]).mean(axis=1)
    return data.mean(axis=1)


def envelope(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP * HOP
    x = x[:n].reshape(-1, HOP)
    e = np.sqrt(np.mean(np.square(x), axis=1) + 1e-10)
    # Log compression makes matching resilient to YouTube's level changes.
    e = np.log(e + 1e-5)
    return (e - e.mean()) / (e.std() + 1e-8)


def top_match(reference: np.ndarray, query: np.ndarray) -> tuple[int, float, float]:
    # Locally normalized cross-correlation.  The compilation's level is not
    # constant, so dividing only by query length incorrectly favours the loud
    # first minute.  Normalization is recalculated for every candidate window.
    n = len(query)
    raw = fftconvolve(reference, query[::-1], mode="valid")
    local_sum = np.convolve(reference, np.ones(n, dtype=np.float32), mode="valid")
    local_sq = np.convolve(np.square(reference), np.ones(n, dtype=np.float32), mode="valid")
    local_var = np.maximum(local_sq - np.square(local_sum) / n, 1e-8)
    corr = raw / np.sqrt(local_var * np.sum(np.square(query)))
    order = np.argpartition(corr, -2)[-2:]
    order = order[np.argsort(corr[order])[::-1]]
    return int(order[0]), float(corr[order[0]]), float(corr[order[1]])


def main() -> None:
    global VIDEO_AUDIO, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=VIDEO_AUDIO)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    VIDEO_AUDIO, OUT = args.audio, args.out
    ref = envelope(mono(VIDEO_AUDIO))
    items = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        path = Path(item["source_full_6ch"])
        q = envelope(mono(path))
        pos, score, runner_up = top_match(ref, q)
        rows.append({
            "scene": item["scene"],
            "reference_start_s": round(pos / ENVELOPE_RATE, 3),
            "source_duration_s": round(len(q) / ENVELOPE_RATE, 3),
            "score": round(score, 4),
            "runner_up_score": round(runner_up, 4),
            "margin": round(score - runner_up, 4),
            "status": "AUTO_CANDIDATE",
        })
    rows.sort(key=lambda r: r["reference_start_s"])
    OUT.write_text(json.dumps({"method": "audio-envelope alignment only; no ASR", "rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['reference_start_s']:8.3f}s  {row['scene']:12} score={row['score']:.3f} margin={row['margin']:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
