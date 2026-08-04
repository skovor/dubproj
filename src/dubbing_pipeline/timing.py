"""Duration/onset policy; no active consonant is cut to fit a window."""
from __future__ import annotations

import subprocess
import math
from pathlib import Path


def _np():
    import numpy as np
    return np


def speech_end(audio, sample_rate: int, floor_db: float = -45.0) -> float:
    np = _np()
    value = np.asarray(audio)
    peak = float(np.max(np.abs(value))) if len(value) else 0.0
    if peak <= 0:
        return 0.0
    indexes = np.where(np.abs(value) > peak * 10 ** (floor_db / 20))[0]
    return float(indexes[-1]) / sample_rate if len(indexes) else 0.0


def atempo_chain(ratio: float) -> str:
    stages, value = [], float(ratio)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("atempo ratio must be finite and positive")
    while value > 2.0:
        stages.append(2.0); value /= 2.0
    while value < .5:
        stages.append(.5); value /= .5
    stages.append(value)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def correct_length(audio, sample_rate: int, reference_end: float, ffmpeg: str | Path | None, tmpdir: str | Path, under_tol: float = .35, over_tol: float = .35, max_ratio_deviation: float | None = .20):
    """Use native duration first; use atempo only when outside +/-0.35 s."""
    np = _np()
    end = speech_end(audio, sample_rate)
    diff = end - reference_end
    info = {"reference_end": round(reference_end, 3), "voice_end": round(end, 3), "diff": round(diff, 3), "method": "duration"}
    if end <= 0 or -under_tol <= diff <= over_tol or ffmpeg is None:
        return np.asarray(audio), info
    target = reference_end + over_tol if diff > over_tol else reference_end - under_tol
    if target <= .05:
        info["method"] = "atempo_refused_target"
        return np.asarray(audio), info
    ratio = end / target
    if max_ratio_deviation is not None and abs(ratio - 1) > max_ratio_deviation:
        info.update({"method": "atempo_refused_ratio", "ratio": round(ratio, 4)})
        return np.asarray(audio), info
    tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    from .audio import write, read
    source, output = tmpdir / "timing_in.wav", tmpdir / "timing_out.wav"
    write(source, audio, sample_rate)
    command = [str(ffmpeg), "-y", "-i", str(source), "-af", atempo_chain(ratio), "-loglevel", "error", str(output)]
    completed = subprocess.run(command, capture_output=True, timeout=120, text=True, encoding="utf-8", errors="replace")
    if completed.returncode or not output.is_file():
        info.update({"method": "atempo_failed", "returncode": completed.returncode})
        return np.asarray(audio), info
    corrected, rate = read(output)
    if corrected.ndim > 1:
        corrected = corrected.mean(axis=1)
    info.update({"method": "atempo", "ratio": round(ratio, 4), "voice_end_corrected": round(speech_end(corrected, rate), 3)})
    return np.asarray(corrected), info


def trim_lead_silence(audio, sample_rate: int, floor_db: float = -45.0):
    np = _np()
    value = np.asarray(audio)
    peak = float(np.max(np.abs(value))) if len(value) else 0.0
    if peak <= 0:
        return value
    indexes = np.where(np.abs(value) > peak * 10 ** (floor_db / 20))[0]
    if not len(indexes):
        return value
    start = int(indexes[0])
    result = value[start:].copy()
    fade = min(round(.003 * sample_rate), len(result))
    if fade:
        result[:fade] *= .5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    return result
