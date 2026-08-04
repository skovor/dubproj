"""Sample-exact audio primitives shared by line and FMV routes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioSpec:
    frames: int
    sample_rate: int
    channels: int


def _np():
    import numpy as np
    return np


def _sf():
    import soundfile as sf
    return sf


def read(path: str | Path, always_2d: bool = False):
    sf = _sf()
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=always_2d)
    return _np().asarray(audio, dtype="float32"), int(sample_rate)


def write(path: str | Path, audio, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _sf().write(str(target), _np().asarray(audio, dtype="float32"), int(sample_rate), subtype="PCM_16")


def spec(path: str | Path) -> AudioSpec:
    info = _sf().info(str(path))
    return AudioSpec(int(info.frames), int(info.samplerate), int(info.channels))


def mono(audio):
    np = _np()
    value = np.asarray(audio)
    return value.mean(axis=1) if value.ndim == 2 else value


def resample_exact(audio, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return _np().asarray(audio, dtype="float32")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    np = _np(); value = np.asarray(audio, dtype="float32")
    try:
        import librosa
        # soundfile's canonical layout is (frames, channels); never let a
        # multichannel array be resampled along its channel axis.
        result = librosa.resample(value, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_vhq", axis=0)
    except (ImportError, ModuleNotFoundError):
        # A deterministic linear fallback keeps contract/audit operations
        # usable without the optional DSP stack. Production can select soxr.
        old_frames = value.shape[0]
        new_frames = max(1, round(old_frames * target_rate / source_rate))
        old_x = np.linspace(0.0, 1.0, old_frames, endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_frames, endpoint=False)
        if value.ndim == 1:
            result = np.interp(new_x, old_x, value)
        else:
            result = np.column_stack([np.interp(new_x, old_x, value[:, channel]) for channel in range(value.shape[1])])
    return np.asarray(result, dtype="float32")


def active_span(audio, sample_rate: int, top_db: float = 35.0) -> tuple[int, int]:
    import librosa
    intervals = librosa.effects.split(mono(audio), top_db=top_db, frame_length=1024, hop_length=128)
    if not len(intervals):
        return 0, 0
    return int(intervals[0][0]), int(intervals[-1][1])


def align_onset_exact(source, generated, sample_rate: int, output_frames: int, lead_guard_seconds: float = .010, tail_guard_seconds: float = .050, fade_seconds: float = .010):
    np = _np()
    src_start, _ = active_span(source, sample_rate)
    gen_start, gen_end = active_span(generated, sample_rate)
    if gen_end <= gen_start:
        raise ValueError("generated candidate has no active speech")
    lead, tail = round(lead_guard_seconds * sample_rate), round(tail_guard_seconds * sample_rate)
    body = np.array(generated[max(0, gen_start - lead):min(len(generated), gen_end + tail)], dtype="float32", copy=True)
    destination = max(0, src_start - lead)
    if destination + len(body) > output_frames:
        raise ValueError(f"active body overflow: need {destination + len(body)}, have {output_frames}")
    fade = min(round(fade_seconds * sample_rate), len(body) // 2)
    if fade:
        body[:fade] *= np.linspace(0, 1, fade, dtype="float32")
        body[-fade:] *= np.linspace(1, 0, fade, dtype="float32")
    result = np.zeros(output_frames, dtype="float32")
    result[destination:destination + len(body)] = body
    return result


def constant_gain(audio, gain_db: float, peak_limit: float = .98):
    np = _np()
    value = np.asarray(audio, dtype="float32")
    factor = float(10 ** (gain_db / 20))
    peak = float(np.max(np.abs(value))) if len(value) else 0.0
    if peak and peak * factor > peak_limit:
        factor = peak_limit / peak
    applied = 20 * np.log10(max(factor, 1e-12))
    return value * factor, float(applied)


def assert_exact(path: str | Path, expected: AudioSpec) -> None:
    actual = spec(path)
    if actual != expected:
        raise ValueError(f"audio contract mismatch: expected={expected} actual={actual}")


def clipping(audio, limit: float = .999):
    np = _np()
    value = np.asarray(audio)
    return bool(np.any(np.abs(value) >= limit))


def peak_dbfs(audio) -> float:
    import math
    np = _np()
    peak = float(np.max(np.abs(np.asarray(audio)))) if len(audio) else 0.0
    return -120.0 if peak <= 1e-12 else 20.0 * math.log10(peak)
