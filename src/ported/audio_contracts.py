"""Sample-exact audio contracts shared by movie and in-engine delivery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioSpec:
    frames: int
    sample_rate: int
    channels: int


def read(path: str | Path, always_2d: bool = False) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=always_2d)
    return np.asarray(audio, dtype=np.float32), int(sr)


def spec(path: str | Path) -> AudioSpec:
    info = sf.info(path)
    return AudioSpec(int(info.frames), int(info.samplerate), int(info.channels))


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def resample_exact(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    result = librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=source_sr,
        target_sr=target_sr,
        res_type="soxr_vhq",
    )
    return np.asarray(result, dtype=np.float32)


def active_span(audio: np.ndarray, sr: int, top_db: float = 35.0) -> tuple[int, int]:
    y = mono(audio)
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=128)
    if not len(intervals):
        return 0, 0
    return int(intervals[0][0]), int(intervals[-1][1])


def align_onset_exact(
    source: np.ndarray,
    generated: np.ndarray,
    sr: int,
    output_frames: int,
    lead_guard_seconds: float = 0.010,
    tail_guard_seconds: float = 0.050,
    fade_seconds: float = 0.010,
) -> np.ndarray:
    """Move active speech to the source onset without stretching or cutting words.

    Raises when the complete generated active body cannot fit. Returned length is
    always exactly ``output_frames``.
    """
    src_start, _ = active_span(source, sr)
    gen_start, gen_end = active_span(generated, sr)
    if gen_end <= gen_start:
        raise ValueError("generated candidate has no active speech")
    lead = round(lead_guard_seconds * sr)
    tail = round(tail_guard_seconds * sr)
    body_start = max(0, gen_start - lead)
    body_end = min(len(generated), gen_end + tail)
    body = np.array(generated[body_start:body_end], dtype=np.float32, copy=True)
    destination = max(0, src_start - lead)
    if destination + len(body) > output_frames:
        raise ValueError(
            f"active body overflow: need {destination + len(body)} frames, have {output_frames}"
        )
    fade = min(round(fade_seconds * sr), len(body) // 2)
    if fade:
        body[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        body[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    result = np.zeros(output_frames, dtype=np.float32)
    result[destination:destination + len(body)] = body
    return result


def constant_gain(audio: np.ndarray, gain_db: float, peak_limit: float = 0.98) -> tuple[np.ndarray, float]:
    factor = float(10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak and peak * factor > peak_limit:
        factor = peak_limit / peak
    applied_db = 20.0 * np.log10(max(factor, 1e-12))
    return np.asarray(audio * factor, dtype=np.float32), float(applied_db)


def assert_exact(path: str | Path, expected: AudioSpec) -> None:
    actual = spec(path)
    if actual != expected:
        raise AssertionError(f"audio contract failed: expected {expected}, got {actual}: {path}")


def write_exact(path: str | Path, audio: np.ndarray, sr: int, expected_frames: int) -> None:
    if len(audio) != expected_frames:
        raise ValueError(f"refusing non-exact write: {len(audio)} != {expected_frames}")
    sf.write(path, audio, sr, subtype="PCM_16")
