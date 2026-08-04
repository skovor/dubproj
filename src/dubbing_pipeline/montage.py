"""Sample-surgical FMV montage and Empalme B preservation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio import active_span, resample_exact
from .contracts import DeliveryWindow, ContractError
from .hashing import sha256_bytes


class MountError(ValueError):
    pass


@dataclass(frozen=True)
class MountMetrics:
    body_start_sample: int
    body_end_sample: int
    preserved_hash_before: str
    preserved_hash_after: str
    untouched_channel_hashes_before: tuple[str, ...]
    untouched_channel_hashes_after: tuple[str, ...]
    source_resume_sample: int | None
    empalme_b: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _as_2d(value: Any):
    import numpy as np
    data = np.asarray(value, dtype="float32")
    return data[:, None] if data.ndim == 1 else data


def _interval_bytes(audio, intervals: tuple[tuple[int, int], ...], channel: int) -> bytes:
    import numpy as np
    if not intervals:
        return b""
    return b"".join(np.asarray(audio[start:end, channel], dtype="float32").tobytes() for start, end in intervals)


def mount_surgical(stem: Any, generated: Any, generated_rate: int, window: DeliveryWindow, stem_rate: int, *, empalme_b: bool = False) -> tuple[Any, MountMetrics]:
    """Replace only the subtitle-owned active mask, preserving the original bed.

    The original array is copied once per scene.  All non-dialogue channels and
    every sample outside the declared speech mask remain byte-identical.
    """
    import numpy as np
    source = _as_2d(stem)
    body = _as_2d(generated)
    if source.ndim != 2 or body.ndim != 2:
        raise MountError("stem and generated audio must be one- or two-dimensional")
    if not 0 <= window.dialogue_channel < source.shape[1]:
        raise MountError("dialogue channel outside stem")
    if window.end_sample > len(source):
        raise MountError("delivery window outside stem")
    if generated_rate <= 0 or stem_rate <= 0:
        raise MountError("sample rates must be positive")
    if generated_rate != stem_rate:
        body = resample_exact(body, generated_rate, stem_rate)
    body_mono = body[:, 0] if body.shape[1] == 1 else body.mean(axis=1)
    try:
        active_start, active_end = active_span(body_mono, stem_rate)
    except Exception:
        import numpy as np
        indexes = np.flatnonzero(np.abs(body_mono) > max(1e-7, float(np.max(np.abs(body_mono))) * 10 ** (-45 / 20)))
        active_start, active_end = (int(indexes[0]), int(indexes[-1]) + 1) if len(indexes) else (0, 0)
    if active_end <= active_start:
        raise MountError("generated candidate has no active speech")
    body = body[active_start:active_end]
    destination = window.speech_start_sample
    limit = window.speech_end_sample
    if destination + len(body) > limit:
        raise MountError(f"active TTS body would be cut: {destination + len(body)} > {limit}")
    channel = window.dialogue_channel
    preserved_before = _interval_bytes(source, window.preserved_source_intervals, channel)
    untouched_before = tuple(sha256_bytes(source[:, idx].tobytes()) for idx in range(source.shape[1]) if idx != channel)
    result = np.array(source, dtype="float32", copy=True)
    # Clear only the subtitle-owned active region; the original head/effort,
    # room tone, and resume tail are not erased.
    result[destination:limit, channel] = 0.0
    result[destination:destination + len(body), channel] = body[:, 0] if body.shape[1] == 1 else body[:, min(channel, body.shape[1] - 1)]
    if empalme_b and window.source_resume_sample is not None:
        resume = max(window.speech_end_sample, window.source_resume_sample)
        if resume < window.end_sample:
            result[resume:window.end_sample, channel] = source[resume:window.end_sample, channel]
    for start, end in window.preserved_source_intervals:
        result[start:end, channel] = source[start:end, channel]
    preserved_after = _interval_bytes(result, window.preserved_source_intervals, channel)
    untouched_after = tuple(sha256_bytes(result[:, idx].tobytes()) for idx in range(source.shape[1]) if idx != channel)
    if preserved_before != preserved_after:
        raise MountError("preserved source interval changed during montage")
    if untouched_before != untouched_after:
        raise MountError("non-dialogue channel changed during montage")
    metrics = MountMetrics(destination, destination + len(body), sha256_bytes(preserved_before), sha256_bytes(preserved_after), untouched_before, untouched_after, window.source_resume_sample, empalme_b)
    return result, metrics


__all__ = ["MountError", "MountMetrics", "mount_surgical"]
