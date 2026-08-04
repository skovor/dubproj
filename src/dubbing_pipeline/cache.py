"""Validated artifact cache helpers; corrupt hits go to quarantine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .audio import read
from .hashing import sha256_file


def validated_audio_hit(path: str | Path, *, sample_rate: int, quarantine_root: str | Path | None = None, validator: Callable | None = None) -> bool:
    candidate = Path(path)
    try:
        audio, rate = read(candidate, always_2d=True)
        import numpy as np
        valid = candidate.is_file() and rate == sample_rate and len(audio) > 0 and bool(np.isfinite(audio).all())
        if valid and validator is not None:
            valid = bool(validator(audio, rate))
        if valid:
            return True
    except Exception:
        valid = False
    if candidate.exists() and quarantine_root is not None:
        root = Path(quarantine_root); root.mkdir(parents=True, exist_ok=True)
        target = root / f"{candidate.stem}.{sha256_file(candidate)[:12]}.corrupt{candidate.suffix}"
        os.replace(candidate, target)
    return False


__all__ = ["validated_audio_hit"]
