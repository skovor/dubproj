"""Minimal review payload builder; optional UI integrations must use it."""
from __future__ import annotations

from typing import Iterable

from .goldset import ClipRecord


def review_rows(clips: Iterable[ClipRecord]) -> list[dict]:
    return [clip.review_payload() for clip in clips]


def gradio_available() -> bool:
    try:
        import gradio  # type: ignore
    except ImportError:
        return False
    return True


__all__ = ["review_rows", "gradio_available"]
