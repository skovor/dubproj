"""Optional ASR layer used for screening and confirmation, never for mapping authority."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ASRBackend(Protocol):
    def transcribe(self, path: str | Path, *, language: str | None = None) -> dict[str, Any]: ...


class FasterWhisperBackend:
    """Lazy faster-whisper adapter; model stays alive for one QA round."""

    def __init__(self, model_size: str = "large-v3-turbo", *, device: str = "cuda", compute_type: str = "float16"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("install the optional qa dependencies") from exc
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, path: str | Path, *, language: str | None = None) -> dict[str, Any]:
        segments, info = self.model.transcribe(str(path), language=language, beam_size=5, vad_filter=True)
        rows = list(segments)
        text = " ".join(item.text.strip() for item in rows).strip()
        return {"text": text, "language": getattr(info, "language", None), "probability": getattr(info, "language_probability", None), "segments": [{"start": item.start, "end": item.end, "text": item.text} for item in rows]}


@dataclass
class ASRScreeningResult:
    text: str
    language: str | None
    probability: float | None
    confirmed_target: bool = False
    segments: list[dict[str, Any]] | None = None


def screen_and_confirm(backend: ASRBackend, path: str | Path, *, source_language: str, target_language: str) -> ASRScreeningResult:
    source = backend.transcribe(path, language=source_language)
    language = source.get("language"); probability = source.get("probability")
    target = backend.transcribe(path, language=target_language)
    return ASRScreeningResult(
        text=str(source.get("text", "")), language=language, probability=probability,
        confirmed_target=bool(target.get("text", "").strip()) and target.get("language") == target_language,
        segments=source.get("segments"),
    )
