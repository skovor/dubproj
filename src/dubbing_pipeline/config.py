"""Project configuration with no game-specific defaults."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _path(value: str | Path | None, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


@dataclass
class QAConfig:
    """Hard-gate and diagnostic policy shared by both topologies."""

    hard_gates: list[str] = field(default_factory=lambda: [
        "not_empty", "finite_audio", "sample_rate", "channels", "frames",
        "clipping", "active_loudness", "source_language", "tail",
        "final_word", "content", "serialization_contract",
    ])
    diagnostic_metrics: list[str] = field(default_factory=lambda: [
        "text", "wer", "onset", "span", "rate", "pause", "pitch_identity",
    ])
    final_word_min_tokens: int = 1
    tail_guard_ms: float = 80.0
    max_lufs_delta: float = 3.5
    max_splice_speech_onset_error_ms: float = 60.0
    max_seam_notch_db: float = 12.0
    english_markers: list[str] = field(default_factory=lambda: [
        "the", "you", "what", "why", "yes", "no", "not", "are", "is",
        "can", "will", "this", "that", "your", "to", "of", "and",
    ])
    strong_source_words: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """All paths and policy knobs required by the generic runner.

    Relative paths are resolved against the directory containing the config
    file. Environment variables may be used as ``${NAME}`` in strings; no
    secret is stored in this file.
    """

    project_root: Path = Path(".")
    source_language: str = "en"
    target_language: str = "de"
    topology_default: str = "LINE_SEPARATED"
    output_root: Path = Path("artifacts")
    cache_root: Path = Path("cache")
    model_id: str = "k2-fsa/OmniVoice"
    model_revision: str = "unknown"
    backend_version: str = "unknown"
    device: str = "cuda"
    dtype: str = "float16"
    generation_steps: int = 32
    guidance_scale: float = 2.0
    initial_takes: int = 1
    retry_takes: int = 0
    fmv_initial_takes: int = 4
    fmv_retry_takes: int = 4
    append_ellipsis_experiment: bool = True
    sample_rate: int = 48000
    native_sample_rate: int = 24000
    channels: int = 1
    seed: int | None = None
    temperature: float | None = None
    t_shift: float | None = None
    postprocess_output: str = "none"
    text_normalization_version: str = "ellipsis-v1"
    dialogue_channel: int = 0
    ffmpeg: Path | None = None
    vgmstream: Path | None = None
    vgaudio: Path | None = None
    runtime_root: Path | None = None
    runtime_adapter: str | None = None
    reference_root: Path | None = None
    lab_mode: bool = True
    sandbox_root: Path | None = None
    qa: QAConfig = field(default_factory=QAConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent

        def expand(value: Any) -> Any:
            if isinstance(value, str):
                return os.path.expandvars(value)
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, dict):
                return {key: expand(item) for key, item in value.items()}
            return value

        raw = expand(raw)
        qa = QAConfig(**raw.pop("qa", {}))
        known = {
            "project_root", "source_language", "target_language", "topology_default",
            "output_root", "cache_root", "model_id", "model_revision", "backend_version", "device", "dtype",
            "generation_steps", "guidance_scale", "initial_takes", "retry_takes",
            "fmv_initial_takes", "fmv_retry_takes", "append_ellipsis_experiment",
            "sample_rate", "native_sample_rate", "channels", "seed", "temperature", "t_shift", "postprocess_output", "text_normalization_version", "dialogue_channel", "ffmpeg", "vgmstream", "vgaudio",
            "runtime_root", "runtime_adapter", "reference_root", "lab_mode", "sandbox_root",
        }
        values = {key: raw.pop(key) for key in list(raw) if key in known}
        for key in ("project_root", "output_root", "cache_root", "ffmpeg", "vgmstream", "vgaudio", "runtime_root", "reference_root", "sandbox_root"):
            if key in values:
                values[key] = _path(values[key], base)
        values["qa"] = qa
        values["extra"] = raw
        config = cls(**values)
        config.project_root = config.project_root or base
        config.output_root = config.output_root or (base / "artifacts")
        config.cache_root = config.cache_root or (base / "cache")
        return config

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("project_root", "output_root", "cache_root", "ffmpeg", "vgmstream", "vgaudio", "runtime_root", "reference_root", "sandbox_root"):
            if result.get(key) is not None:
                result[key] = str(result[key])
        return result
