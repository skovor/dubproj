"""Persistent OmniVoice runtime, cache keys and directed candidate rounds."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import PipelineConfig
from .hashing import atomic_json, contract_hash
from .models import Candidate, Line
from .policy import append_ellipsis


class SpeechBackend(Protocol):
    def generate(self, *, text: str, language: str, ref_audio: str, ref_text: str) -> Any: ...


@dataclass
class GenerationRuntime:
    backend: SpeechBackend
    prompt_cache: dict[tuple[str, str], Any] = field(default_factory=dict)


class OmniVoiceBackend:
    """Lazy OmniVoice adapter; it loads one model and reuses it for a round."""

    def __init__(self, config: PipelineConfig):
        try:
            import torch
            from omnivoice import OmniVoice, OmniVoiceGenerationConfig
        except ImportError as exc:
            raise RuntimeError("install the optional omnivoice/torch dependencies") from exc
        self._torch = torch
        self._config_class = OmniVoiceGenerationConfig
        self.model = OmniVoice.from_pretrained(config.model_id, device_map=config.device, dtype=getattr(torch, config.dtype, torch.float16))
        self.generation_config = OmniVoiceGenerationConfig(num_step=config.generation_steps, guidance_scale=config.guidance_scale)

    def generate(self, *, text: str, language: str, ref_audio: str, ref_text: str) -> Any:
        output = self.model.generate(text=[text], language=[language], ref_audio=[ref_audio], ref_text=[ref_text], generation_config=self.generation_config)[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return output.squeeze()


def synthesis_text(line: Line, config: PipelineConfig) -> str:
    target = line.synthesis_text_override or line.effective_target_text
    return append_ellipsis(target, config.append_ellipsis_experiment)


def generation_key(line: Line, ref_audio: str, config: PipelineConfig) -> str:
    return contract_hash("generation", {
        "line_id": line.id, "source_text": line.source_text,
        "target_text": line.effective_target_text, "tts_text": synthesis_text(line, config),
        "language": config.target_language, "model_id": config.model_id,
    }, [ref_audio])


def _take_path(cache_root: Path, key: str, round_index: int, take_index: int) -> Path:
    return cache_root / "candidates" / key / f"r{round_index}_t{take_index}.wav"


def generate_candidates(runtime: GenerationRuntime, line: Line, ref_audio: str, config: PipelineConfig, *, round_index: int = 1, takes: int | None = None, cache_root: str | Path | None = None, should_retry: Callable[[list[Candidate]], bool] | None = None) -> list[Candidate]:
    """Generate an initial batch, then a directed retry batch if requested."""
    import soundfile as sf
    root = Path(cache_root or config.cache_root); key = generation_key(line, ref_audio, config)
    count = takes if takes is not None else (config.fmv_initial_takes if line.topology == "EMBEDDED_FMV" else config.initial_takes)
    candidates: list[Candidate] = []
    for take in range(1, int(count) + 1):
        path = _take_path(root, key, round_index, take); path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            audio = runtime.backend.generate(text=synthesis_text(line, config), language=config.target_language, ref_audio=ref_audio, ref_text=line.reference_text)
            sf.write(str(path), audio, config.sample_rate)
        candidates.append(Candidate(line.id, str(path), round_index, take, synthesis_text(line, config), key))
    if should_retry and should_retry(candidates):
        retry_count = config.fmv_retry_takes if line.topology == "EMBEDDED_FMV" else config.retry_takes
        if retry_count:
            candidates.extend(generate_candidates(runtime, line, ref_audio, config, round_index=round_index + 1, takes=retry_count, cache_root=root, should_retry=None))
    return candidates


def persist_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    atomic_json(path, {"schema": "candidate-manifest-v1", "candidates": [item.to_dict() for item in candidates]})
