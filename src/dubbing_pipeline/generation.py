"""Persistent OmniVoice runtime, cache keys and directed candidate rounds."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import PipelineConfig
from .hashing import atomic_json, contract_hash, sha256_file
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
        self.native_sample_rate = int(getattr(config, "native_sample_rate", 24000))

    def generate_batch(self, payload: list[dict[str, Any]]) -> list[Any]:
        output = self.model.generate(text=[item["text"] for item in payload], language=[item["language"] for item in payload], ref_audio=[item["ref_audio"] for item in payload], ref_text=[item["ref_text"] for item in payload], generation_config=self.generation_config)
        result = []
        for item in output:
            if hasattr(item, "detach"):
                item = item.detach().cpu().numpy()
            result.append((item.squeeze(), self.native_sample_rate))
        return result

    def generate(self, *, text: str, language: str, ref_audio: str, ref_text: str, **_kwargs: Any) -> Any:
        output = self.generate_batch([{"text": text, "language": language, "ref_audio": ref_audio, "ref_text": ref_text}])[0]
        return output


def synthesis_text(line: Line, config: PipelineConfig) -> str:
    target = line.synthesis_text_override or line.effective_target_text
    return append_ellipsis(target, config.append_ellipsis_experiment)


def generation_key(line: Line, ref_audio: str, config: PipelineConfig) -> str:
    reference_hash = sha256_file(ref_audio) if Path(ref_audio).is_file() else None
    return contract_hash("generation", {
        "line_id": line.id, "source_text": line.source_text,
        "target_text": line.effective_target_text, "tts_text": synthesis_text(line, config),
        "language": config.target_language, "model_id": config.model_id,
        "model_revision": getattr(config, "model_revision", "unknown"),
        "backend_version": getattr(config, "backend_version", "unknown"),
        "generation_steps": config.generation_steps, "guidance_scale": config.guidance_scale,
        "native_sample_rate": getattr(config, "native_sample_rate", config.sample_rate),
        "seed": getattr(config, "seed", None), "temperature": getattr(config, "temperature", None),
        "t_shift": getattr(config, "t_shift", None), "postprocess_output": getattr(config, "postprocess_output", "none"),
        "text_normalization_version": getattr(config, "text_normalization_version", "ellipsis-v1"),
        "reference_audio_sha256": reference_hash,
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
            output = runtime.backend.generate(text=synthesis_text(line, config), language=config.target_language, ref_audio=ref_audio, ref_text=line.reference_text)
            output_rate = getattr(config, "native_sample_rate", config.sample_rate)
            if isinstance(output, tuple) and len(output) == 2:
                output, output_rate = output
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            sf.write(str(path), output, int(output_rate))
        else:
            try:
                info = sf.info(str(path))
                expected_rate = int(getattr(config, "native_sample_rate", config.sample_rate))
                if info.samplerate != expected_rate or info.frames <= 0:
                    path.unlink()
                    raise ValueError("stale or corrupt candidate cache")
            except Exception:
                if path.exists(): path.unlink()
                output = runtime.backend.generate(text=synthesis_text(line, config), language=config.target_language, ref_audio=ref_audio, ref_text=line.reference_text)
                output_rate = getattr(config, "native_sample_rate", config.sample_rate)
                if isinstance(output, tuple) and len(output) == 2:
                    output, output_rate = output
                sf.write(str(path), output, int(output_rate))
        candidates.append(Candidate(line.id, str(path), round_index, take, synthesis_text(line, config), key))
    if should_retry and should_retry(candidates):
        retry_count = config.fmv_retry_takes if line.topology == "EMBEDDED_FMV" else config.retry_takes
        if retry_count:
            candidates.extend(generate_candidates(runtime, line, ref_audio, config, round_index=round_index + 1, takes=retry_count, cache_root=root, should_retry=None))
    return candidates


def persist_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    atomic_json(path, {"schema": "candidate-manifest-v1", "candidates": [item.to_dict() for item in candidates]})
