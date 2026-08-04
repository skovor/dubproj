"""Persistent/batched generation with complete semantic cache keys."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .audio import read, write
from .contracts import AudioArtifact, CandidateArtifact, ReferenceEvidence
from .hashing import contract_hash, sha256_file
from .models import Line
from .policy import append_ellipsis


class BatchSpeechBackend(Protocol):
    def generate(self, *, text: str, language: str, ref_audio: str, ref_text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class GenerationRequest:
    line: Line
    reference: ReferenceEvidence
    target_language: str
    model_id: str
    model_revision: str
    backend_version: str
    native_sample_rate: int
    generation_steps: int
    guidance_scale: float
    seed: int | None
    temperature: float | None
    t_shift: float | None
    postprocess_output: str
    text_normalization_version: str

    @property
    def text(self) -> str:
        return append_ellipsis(self.line.effective_target_text, True)

    def semantic_payload(self) -> dict[str, Any]:
        return {"line_id": self.line.id, "reference_id": self.reference.reference_id, "reference_audio_sha256": self.reference.audio_sha256,
                "reference_transcript": self.reference.exact_transcript, "reference_range": [self.reference.start_sample, self.reference.end_sample],
                "reference_channel": self.reference.channel, "tts_text": self.text, "target_language": self.target_language,
                "speaker_id": self.reference.speaker_id, "model_id": self.model_id, "model_revision": self.model_revision,
                "backend_version": self.backend_version, "native_sample_rate": self.native_sample_rate,
                "generation_steps": self.generation_steps, "guidance_scale": self.guidance_scale, "seed": self.seed,
                "temperature": self.temperature, "t_shift": self.t_shift, "postprocess_output": self.postprocess_output,
                "text_normalization_version": self.text_normalization_version}

    @property
    def generation_hash(self) -> str:
        return contract_hash("generation-v2", self.semantic_payload())


@dataclass
class GenerationRuntimeV2:
    backend: BatchSpeechBackend
    backend_version: str = "unknown"
    prompt_cache: dict[str, Any] = field(default_factory=dict)

    def _prepare_prompt(self, request: GenerationRequest) -> Any:
        key = request.reference.audio_sha256
        if key not in self.prompt_cache and hasattr(self.backend, "create_voice_clone_prompt"):
            self.prompt_cache[key] = self.backend.create_voice_clone_prompt(request.reference.audio_path, request.reference.exact_transcript)
        return self.prompt_cache.get(key)

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Any]:
        if not requests:
            return []
        prompts = [self._prepare_prompt(item) for item in requests]
        payload = [{"text": item.text, "language": item.target_language, "ref_audio": item.reference.audio_path,
                    "ref_text": item.reference.exact_transcript, "prompt": prompt,
                    "generation_steps": item.generation_steps, "guidance_scale": item.guidance_scale, "seed": item.seed}
                   for item, prompt in zip(requests, prompts)]
        batch = getattr(self.backend, "generate_batch", None)
        if callable(batch):
            return list(batch(payload))
        # The fallback preserves one persistent model and one reference cache;
        # only the backend call is serial when a backend lacks batching.
        return [self.backend.generate(**item) for item in payload]


def _normalise_output(output: Any, default_rate: int) -> tuple[Any, int]:
    if isinstance(output, AudioArtifact):
        audio, rate = read(output.path), output.native_sample_rate
        return audio[0], rate
    if isinstance(output, dict) and "audio" in output:
        return output["audio"], int(output.get("sample_rate", default_rate))
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], (int, float)):
        return output[0], int(output[1])
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    return output, default_rate


def _atomic_candidate(path: Path, audio: Any, sample_rate: int) -> AudioArtifact:
    import numpy as np
    value = np.asarray(audio, dtype="float32")
    if value.ndim > 2:
        value = np.squeeze(value)
    if value.ndim not in (1, 2) or len(value) == 0:
        raise ValueError("backend returned an empty or invalid audio array")
    if not np.isfinite(value).all():
        raise ValueError("backend returned non-finite samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".wav", dir=str(path.parent))
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary, value, sample_rate)
        check, rate = read(temporary, always_2d=True)
        if rate != sample_rate or len(check) != len(value) or not np.isfinite(check).all():
            raise IOError("candidate readback failed contract")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    check, rate = read(path, always_2d=True)
    peak = int(np.count_nonzero(np.abs(check) >= .999))
    return AudioArtifact(path=str(path), sha256=sha256_file(path), native_sample_rate=rate, frames=int(len(check)), channels=int(check.shape[1]), subtype="PCM_16", duration_seconds=float(len(check) / rate), nonfinite_samples=int((~np.isfinite(check)).sum()), clipping_samples=peak, producer="omnivoice", producer_version="v2")


def generate_candidates_v2(runtime: GenerationRuntimeV2, line: Line, reference: ReferenceEvidence, config: Any, *, round_index: int = 1, takes: int = 1, cache_root: str | Path | None = None) -> list[CandidateArtifact]:
    root = Path(cache_root or config.cache_root)
    requests = [GenerationRequest(line=line, reference=reference, target_language=config.target_language, model_id=config.model_id,
                                  model_revision=str(getattr(config, "model_revision", "unknown")), backend_version=runtime.backend_version,
                                  native_sample_rate=int(getattr(config, "native_sample_rate", 24000)), generation_steps=int(config.generation_steps),
                                  guidance_scale=float(config.guidance_scale), seed=(None if getattr(config, "seed", None) is None else int(config.seed) + take),
                                  temperature=getattr(config, "temperature", None), t_shift=getattr(config, "t_shift", None),
                                  postprocess_output=str(getattr(config, "postprocess_output", "none")), text_normalization_version=str(getattr(config, "text_normalization_version", "ellipsis-v1")))
                 for take in range(int(takes))]
    outputs: list[Any] = []
    pending: list[tuple[int, GenerationRequest]] = []
    result: list[CandidateArtifact] = []
    for index, request in enumerate(requests, start=1):
        path = root / "candidates" / request.generation_hash / f"r{round_index}_t{index:02d}.wav"
        if path.is_file():
            try:
                check, rate = read(path, always_2d=True)
                if rate == request.native_sample_rate and len(check) > 0:
                    result.append(CandidateArtifact(f"{line.id}:r{round_index}:t{index}", line.id, request.generation_hash, None, None, round_index, index, request.seed, str(path), None, None, "CACHED"))
                    continue
            except Exception:
                pass
            quarantine = root / "quarantine" / f"{path.stem}.corrupt.wav"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, quarantine)
        pending.append((index, request))
    if pending:
        outputs = runtime.generate_batch([request for _, request in pending])
        if len(outputs) != len(pending):
            raise ValueError(f"backend returned {len(outputs)} outputs for {len(pending)} requests")
        for (index, request), output in zip(pending, outputs):
            audio, rate = _normalise_output(output, request.native_sample_rate)
            path = root / "candidates" / request.generation_hash / f"r{round_index}_t{index:02d}.wav"
            _atomic_candidate(path, audio, rate)
            if rate != request.native_sample_rate:
                # Native SR is evidence, not a label. The destination stage
                # must explicitly resample; generation never relabels it.
                raise ValueError(f"native sample rate mismatch: expected {request.native_sample_rate}, got {rate}")
            result.append(CandidateArtifact(f"{line.id}:r{round_index}:t{index}", line.id, request.generation_hash, None, None, round_index, index, request.seed, str(path), None, None, "GENERATED"))
    return sorted(result, key=lambda item: item.take_index)


def _request_for(runtime: GenerationRuntimeV2, line: Line, reference: ReferenceEvidence, config: Any, take_index: int) -> GenerationRequest:
    return GenerationRequest(line=line, reference=reference, target_language=config.target_language, model_id=config.model_id,
                             model_revision=str(getattr(config, "model_revision", "unknown")), backend_version=runtime.backend_version,
                             native_sample_rate=int(getattr(config, "native_sample_rate", 24000)), generation_steps=int(config.generation_steps),
                             guidance_scale=float(config.guidance_scale), seed=(None if getattr(config, "seed", None) is None else int(config.seed) + take_index),
                             temperature=getattr(config, "temperature", None), t_shift=getattr(config, "t_shift", None),
                             postprocess_output=str(getattr(config, "postprocess_output", "none")), text_normalization_version=str(getattr(config, "text_normalization_version", "ellipsis-v1")))


def generate_cohort_v2(runtime: GenerationRuntimeV2, items: Sequence[tuple[Line, ReferenceEvidence, int]], config: Any, *, round_index: int = 1, cache_root: str | Path | None = None) -> dict[str, list[CandidateArtifact]]:
    """Generate all uncached takes in one backend cohort call."""
    root = Path(cache_root or config.cache_root); result: dict[str, list[CandidateArtifact]] = {line.id: [] for line, _, _ in items}; pending: list[tuple[Line, GenerationRequest, int, Path]] = []
    for line, reference, takes in items:
        for index in range(1, max(0, int(takes)) + 1):
            request = _request_for(runtime, line, reference, config, index)
            path = root / "candidates" / request.generation_hash / f"r{round_index}_t{index:02d}.wav"
            if path.is_file():
                try:
                    check, rate = read(path, always_2d=True)
                    if rate == request.native_sample_rate and len(check) > 0:
                        result[line.id].append(CandidateArtifact(f"{line.id}:r{round_index}:t{index}", line.id, request.generation_hash, None, None, round_index, index, request.seed, str(path), None, None, "CACHED")); continue
                except Exception:
                    pass
                quarantine = root / "quarantine" / f"{path.stem}.corrupt.wav"; quarantine.parent.mkdir(parents=True, exist_ok=True); os.replace(path, quarantine)
            pending.append((line, request, index, path))
    if pending:
        outputs = runtime.generate_batch([request for _, request, _, _ in pending])
        if len(outputs) != len(pending): raise ValueError(f"backend returned {len(outputs)} outputs for {len(pending)} cohort requests")
        for (line, request, index, path), output in zip(pending, outputs):
            audio, rate = _normalise_output(output, request.native_sample_rate); _atomic_candidate(path, audio, rate)
            if rate != request.native_sample_rate: raise ValueError(f"native sample rate mismatch: expected {request.native_sample_rate}, got {rate}")
            result[line.id].append(CandidateArtifact(f"{line.id}:r{round_index}:t{index}", line.id, request.generation_hash, None, None, round_index, index, request.seed, str(path), None, None, "GENERATED"))
    for line_id in result: result[line_id].sort(key=lambda item: item.take_index)
    return result


__all__ = ["BatchSpeechBackend", "GenerationRequest", "GenerationRuntimeV2", "generate_candidates_v2", "generate_cohort_v2"]
