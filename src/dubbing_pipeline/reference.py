"""Materialise exact reference segments before any TTS call."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .audio import read, write
from .contracts import ContractError, ReferenceEvidence
from .hashing import contract_hash, sha256_file
from .models import Line


def _resolve(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (Path(project_root) / candidate).resolve()


def _atomic_audio(path: Path, audio: Any, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".wav", dir=str(path.parent))
    os.close(handle)
    temporary = Path(name)
    try:
        write(temporary, audio, sample_rate)
        # Reopen before publishing so a truncated encoder output is never a
        # valid-looking reference cache hit.
        check, rate = read(temporary)
        if rate != sample_rate or len(check) == 0:
            raise ContractError(f"materialized reference failed readback: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_reference(line: Line, project_root: str | Path, cache_root: str | Path, *, language: str = "en", extraction_tool: str = "dubbing_pipeline.reference", extraction_tool_version: str = "v2") -> ReferenceEvidence:
    """Return a validated physical reference paired with its exact transcript."""
    if line.reference_segments:
        segments = line.reference_segments
        source = _resolve(segments[0].path, project_root)
    elif line.reference_audio:
        segments = []
        source = _resolve(line.reference_audio, project_root)
    else:
        raise ContractError(f"{line.id}: no reference audio declared")
    if not source.is_file():
        raise ContractError(f"{line.id}: reference does not exist: {source}")
    if segments:
        pieces = []
        texts: list[str] = []
        selected_channel: int | None = None
        channel_mode_set = False
        sample_rate: int | None = None
        source_hashes: list[str] = []
        source_paths: list[Path] = []
        for segment in segments:
            segment_source = _resolve(segment.path, project_root)
            if not segment_source.is_file():
                raise ContractError(f"{line.id}: reference segment does not exist: {segment_source}")
            full, segment_rate = read(segment_source, always_2d=True)
            if full.ndim != 2 or len(full) == 0:
                raise ContractError(f"{line.id}: reference segment is empty or malformed")
            if sample_rate is None: sample_rate = segment_rate
            if segment_rate != sample_rate:
                raise ContractError(f"{line.id}: reference segments use different sample rates")
            start = max(0, round(float(segment.start) * segment_rate))
            end = len(full) if segment.end is None else round(float(segment.end) * segment_rate)
            if not 0 <= start < end <= len(full):
                raise ContractError(f"{line.id}: reference segment range outside source")
            channel = segment.channel
            if channel is not None and not 0 <= channel < full.shape[1]:
                raise ContractError(f"{line.id}: reference channel outside source")
            if channel_mode_set and channel != selected_channel:
                raise ContractError(f"{line.id}: reference segments must use one consistent channel")
            if not channel_mode_set:
                selected_channel = channel; channel_mode_set = True
            piece = full[start:end, channel] if channel is not None else full[start:end]
            pieces.append(piece)
            if segment.text.strip():
                texts.append(segment.text.strip())
            source_hashes.append(sha256_file(segment_source))
            source_paths.append(segment_source)
        audio = pieces[0] if len(pieces) == 1 else __import__("numpy").concatenate([__import__("numpy").asarray(item) for item in pieces], axis=0)
        transcript = " ".join(texts).strip() or line.source_text.strip()
        channel = selected_channel
        assert sample_rate is not None
    else:
        full, sample_rate = read(source, always_2d=True)
        if full.ndim != 2 or len(full) == 0:
            raise ContractError(f"{line.id}: reference audio is empty or malformed")
        audio = full[:, 0] if full.shape[1] == 1 else full
        transcript = line.source_text.strip()
        channel = None
        source_hashes = [sha256_file(source)]
        source_paths = [source]
    if not transcript:
        raise ContractError(f"{line.id}: reference transcript is empty")
    import numpy as np
    value = np.asarray(audio, dtype="float32")
    if not np.isfinite(value).all():
        raise ContractError(f"{line.id}: reference contains non-finite samples")
    semantic = {"line_id": line.id, "source_line_id": line.id, "transcript": transcript, "language": language, "speaker": line.speaker, "sample_rate": sample_rate, "channel": channel, "source_sha256": source_hashes if segments else sha256_file(source)}
    reference_id = contract_hash("reference", semantic, source_paths)
    target = Path(cache_root) / "references" / f"{reference_id}.wav"
    if not target.is_file():
        _atomic_audio(target, value, sample_rate)
    materialized, materialized_rate = read(target, always_2d=True)
    if materialized_rate != sample_rate or len(materialized) != len(value):
        raise ContractError(f"{line.id}: reference cache readback mismatch")
    audio_hash = sha256_file(target)
    validation_hash = contract_hash("reference-validation", {"reference_id": reference_id, "audio_sha256": audio_hash, "transcript": transcript, "rate": sample_rate, "frames": len(materialized)})
    # The materialized artifact is mono when a source channel was selected;
    # channel=0 then names the physical channel in the artifact. The original
    # channel choice is retained in the semantic reference hash.
    evidence_channel = 0 if channel is not None else None
    return ReferenceEvidence(reference_id=reference_id, audio_path=str(target), audio_sha256=audio_hash,
                             native_sample_rate=sample_rate, channels=int(materialized.shape[1]), samples=int(len(materialized)),
                             start_sample=0, end_sample=int(len(materialized)), channel=evidence_channel, exact_transcript=transcript,
                             language=language, speaker_id=line.speaker, source_line_id=line.id,
                             extraction_tool=extraction_tool, extraction_tool_version=extraction_tool_version,
                             validation_hash=validation_hash)


__all__ = ["materialize_reference"]
