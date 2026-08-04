"""Post-generation processing in the canonical, non-destructive order."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import read, resample_exact
from .hashing import contract_hash
from .timing import correct_length


@dataclass(frozen=True)
class ProcessingResult:
    audio: Any
    sample_rate: int
    processing_hash: str
    diagnostics: dict[str, Any]


def process_candidate(path: str | Path, *, target_sample_rate: int, reference_end: float | None = None, ffmpeg: str | Path | None = None, tmpdir: str | Path | None = None, under_tol: float = .35, over_tol: float = .35, max_ratio_deviation: float = .20) -> ProcessingResult:
    audio, source_rate = read(path, always_2d=True)
    value = audio
    diagnostics: dict[str, Any] = {"source_sample_rate": source_rate, "target_sample_rate": target_sample_rate, "steps": []}
    if source_rate != target_sample_rate:
        value = resample_exact(value, source_rate, target_sample_rate); diagnostics["steps"].append("explicit_resample")
    if reference_end is not None:
        value, duration_info = correct_length(value, target_sample_rate, reference_end, ffmpeg, tmpdir or Path(path).parent, under_tol=under_tol, over_tol=over_tol, max_ratio_deviation=max_ratio_deviation)
        diagnostics["duration"] = duration_info; diagnostics["steps"].append(duration_info.get("method"))
    processing_hash = contract_hash("processing-v2", {"source_path_sha256": __import__("dubbing_pipeline.hashing", fromlist=["sha256_file"]).sha256_file(path), "source_rate": source_rate, "target_rate": target_sample_rate, "reference_end": reference_end, "under_tol": under_tol, "over_tol": over_tol, "max_ratio_deviation": max_ratio_deviation})
    return ProcessingResult(value, target_sample_rate, processing_hash, diagnostics)


__all__ = ["ProcessingResult", "process_candidate"]
