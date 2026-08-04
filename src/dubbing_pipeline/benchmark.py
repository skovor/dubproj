"""Reproducible benchmark and promotion evidence; synthetic runs are labelled."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    line_ids: tuple[str, ...]
    audio_paths: tuple[str, ...]
    reference_paths: tuple[str, ...]
    model_lock: str
    runtime_lock: str
    calibration_profile: str | None
    config_hash: str
    commit: str
    topology: str
    real_audio: bool = False

    def __post_init__(self):
        if not self.benchmark_id or not self.line_ids: raise ValueError("benchmark must contain line IDs")
        if len(self.line_ids) != len(self.audio_paths) or len(self.line_ids) != len(self.reference_paths): raise ValueError("benchmark manifest arrays must have equal length")
        for name in ("model_lock","runtime_lock","config_hash","commit"):
            if not str(getattr(self,name)).strip(): raise ValueError(f"missing benchmark identity: {name}")

    def to_dict(self): return dict(self.__dict__)
    def digest(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

@dataclass(frozen=True)
class BenchmarkResult:
    manifest_digest: str
    line_count: int
    elapsed_seconds: float
    lines_per_minute: float
    passed: int
    failed: int
    blocked: int
    real_audio: bool
    stages: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    second_game: bool = False

    def to_dict(self): return dict(self.__dict__)

def validate_manifest(manifest: BenchmarkManifest, *, require_files: bool = True) -> dict[str, Any]:
    missing=[]
    if require_files:
        for path in (*manifest.audio_paths,*manifest.reference_paths,manifest.model_lock,manifest.runtime_lock):
            if not Path(path).is_file(): missing.append(path)
    return {"valid":not missing,"missing":missing,"manifest_digest":manifest.digest(),"real_audio":manifest.real_audio}

def run_benchmark(manifest: BenchmarkManifest, runner: Callable[[str, str, str], Mapping[str, Any]], *, require_files: bool = True) -> BenchmarkResult:
    validation=validate_manifest(manifest,require_files=require_files)
    if not validation["valid"]: raise ValueError(f"benchmark manifest has missing files: {validation['missing']}")
    started=time.perf_counter(); passed=failed=blocked=0; stage_time={}; quality=[]
    for line_id,audio,reference in zip(manifest.line_ids,manifest.audio_paths,manifest.reference_paths):
        item_start=time.perf_counter(); result=dict(runner(line_id,audio,reference)); stage_time["line_total"] = stage_time.get("line_total",0.0)+(time.perf_counter()-item_start); status=str(result.get("status","BLOCKED")); passed+=status in {"PASS","FINAL_PASS"}; failed+=status in {"FAIL","FAILED"}; blocked+=status not in {"PASS","FINAL_PASS","FAIL","FAILED"}; quality.append(result)
    elapsed=max(1e-9,time.perf_counter()-started); return BenchmarkResult(manifest.digest(),len(manifest.line_ids),elapsed,len(manifest.line_ids)/(elapsed/60),passed,failed,blocked,manifest.real_audio,stage_time,{"rows":quality},False)

__all__=["BenchmarkManifest","BenchmarkResult","validate_manifest","run_benchmark"]
