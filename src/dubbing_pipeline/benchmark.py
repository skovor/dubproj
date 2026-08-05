"""Reproducible benchmark and promotion evidence; synthetic runs are labelled."""
from __future__ import annotations
import hashlib, inspect, json, time
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
    content_hashes: dict[str, str] = field(default_factory=dict)
    runner_identity: str = ""

    def __post_init__(self):
        if not self.benchmark_id or not self.line_ids: raise ValueError("benchmark must contain line IDs")
        if len(self.line_ids) != len(self.audio_paths) or len(self.line_ids) != len(self.reference_paths): raise ValueError("benchmark manifest arrays must have equal length")
        for name in ("model_lock","runtime_lock","config_hash","commit"):
            if not str(getattr(self,name)).strip(): raise ValueError(f"missing benchmark identity: {name}")

    def to_dict(self): return dict(self.__dict__)
    def digest(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def from_paths(cls, **values: Any) -> "BenchmarkManifest":
        """Build a manifest whose identity includes every input byte hash."""
        line_ids=tuple(values["line_ids"]); audio_paths=tuple(values["audio_paths"]); reference_paths=tuple(values["reference_paths"])
        hashes={}
        for label, paths in (("audio", audio_paths), ("reference", reference_paths)):
            for line_id, path in zip(line_ids, paths):
                hashes[f"{label}:{line_id}"]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        hashes["model_lock"]=hashlib.sha256(Path(values["model_lock"]).read_bytes()).hexdigest()
        hashes["runtime_lock"]=hashlib.sha256(Path(values["runtime_lock"]).read_bytes()).hexdigest()
        values={**values,"line_ids":line_ids,"audio_paths":audio_paths,"reference_paths":reference_paths,"content_hashes":hashes}
        return cls(**values)

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
    runner_identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self): return dict(self.__dict__)

def validate_manifest(manifest: BenchmarkManifest, *, require_files: bool = True) -> dict[str, Any]:
    missing=[]
    if require_files:
        for path in (*manifest.audio_paths,*manifest.reference_paths,manifest.model_lock,manifest.runtime_lock):
            if not Path(path).is_file(): missing.append(path)
    observed={}
    if require_files and not missing:
        for label, paths in (("audio", manifest.audio_paths), ("reference", manifest.reference_paths)):
            for line_id, path in zip(manifest.line_ids, paths):
                observed[f"{label}:{line_id}"]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        observed["model_lock"]=hashlib.sha256(Path(manifest.model_lock).read_bytes()).hexdigest()
        observed["runtime_lock"]=hashlib.sha256(Path(manifest.runtime_lock).read_bytes()).hexdigest()
        if not manifest.content_hashes:
            missing.append("content_hashes")
        else:
            for key, value in observed.items():
                if manifest.content_hashes.get(key) != value:
                    missing.append(f"content_hash_mismatch:{key}")
    audio_suffixes = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus"}
    derived_real_audio = bool(require_files and not missing and all(Path(path).suffix.casefold() in audio_suffixes and Path(path).stat().st_size > 44 for path in (*manifest.audio_paths, *manifest.reference_paths)))
    return {"valid":not missing,"missing":missing,"manifest_digest":manifest.digest(),"real_audio":derived_real_audio,"declared_real_audio":bool(manifest.real_audio),"content_hashes":observed}

def trusted_runner_identity(runner: Callable[[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Accept only an inspectable runner that delegates to run_scene_v2."""
    if not callable(runner):
        raise TypeError("a callable runner is required")
    module = str(getattr(runner, "__module__", "")); qualname = str(getattr(runner, "__qualname__", ""))
    lowered = f"{module}:{qualname}".casefold()
    if any(token in lowered for token in ("test", "mock", "fake", "fixture", "lambda")):
        raise ValueError("benchmark runner identity is not production-eligible")
    try:
        source = inspect.getsource(runner)
    except (OSError, TypeError) as exc:
        raise ValueError("benchmark runner source is not inspectable") from exc
    if "run_scene_v2" not in source:
        raise ValueError("benchmark runner must invoke run_scene_v2")
    return {"module": module, "qualname": qualname, "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(), "trusted": True}


def run_benchmark(manifest: BenchmarkManifest, runner: Callable[[str, str, str], Mapping[str, Any]], *, require_files: bool = True, require_trusted_runner: bool = False) -> BenchmarkResult:
    if not callable(runner):
        raise TypeError("a real pipeline runner is required; fixed/mock runners are forbidden")
    validation=validate_manifest(manifest,require_files=require_files)
    if not validation["valid"]: raise ValueError(f"benchmark manifest has missing files: {validation['missing']}")
    identity = trusted_runner_identity(runner) if require_trusted_runner else {"trusted": False, "mode": "caller_supplied"}
    started=time.perf_counter(); passed=failed=blocked=0; stage_time={}; quality=[]
    for line_id,audio,reference in zip(manifest.line_ids,manifest.audio_paths,manifest.reference_paths):
        item_start=time.perf_counter(); result=dict(runner(line_id,audio,reference)); stage_time["line_total"] = stage_time.get("line_total",0.0)+(time.perf_counter()-item_start); status=str(result.get("status","BLOCKED")); passed+=status in {"PASS","FINAL_PASS"}; failed+=status in {"FAIL","FAILED"}; blocked+=status not in {"PASS","FINAL_PASS","FAIL","FAILED"}; quality.append(result)
    elapsed=max(1e-9,time.perf_counter()-started); return BenchmarkResult(manifest.digest(),len(manifest.line_ids),elapsed,len(manifest.line_ids)/(elapsed/60),passed,failed,blocked,bool(validation.get("real_audio")),stage_time,{"rows":quality,"manifest_validation":validation},False,identity)

__all__=["BenchmarkManifest","BenchmarkResult","validate_manifest","trusted_runner_identity","run_benchmark"]
