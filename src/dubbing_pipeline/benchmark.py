"""Reproducible benchmark and promotion evidence; synthetic runs are labelled."""
from __future__ import annotations
import ast, hashlib, inspect, json, textwrap, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from .audio import read
from .hashing import canonical_json, contract_hash, sha256_bytes, sha256_file


def _report_without_invocation_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report); value.pop("invocation_receipt", None); return value


def build_invocation_receipt(report: Mapping[str, Any], *, code_commit: str) -> dict[str, Any]:
    """Create a receipt over the report before the receipt is inserted."""
    import re
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(code_commit or "")):
        raise ValueError("scene invocation requires a full code commit SHA")
    base = _report_without_invocation_receipt(report)
    outputs: dict[str, str] = {}
    for row in base.get("lines", []) if isinstance(base.get("lines"), list) else []:
        if isinstance(row, Mapping) and row.get("output") and Path(str(row["output"])).is_file():
            outputs[str(row.get("id") or row.get("line_id"))] = sha256_file(Path(str(row["output"])))
    if base.get("mounted_output") and Path(str(base["mounted_output"])).is_file():
        outputs["__scene__"] = sha256_file(Path(str(base["mounted_output"])))
    payload = {"schema": "scene-invocation-receipt-v1", "scene_id": str(base.get("scene_id") or base.get("scene") or ""), "run_id": str(base.get("run_id") or contract_hash("scene-run", {"scene": base.get("scene_id") or base.get("scene"), "contract": base.get("contract_hash")})), "code_commit": str(code_commit).lower(), "report_sha256": sha256_bytes(canonical_json(base)), "outputs_sha256": outputs}
    payload["receipt_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def verify_invocation_receipt(report: Mapping[str, Any], *, line_id: str, output_sha256: str, expected_commit: str | None = None) -> dict[str, Any]:
    receipt = report.get("invocation_receipt") if isinstance(report, Mapping) else None
    errors: list[str] = []
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "scene-invocation-receipt-v1":
        return {"valid": False, "errors": ["invocation_receipt_missing"]}
    payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if str(receipt.get("receipt_sha256", "")) != sha256_bytes(canonical_json(payload)): errors.append("invocation_receipt_hash_mismatch")
    if str(receipt.get("report_sha256", "")) != sha256_bytes(canonical_json(_report_without_invocation_receipt(report))): errors.append("invocation_report_hash_mismatch")
    if expected_commit is not None and str(receipt.get("code_commit", "")).lower() != str(expected_commit).lower(): errors.append("invocation_commit_mismatch")
    outputs = receipt.get("outputs_sha256") if isinstance(receipt.get("outputs_sha256"), Mapping) else {}
    if str(outputs.get(str(line_id), "")) != str(output_sha256): errors.append("invocation_output_hash_mismatch")
    return {"valid": not errors, "errors": errors, "code_commit": receipt.get("code_commit"), "run_id": receipt.get("run_id")}

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
    decoded=[]; decode_errors=[]
    if require_files and not missing:
        for label, paths in (("audio", manifest.audio_paths), ("reference", manifest.reference_paths)):
            for line_id, path in zip(manifest.line_ids, paths):
                try:
                    value, rate = read(path, always_2d=True)
                    import numpy as np
                    if len(value) <= 0 or int(rate) <= 0 or value.ndim != 2 or value.shape[1] <= 0 or not bool(np.isfinite(value).all()):
                        raise ValueError("decoded audio has invalid frames/rate/channels/finiteness")
                    decoded.append({"kind": label, "line_id": line_id, "path": str(path), "frames": int(len(value)), "sample_rate": int(rate), "channels": int(value.shape[1]), "sha256": sha256_file(path)})
                except Exception as exc:
                    decode_errors.append({"kind": label, "line_id": line_id, "path": str(path), "error": str(exc)})
    if decode_errors:
        missing.extend([f"audio_decode:{item['kind']}:{item['line_id']}" for item in decode_errors])
    derived_real_audio = bool(require_files and not missing and len(decoded) == 2 * len(manifest.line_ids))
    return {"valid":not missing,"missing":missing,"manifest_digest":manifest.digest(),"real_audio":derived_real_audio,"declared_real_audio":bool(manifest.real_audio),"content_hashes":observed,"decoded_audio":decoded,"decode_errors":decode_errors}

def _ast_calls_run_scene(source: str) -> bool:
    tree=ast.parse(textwrap.dedent(source))
    class Visitor(ast.NodeVisitor):
        found=False
        def visit_If(self, node):
            if isinstance(node.test, ast.Constant) and node.test.value is False:
                for child in node.orelse: self.visit(child)
                return
            for child in node.body + node.orelse: self.visit(child)
        def visit_Call(self, node):
            function=node.func
            if (isinstance(function, ast.Name) and function.id == "run_scene_v2") or (isinstance(function, ast.Attribute) and function.attr == "run_scene_v2"):
                self.found=True
            self.generic_visit(node)
    visitor=Visitor(); visitor.visit(tree); return bool(visitor.found)


def trusted_runner_identity(runner: Callable[[str, str, str], Mapping[str, Any]], *, repository_root: str | Path | None = None) -> dict[str, Any]:
    """Accept only repository code with an executable AST call to run_scene_v2."""
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
    source_file=inspect.getsourcefile(runner)
    root=Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    if source_file is None or root not in Path(source_file).resolve().parents:
        raise ValueError("benchmark runner must live inside the repository allowlist")
    if not _ast_calls_run_scene(source):
        raise ValueError("benchmark runner must contain an executable run_scene_v2 call")
    source_path=Path(source_file).resolve()
    return {"module": module, "qualname": qualname, "source_file": str(source_path), "source_sha256": sha256_file(source_path), "trusted": True, "allowlist_root": str(root)}


def _verify_runner_row(line_id: str, result: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    """Reopen and verify the report/output artifacts emitted by a trusted runner."""
    required=("report_path", "output_path", "report_sha256", "output_sha256", "report_contract_hash")
    missing=[name for name in required if not str(result.get(name) or "").strip()]
    if missing: return {"valid": False, "errors":[f"row_missing:{name}" for name in missing]}
    report_path=Path(str(result["report_path"])); output_path=Path(str(result["output_path"]))
    errors=[]
    if not report_path.is_file(): errors.append("report_missing")
    if not output_path.is_file(): errors.append("output_missing")
    if errors: return {"valid": False, "errors": errors}
    observed_report_sha=sha256_file(report_path); observed_output_sha=sha256_file(output_path)
    if observed_report_sha != str(result["report_sha256"]): errors.append("report_sha256_mismatch")
    if observed_output_sha != str(result["output_sha256"]): errors.append("output_sha256_mismatch")
    try:
        report=json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "errors": [*errors, f"report_json_invalid:{exc}"]}
    try:
        audio, rate=read(output_path, always_2d=True)
        import numpy as np
        if len(audio) <= 0 or int(rate) <= 0 or audio.ndim != 2 or not bool(np.isfinite(audio).all()): errors.append("output_audio_invalid")
    except Exception as exc:
        errors.append(f"output_audio_invalid:{exc}")
    lines=report.get("lines") if isinstance(report, Mapping) else None
    line_rows=[row for row in (lines or []) if isinstance(row, Mapping) and str(row.get("id") or row.get("line_id")) == str(line_id)]
    if len(line_rows) != 1: errors.append("report_line_missing_or_ambiguous")
    else:
        line_row=line_rows[0]
        if str(line_row.get("status")) not in {"FINAL_PASS", "PASS"}: errors.append("report_line_not_final_pass")
        declared_output=line_row.get("output")
        if declared_output and Path(str(declared_output)).resolve() != output_path.resolve(): errors.append("report_output_mismatch")
    computed_contract=contract_hash("benchmark-report-v1", {"line_id": str(line_id), "report": report}, files=[output_path])
    if computed_contract != str(result["report_contract_hash"]): errors.append("report_contract_hash_mismatch")
    invocation = verify_invocation_receipt(report, line_id=str(line_id), output_sha256=observed_output_sha, expected_commit=expected_commit)
    if not invocation["valid"]: errors.extend(invocation["errors"])
    return {"valid": not errors, "errors": errors, "report_sha256": observed_report_sha, "output_sha256": observed_output_sha, "report_contract_hash": computed_contract, "invocation": invocation}


def verify_benchmark_row(line_id: str, result: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    """Public promotion-time recheck of one emitted report/output pair."""
    return _verify_runner_row(line_id, result, expected_commit=expected_commit)


def run_benchmark(manifest: BenchmarkManifest, runner: Callable[[str, str, str], Mapping[str, Any]], *, require_files: bool = True, require_trusted_runner: bool = False, repository_root: str | Path | None = None) -> BenchmarkResult:
    if not callable(runner):
        raise TypeError("a real pipeline runner is required; fixed/mock runners are forbidden")
    validation=validate_manifest(manifest,require_files=require_files)
    if not validation["valid"]: raise ValueError(f"benchmark manifest has missing files: {validation['missing']}")
    identity = trusted_runner_identity(runner, repository_root=repository_root) if require_trusted_runner else {"trusted": False, "mode": "caller_supplied"}
    started=time.perf_counter(); passed=failed=blocked=0; stage_time={}; quality=[]
    for line_id,audio,reference in zip(manifest.line_ids,manifest.audio_paths,manifest.reference_paths):
        item_start=time.perf_counter(); result=dict(runner(line_id,audio,reference)); stage_time["line_total"] = stage_time.get("line_total",0.0)+(time.perf_counter()-item_start)
        if require_trusted_runner:
            row_evidence=_verify_runner_row(line_id, result, expected_commit=manifest.commit if len(str(manifest.commit)) == 40 else None); result={**result, "artifact_verification": row_evidence}
            if not row_evidence["valid"]: result={**result, "status": "BLOCKED"}
        status=str(result.get("status","BLOCKED")); passed+=status in {"PASS","FINAL_PASS"}; failed+=status in {"FAIL","FAILED"}; blocked+=status not in {"PASS","FINAL_PASS","FAIL","FAILED"}; quality.append(result)
    elapsed=max(1e-9,time.perf_counter()-started); return BenchmarkResult(manifest.digest(),len(manifest.line_ids),elapsed,len(manifest.line_ids)/(elapsed/60),passed,failed,blocked,bool(validation.get("real_audio")),stage_time,{"rows":quality,"manifest_validation":validation,"evidence_integrity":{"mode":"self_hash_only","promotion_allowed":False}},False,identity)

__all__=["BenchmarkManifest","BenchmarkResult","validate_manifest","trusted_runner_identity","verify_benchmark_row","run_benchmark","build_invocation_receipt","verify_invocation_receipt"]
