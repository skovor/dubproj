"""Require benchmark evidence and second-game validation before promotion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import hashlib, importlib, inspect, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.benchmark import BenchmarkManifest, validate_manifest, verify_benchmark_row, _ast_calls_run_scene
from dubbing_pipeline.attestation import load_trust_store, subject_digest, verify_trusted_attestation


def _load_adapter(entrypoint: str, repository_root: Path):
    module_name, separator, symbol = entrypoint.partition(":")
    if not separator or not module_name.startswith("adapters."):
        raise ValueError("second-game adapter must be an allowlisted adapters.*:Symbol entrypoint")
    adapter_module=importlib.import_module(module_name)
    adapter=getattr(adapter_module, symbol)
    source=inspect.getsourcefile(adapter)
    allowed=(repository_root / "adapters").resolve()
    if source is None or allowed not in Path(source).resolve().parents:
        raise ValueError("second-game adapter source is outside the repository adapter allowlist")
    if any(token in f"{module_name}:{symbol}".casefold() for token in ("test", "mock", "fake")):
        raise ValueError("test/mock adapters are not promotion eligible")
    return adapter


def execute_second_game_adapter(descriptor: dict, repository_root: Path, *, expected_commit: str | None = None) -> dict:
    """Execute an adapter and recompute evidence; descriptor booleans are ignored."""
    entrypoint=str(descriptor.get("adapter_entrypoint") or "")
    manifest_path=Path(str(descriptor.get("manifest_path") or ""))
    if not entrypoint or not manifest_path.is_file():
        return {"valid": False, "errors": ["SECOND_GAME_ADAPTER_REQUIRED"]}
    try:
        adapter_type=_load_adapter(entrypoint, repository_root)
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        game_id=str(descriptor.get("game_id") or manifest.get("game_id") or "")
        adapter=adapter_type(game_id) if isinstance(adapter_type, type) else adapter_type
        evidence=adapter.validate(manifest, expected_commit=expected_commit)
        if not isinstance(evidence, dict):
            return {"valid": False, "errors": ["SECOND_GAME_ADAPTER_INVALID_RESULT"]}
        return {"executed": True, "adapter_entrypoint": entrypoint, "manifest_path": str(manifest_path.resolve()), "recomputed": evidence, "valid": bool(evidence.get("valid")), "content_verified": bool(evidence.get("content_verified")), "independent_adapter": bool(evidence.get("independent_adapter")), "game_id": evidence.get("game_id")}
    except Exception as exc:
        return {"valid": False, "errors": [f"SECOND_GAME_ADAPTER_ERROR:{exc}"]}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("benchmark"); p.add_argument("second_game"); p.add_argument("--manifest", required=True); p.add_argument("--expected-commit", required=True); p.add_argument("--attestation-trust-store", required=True); p.add_argument("--attestation-key-id", required=True); args=p.parse_args()
    benchmark_path, second_path, manifest_path = Path(args.benchmark), Path(args.second_game), Path(args.manifest)
    benchmark=json.loads(benchmark_path.read_text(encoding="utf-8")); second_descriptor=json.loads(second_path.read_text(encoding="utf-8")); manifest_value=json.loads(manifest_path.read_text(encoding="utf-8")); errors=[]
    expected_evidence = hashlib.sha256(json.dumps({key: value for key, value in benchmark.items() if key != "evidence_sha256"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if benchmark.get("evidence_sha256") != expected_evidence: errors.append({"code":"BENCHMARK_EVIDENCE_HASH_MISMATCH"})
    manifest=BenchmarkManifest(**{**manifest_value,"line_ids":tuple(manifest_value["line_ids"]),"audio_paths":tuple(manifest_value["audio_paths"]),"reference_paths":tuple(manifest_value["reference_paths"])})
    actual_commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
    if actual_commit != args.expected_commit: errors.append({"code":"COMMIT_MISMATCH","expected":args.expected_commit,"actual":actual_commit})
    if manifest.commit != args.expected_commit: errors.append({"code":"MANIFEST_COMMIT_MISMATCH","expected":args.expected_commit,"actual":manifest.commit})
    unsigned_payload={key: value for key, value in benchmark.items() if key not in {"evidence_sha256", "attestation"}}
    runner_source_sha = str((benchmark.get("runner_identity") or {}).get("source_sha256", ""))
    subject={"schema":"benchmark-attestation-subject-v1","repository":"skovor/dubproj","workflow":".github/workflows/ci.yml","manifest_digest":benchmark.get("manifest_digest"),"code_commit":manifest.commit,"benchmark_payload_sha256":subject_digest(unsigned_payload),"runner_source_sha256":runner_source_sha}
    try:
        trust_store=load_trust_store(args.attestation_trust_store)
        if not verify_trusted_attestation(benchmark.get("attestation") or {}, trust_store, key_id=args.attestation_key_id, expected_subject=subject, expected_commit=args.expected_commit, repository="skovor/dubproj", workflow=".github/workflows/ci.yml"): errors.append({"code":"BENCHMARK_ATTESTATION_INVALID"})
    except Exception as exc:
        errors.append({"code":"BENCHMARK_ATTESTATION_KEY_INVALID","details":str(exc)})
    validation=validate_manifest(manifest, require_files=True)
    if not validation["valid"]: errors.append({"code":"MANIFEST_INVALID","details":validation})
    if benchmark.get("manifest_digest") != manifest.digest(): errors.append({"code":"MANIFEST_DIGEST_MISMATCH"})
    if benchmark.get("status") not in {"EXECUTED", "VERIFIED"}: errors.append({"code":"BENCHMARK_NOT_EXECUTED","status":benchmark.get("status")})
    if int(benchmark.get("blocked",1)) or int(benchmark.get("failed",1)): errors.append({"code":"BENCHMARK_HAS_NONPASS_LINES"})
    if not benchmark.get("real_audio") or not (benchmark.get("quality") or {}).get("manifest_validation", {}).get("real_audio"): errors.append({"code":"BENCHMARK_NOT_REAL_AUDIO"})
    if not (benchmark.get("runner_identity") or {}).get("trusted"): errors.append({"code":"BENCHMARK_RUNNER_NOT_TRUSTED"})
    identity=benchmark.get("runner_identity") or {}; source_file=Path(str(identity.get("source_file") or ""))
    if identity.get("trusted"):
        if not source_file.is_file() or hashlib.sha256(source_file.read_bytes()).hexdigest() != str(identity.get("source_sha256")):
            errors.append({"code":"BENCHMARK_RUNNER_SOURCE_MISMATCH"})
        else:
            try:
                if not _ast_calls_run_scene(source_file.read_text(encoding="utf-8")): errors.append({"code":"BENCHMARK_RUNNER_CALL_NOT_VERIFIED"})
            except Exception as exc: errors.append({"code":"BENCHMARK_RUNNER_SOURCE_INVALID","details":str(exc)})
    rows=(benchmark.get("quality") or {}).get("rows", [])
    if len(rows) != len(manifest.line_ids): errors.append({"code":"BENCHMARK_ROW_COUNT_MISMATCH"})
    if any(str(row.get("status")) not in {"PASS","FINAL_PASS"} for row in rows): errors.append({"code":"BENCHMARK_ROW_STATUS_UNVERIFIED"})
    for line_id, row in zip(manifest.line_ids, rows):
        row_evidence=verify_benchmark_row(line_id, row, expected_commit=args.expected_commit)
        if not row_evidence["valid"]: errors.append({"code":"BENCHMARK_ROW_ARTIFACT_UNVERIFIED","line_id":line_id,"details":row_evidence})
    second=execute_second_game_adapter(second_descriptor, Path(__file__).resolve().parents[1], expected_commit=args.expected_commit)
    if not second.get("valid") or not second.get("independent_adapter") or not second.get("content_verified"): errors.append({"code":"SECOND_GAME_EVIDENCE_MISSING","details":second})
    result={"promotable":not errors,"errors":errors,"benchmark":benchmark.get("manifest_digest"),"second_game":second.get("game_id"),"manifest_validation":validation,"second_game_recomputed":second}; print(json.dumps(result,indent=2)); return 0 if not errors else 2
if __name__=="__main__": raise SystemExit(main())
