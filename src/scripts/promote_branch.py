"""Require benchmark evidence and second-game validation before promotion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import hashlib, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.benchmark import BenchmarkManifest, validate_manifest

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("benchmark"); p.add_argument("second_game"); p.add_argument("--manifest", required=True); p.add_argument("--expected-commit"); args=p.parse_args()
    benchmark=json.loads(Path(args.benchmark).read_text(encoding="utf-8")); second=json.loads(Path(args.second_game).read_text(encoding="utf-8")); manifest_value=json.loads(Path(args.manifest).read_text(encoding="utf-8")); errors=[]
    manifest=BenchmarkManifest(**{**manifest_value,"line_ids":tuple(manifest_value["line_ids"]),"audio_paths":tuple(manifest_value["audio_paths"]),"reference_paths":tuple(manifest_value["reference_paths"])})
    validation=validate_manifest(manifest, require_files=True)
    if not validation["valid"]: errors.append({"code":"MANIFEST_INVALID","details":validation})
    if benchmark.get("manifest_digest") != manifest.digest(): errors.append({"code":"MANIFEST_DIGEST_MISMATCH"})
    if benchmark.get("status") not in {"EXECUTED", "VERIFIED"}: errors.append({"code":"BENCHMARK_NOT_EXECUTED","status":benchmark.get("status")})
    if int(benchmark.get("blocked",1)) or int(benchmark.get("failed",1)): errors.append({"code":"BENCHMARK_HAS_NONPASS_LINES"})
    if not benchmark.get("real_audio"): errors.append({"code":"BENCHMARK_NOT_REAL_AUDIO"})
    rows=(benchmark.get("quality") or {}).get("rows", [])
    if len(rows) != len(manifest.line_ids): errors.append({"code":"BENCHMARK_ROW_COUNT_MISMATCH"})
    if any(str(row.get("status")) not in {"PASS","FINAL_PASS"} for row in rows): errors.append({"code":"BENCHMARK_ROW_STATUS_UNVERIFIED"})
    if not second.get("valid") or not second.get("independent_adapter") or not second.get("content_verified"): errors.append({"code":"SECOND_GAME_EVIDENCE_MISSING"})
    if args.expected_commit:
        actual=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
        if actual != args.expected_commit: errors.append({"code":"COMMIT_MISMATCH","expected":args.expected_commit,"actual":actual})
    result={"promotable":not errors,"errors":errors,"benchmark":benchmark.get("manifest_digest"),"second_game":second.get("game_id"),"manifest_validation":validation}; print(json.dumps(result,indent=2)); return 0 if not errors else 2
if __name__=="__main__": raise SystemExit(main())
