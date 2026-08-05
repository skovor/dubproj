"""Run a benchmark from a manifest JSON; no fake real-audio claim is inferred."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import importlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.benchmark import BenchmarkManifest, run_benchmark
from dubbing_pipeline.attestation import sign_attestation, subject_digest

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("output"); p.add_argument("--runner", help="dotted module:function that invokes run_scene_v2"); p.add_argument("--attestation-private-key", help="base64 Ed25519 private key; omit to remain unsigned"); p.add_argument("--attestation-key-id", default="", help="trusted key identifier")
    args=p.parse_args(); value=json.loads(Path(args.manifest).read_text(encoding="utf-8")); manifest=BenchmarkManifest(**{**value,"line_ids":tuple(value["line_ids"]),"audio_paths":tuple(value["audio_paths"]),"reference_paths":tuple(value["reference_paths"])})
    validation=__import__("dubbing_pipeline.benchmark", fromlist=["validate_manifest"]).validate_manifest(manifest, require_files=True)
    if not validation["valid"]:
        payload={"schema":"benchmark-result-v2","status":"BLOCKED","reason":"manifest_validation_failed","validation":validation,"manifest_digest":manifest.digest()}
    elif not args.runner:
        payload={"schema":"benchmark-result-v2","status":"BLOCKED","reason":"real_runner_required","manifest_digest":manifest.digest()}
    else:
        module_name, separator, function_name=args.runner.partition(":")
        if not separator: raise SystemExit("--runner must be module:function")
        runner=getattr(importlib.import_module(module_name), function_name)
        payload=run_benchmark(manifest, runner, require_files=True, require_trusted_runner=True).to_dict()
        payload={"schema":"benchmark-result-v2","status":"EXECUTED",**payload}
    subject = {"schema": "benchmark-attestation-subject-v1", "manifest_digest": payload.get("manifest_digest"), "code_commit": manifest.commit, "benchmark_payload_sha256": subject_digest(payload)}
    if args.attestation_private_key:
        if not args.attestation_key_id: raise SystemExit("--attestation-key-id is required with --attestation-private-key")
        payload["attestation"] = sign_attestation(subject, Path(args.attestation_private_key).read_text(encoding="utf-8").strip(), key_id=args.attestation_key_id)
    else:
        payload["attestation"] = {"schema": "benchmark-attestation-v1", "verified": False, "reason": "UNSIGNED"}
    payload["evidence_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in payload.items() if key != "evidence_sha256"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2)); return 0 if payload.get("status") == "EXECUTED" else 2
if __name__=="__main__": raise SystemExit(main())
