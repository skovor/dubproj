#!/usr/bin/env python3
"""Freeze host/dependency/model identity into reproducibility lockfiles.

The script never downloads a model.  A production lock is complete only when
the caller supplies a concrete revision and at least one model file (or an
already computed SHA-256) for every model in ``--models-manifest``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing_pipeline.runtime_lock import aggregate_model_sha256, collect_runtime_lock, validate_models_lock, verify_model_files, model_file_entry, runtime_completion_errors


def _model_digest(files: list[dict]) -> str | None:
    if not files:
        return None
    return aggregate_model_sha256(files)


def _read_models(path: Path | None, args: argparse.Namespace) -> list[dict]:
    if path is not None:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("--models-manifest must contain a JSON array")
        return [dict(item) for item in value]
    files = [model_file_entry(item, logical_path=Path(item).name) for item in args.model_file]
    return [{
        "model_id": args.model_id,
        "revision": args.model_revision,
        "sha256": args.model_sha256 or _model_digest(files),
        "language": args.language,
        "sample_rate": args.sample_rate,
        "backend": args.backend,
        "backend_id": args.backend_id,
        "backend_version": args.backend_version,
        "files": files,
    }]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "config"))
    parser.add_argument("--models-manifest", type=Path)
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-sha256")
    parser.add_argument("--model-file", action="append", default=[])
    parser.add_argument("--language", default="de")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--backend", default="omnivoice")
    parser.add_argument("--backend-id", default="omnivoice")
    parser.add_argument("--backend-version")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capabilities-json", type=Path)
    parser.add_argument("--models-root", default="models")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    capabilities = None
    if args.capabilities_json is not None:
        capabilities = json.loads(args.capabilities_json.read_text(encoding="utf-8"))
    runtime = collect_runtime_lock(device=args.device, capabilities=capabilities, omnivoice_version=args.backend_version)
    models = _read_models(args.models_manifest, args)
    model_lock_probe = {"schema": "generic-dubbing-model-lock-v2", "status": "UNPROVISIONED", "models": models, "models_root": args.models_root}
    model_errors = validate_models_lock(model_lock_probe, strict=True)
    model_errors.extend(verify_model_files(model_lock_probe, base_dir=out_dir, strict=True))
    model_status = "COMPLETE" if not model_errors else "UNPROVISIONED"
    model_lock = {
        "schema": "generic-dubbing-model-lock-v2",
        "lock_version": 2,
        "status": model_status,
        "generated_by": "scripts/freeze_runtime.py",
        "models_root": args.models_root,
        "models": models,
    }
    runtime["status"] = "COMPLETE" if not runtime_completion_errors(runtime, required_capabilities=capabilities) else "UNPROVISIONED"
    runtime_path = out_dir / "runtime.lock.json"
    models_path = out_dir / "models.lock.json"
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    models_path.write_text(json.dumps(model_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "COMPLETE" if runtime["status"] == "COMPLETE" and model_status == "COMPLETE" else "UNPROVISIONED"
    print(json.dumps({"runtime_lock": str(runtime_path), "models_lock": str(models_path), "status": status, "runtime_errors": runtime_completion_errors(runtime, required_capabilities=capabilities), "model_errors": model_errors}, ensure_ascii=False, indent=2))
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
