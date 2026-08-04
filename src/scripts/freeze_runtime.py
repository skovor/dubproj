#!/usr/bin/env python3
"""Freeze host/dependency/model identity into reproducibility lockfiles.

The script never downloads a model.  A production lock is complete only when
the caller supplies a concrete revision and at least one model file (or an
already computed SHA-256) for every model in ``--models-manifest``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing_pipeline.hashing import canonical_json
from dubbing_pipeline.runtime_lock import collect_runtime_lock, model_file_entry


def _model_digest(files: list[dict]) -> str | None:
    if not files:
        return None
    # File locations are provenance only. The model identity must remain
    # stable when the same bytes move to another machine.
    semantic = [{"bytes": item.get("bytes"), "sha256": item.get("sha256")} for item in files]
    return hashlib.sha256(canonical_json(sorted(semantic, key=lambda item: str(item.get("sha256", ""))))).hexdigest()


def _read_models(path: Path | None, args: argparse.Namespace) -> list[dict]:
    if path is not None:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("--models-manifest must contain a JSON array")
        return [dict(item) for item in value]
    files = [model_file_entry(item) for item in args.model_file]
    return [{
        "model_id": args.model_id,
        "revision": args.model_revision,
        "sha256": args.model_sha256 or _model_digest(files),
        "language": args.language,
        "sample_rate": args.sample_rate,
        "backend": args.backend,
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
    parser.add_argument("--backend-version")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = collect_runtime_lock(omnivoice_version=args.backend_version or "not-installed")
    models = _read_models(args.models_manifest, args)
    model_status = "COMPLETE" if all(item.get("revision") and item.get("sha256") and item.get("backend_version") for item in models) else "UNPROVISIONED"
    model_lock = {
        "schema": "generic-dubbing-model-lock-v1",
        "lock_version": 1,
        "status": model_status,
        "generated_by": "scripts/freeze_runtime.py",
        "models": models,
    }
    runtime_path = out_dir / "runtime.lock.json"
    models_path = out_dir / "models.lock.json"
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    models_path.write_text(json.dumps(model_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_lock": str(runtime_path), "models_lock": str(models_path), "status": model_status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
