"""V2 command line surface with safe preflight/dry-run defaults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .contracts.manifest import normalize_manifest
from .lab import SandboxLayout
from .mapping import validate_manifest
from .review import build_review_bundle
from .scheduler import PHASES
from .state import StateStore


def _config(path: str) -> PipelineConfig:
    return PipelineConfig.load(path)


def _preflight(config: PipelineConfig, manifest: str, *, strict_runtime: bool = False) -> dict:
    scenes = normalize_manifest(json.loads(Path(manifest).read_text(encoding="utf-8")))
    strict = bool(strict_runtime or not config.lab_mode)
    reproducibility = config.reproducibility_report(strict=strict)
    result = {"schema": "dubbing-pipeline-preflight-v2", "lab_mode": bool(config.lab_mode), "strict_runtime": strict, "reproducibility": reproducibility, "manifest": str(Path(manifest).resolve()), "scene_count": len(scenes), "line_count": sum(len(item["lines"]) for item in scenes), "phases": list(PHASES), "roots": {"project_root": str(config.project_root), "output_root": str(config.output_root), "cache_root": str(config.cache_root)}}
    if strict and reproducibility["status"] != "PASS":
        raise ValueError("runtime reproducibility preflight blocked: " + "; ".join(reproducibility["errors"]))
    if config.lab_mode:
        if config.sandbox_root is None:
            raise ValueError("lab_mode=true requires sandbox_root in V2 config")
        layout = SandboxLayout.create(config.sandbox_root); layout.ensure_safe(lab_mode=True)
        result["sandbox_root"] = str(layout.root)
        result["runtime_root"] = str(layout.runtime_root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dubbing-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--config", required=True); validate.add_argument("--manifest", required=True)
    preflight = sub.add_parser("preflight"); preflight.add_argument("--config", required=True); preflight.add_argument("--manifest", required=True); preflight.add_argument("--strict-runtime", action="store_true", help="fail unless runtime and model lockfiles are complete")
    plan = sub.add_parser("plan"); plan.add_argument("--config", required=True); plan.add_argument("--manifest", required=True); plan.add_argument("--out")
    review = sub.add_parser("review-bundle"); review.add_argument("--config", required=True); review.add_argument("--manifest", required=True); review.add_argument("--out", required=True); review.add_argument("--include-text", action="store_true")
    status = sub.add_parser("status"); status.add_argument("--run-root", required=True); status.add_argument("--run-id", required=True)
    sub.add_parser("resume").add_argument("--run-root", required=True); sub.choices["resume"].add_argument("--run-id", required=True)
    for name in ("generate", "qa", "mount", "package", "deploy", "smoke"):
        command = sub.add_parser(name); command.add_argument("--config", required=True); command.add_argument("--manifest", required=True); command.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    if args.command in {"status", "resume"}:
        state = StateStore(args.run_root, args.run_id).load()
        print(json.dumps({"run_id": args.run_id, "state": state.to_dict() if state else None}, ensure_ascii=False, indent=2)); return 0
    config = _config(args.config)
    if args.command == "validate":
        print(json.dumps({"config_project_root": str(config.project_root), **validate_manifest(args.manifest)}, ensure_ascii=False, indent=2)); return 0
    if args.command == "preflight":
        print(json.dumps(_preflight(config, args.manifest, strict_runtime=bool(args.strict_runtime)), ensure_ascii=False, indent=2)); return 0
    if args.command == "plan":
        value = _preflight(config, args.manifest); value["status"] = "PLANNED"; value["next"] = "generate_initial_cohort"
        if args.out: Path(args.out).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value, ensure_ascii=False, indent=2)); return 0
    if args.command == "review-bundle":
        build_review_bundle(args.manifest, output=args.out, include_text=args.include_text)
        print(json.dumps({"review_bundle": str(Path(args.out).resolve())}, ensure_ascii=False)); return 0
    # Heavy phases are intentionally explicit. They cannot accidentally run
    # against a production runtime from this generic CLI.
    print(json.dumps({"command": args.command, "status": "DRY_RUN_ONLY", "reason": "use the V2 scheduler API with a declared sandbox and adapter"}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
