"""Small CLI that is safe to run without audio/OmniVoice dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .mapping import validate_manifest
from .review import build_review_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dubbing-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--config", required=True); validate.add_argument("--manifest", required=True)
    review = sub.add_parser("review-bundle"); review.add_argument("--config", required=True); review.add_argument("--manifest", required=True); review.add_argument("--out", required=True); review.add_argument("--include-text", action="store_true")
    args = parser.parse_args(argv)
    config = PipelineConfig.load(args.config)
    if args.command == "validate":
        print(json.dumps({"config_project_root": str(config.project_root), **validate_manifest(args.manifest)}, ensure_ascii=False, indent=2)); return 0
    build_review_bundle(args.manifest, output=args.out, include_text=args.include_text)
    print(json.dumps({"review_bundle": str(Path(args.out).resolve())}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
