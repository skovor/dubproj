"""Validate exported human gold-set manifests and labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.goldset import ClipRecord, HumanLabel, validate_goldset

def _jsonl(path: Path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("directory"); parser.add_argument("--allow-single-review", action="store_true"); args = parser.parse_args(); root = Path(args.directory)
    clips = [ClipRecord(**row) for row in _jsonl(root / "manifest.jsonl")]; labels = [HumanLabel(**{**row, "affected_tokens": tuple(row.get("affected_tokens") or [])}) for row in _jsonl(root / "labels.jsonl")]
    seal = root / "hidden_seal.json"
    result = validate_goldset(clips, labels, require_double_review=not args.allow_single_review, hidden_sealed=seal.is_file()); print(json.dumps(result, indent=2, ensure_ascii=False)); return 0 if result["valid"] else 2
if __name__ == "__main__": raise SystemExit(main())
