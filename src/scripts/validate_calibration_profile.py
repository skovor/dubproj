"""Evaluate a draft calibrator on an explicit validation/hidden feature JSONL."""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.calibration import FeatureRow, load_draft
from dubbing_pipeline.calibration.validate import evaluate

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact"); parser.add_argument("features"); parser.add_argument("--split", choices=["validation", "hidden_test"], required=True); args = parser.parse_args(); artifact = load_draft(args.artifact); rows = [FeatureRow(str(x["clip_id"]), str(x["split"]), str(x["split_group"]), int(x["label"]), dict(x["features"]), str(x.get("performance_mode", "NEUTRAL"))) for x in (json.loads(line) for line in Path(args.features).read_text(encoding="utf-8").splitlines() if line.strip())]; report = evaluate(artifact, rows, split=args.split, run_id=f"hidden-{uuid.uuid4().hex}" if args.split == "hidden_test" else "validation"); print(json.dumps(report.to_dict(), indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
