"""Train DRAFT calibrators from explicitly human-labelled feature JSONL."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.calibration import FINAL_ANCHOR_FEATURES, TARGET_FEATURES, FeatureRow, export_draft, train_calibrator

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("features"); parser.add_argument("out_dir"); args = parser.parse_args(); path = Path(args.features); raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset_sha = hashlib.sha256(path.read_bytes()).hexdigest(); rows = [FeatureRow(str(row["clip_id"]), str(row["split"]), str(row["split_group"]), int(row["label"]), dict(row["features"]), str(row.get("performance_mode", "NEUTRAL")), row.get("metadata")) for row in raw]
    out = Path(args.out_dir); target = train_calibrator(rows, kind="target", features=TARGET_FEATURES, dataset_sha256=dataset_sha); anchor = train_calibrator(rows, kind="final_anchor", features=FINAL_ANCHOR_FEATURES, dataset_sha256=dataset_sha)
    print(json.dumps({"target": export_draft(target, out / "target-calibrator.draft.json"), "final_anchor": export_draft(anchor, out / "final-anchor-calibrator.draft.json")}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
