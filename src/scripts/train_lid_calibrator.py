"""Train an independent LID draft using the same safe JSON trainer."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.calibration import FeatureRow, export_draft, train_calibrator
from dubbing_pipeline.calibration.lid_features import LID_FEATURES

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("features"); parser.add_argument("output"); args=parser.parse_args(); path=Path(args.features); rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]; parsed=[FeatureRow(str(x["clip_id"]), str(x["split"]), str(x["split_group"]), int(x["label"]), dict(x["features"]), str(x.get("performance_mode","NEUTRAL"))) for x in rows]; artifact=train_calibrator(parsed, kind="lid", features=LID_FEATURES, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest()); print(export_draft(artifact,args.output)); return 0
if __name__ == "__main__": raise SystemExit(main())
