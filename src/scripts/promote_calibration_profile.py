"""Promote only a separately reviewed draft; missing artifacts fail closed."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing_pipeline.calibration import load_draft
from dubbing_pipeline.calibration.promote import promote_profile
from dubbing_pipeline.calibration.validate import ValidationReport

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact"); parser.add_argument("validation_report"); parser.add_argument("hidden_report"); parser.add_argument("--manifest", required=True); parser.add_argument("--labels", required=True); parser.add_argument("--splits", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); artifact = load_draft(args.artifact)
    def report(path):
        value = json.loads(Path(path).read_text(encoding="utf-8")); return ValidationReport(value["split"], value["count"], value["brier_score"], value["expected_calibration_error"], value["false_pass_count"], value["false_fail_count"], tuple(value.get("predictions", [])), value.get("run_id", ""))
    identity = {"backend_id": "unknown", "model_id": "unknown", "model_revision": "unknown", "feature_schema_version": "char-alignment-v2", "target_language": "de", "source_language": "en", "performance_modes": ["NEUTRAL"]}; thresholds = {"target_pass_probability": .8, "target_failure_probability": .2, "final_anchor_pass_probability": .8, "source_lid_probability": .8}; provenance = {"code_commit": "unknown", "runtime_lock_sha256": "0" * 64, "models_lock_sha256": "0" * 64}; artifact["artifact_path"] = str(Path(args.artifact)); import hashlib; artifact["artifact_sha256"] = hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest()
    profile = promote_profile(profile_id="manual-profile", target_artifact=artifact, validation=report(args.validation_report), hidden=report(args.hidden_report), dataset_files={"manifest_sha256": args.manifest, "labels_sha256": args.labels, "split_manifest_sha256": args.splits}, identity=identity, thresholds=thresholds, provenance=provenance, output=args.output); print(json.dumps(profile, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
