from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
from dubbing_pipeline.calibration import TARGET_FEATURES, FeatureRow, train_calibrator
from dubbing_pipeline.calibration.lid_features import LID_FEATURES
from dubbing_pipeline.calibration.promote import PromotionError, promote_profile
from dubbing_pipeline.calibration.validate import ValidationReport, evaluate

def row(i, label, split): return FeatureRow(f"c{i}", split, f"g{i}", label, {name: (3.0 if label else -3.0) for name in TARGET_FEATURES})

class PromotionTests(unittest.TestCase):
    def test_hidden_false_pass_blocks(self):
        artifact = train_calibrator([row(1, 1, "calibration"), row(2, 0, "calibration")], kind="target", features=TARGET_FEATURES, dataset_sha256="x")
        final_artifact = {**artifact.to_dict(), "feature_schema_version": "final-anchor-v1", "features": ["final_coverage"], "coefficients": [1.0]}
        hidden = ValidationReport("hidden_test", 1, .5, .5, 1, 0, ({"clip_id":"c3","label":0,"probability":.9},), "hidden-1")
        with self.assertRaises(PromotionError):
            promote_profile(profile_id="x", target_artifact={**artifact.to_dict(), "artifact_path": "x", "artifact_sha256": "a" * 64}, final_anchor_artifact={**final_artifact, "artifact_path": "x", "artifact_sha256": "b" * 64}, validation=evaluate(artifact, [row(4, 1, "validation")], split="validation"), hidden=hidden, dataset_files={}, identity={"backend_id":"b","model_id":"m","model_revision":"r","feature_schema_version":"char-alignment-v2","target_language":"de","source_language":"en"}, thresholds={"target_pass_probability":.8,"target_failure_probability":.2,"final_anchor_pass_probability":.8,"source_lid_probability":.8}, provenance={"code_commit":"c","runtime_lock_sha256":"0"*64,"models_lock_sha256":"0"*64}, output="x")
    def test_dataset_hashes_are_recomputed(self):
        artifact = train_calibrator([row(1, 1, "calibration"), row(2, 0, "calibration")], kind="target", features=TARGET_FEATURES, dataset_sha256="x")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); artifact_file = root / "artifact.json"; artifact_file.write_text(json.dumps(artifact.to_dict()), encoding="utf-8"); sha = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
            final_file = root / "final-anchor.json"; final_payload = {**artifact.to_dict(), "feature_schema_version": "final-anchor-v1", "features": ["final_coverage"], "coefficients": [1.0], "normalization": [{"mean": 0.0, "scale": 1.0}]}; final_file.write_text(json.dumps(final_payload), encoding="utf-8"); final_sha = hashlib.sha256(final_file.read_bytes()).hexdigest(); files = {}
            lid_file = root / "lid.json"; lid_payload = {**artifact.to_dict(), "feature_schema_version": "lid-fusion-v1", "features": list(LID_FEATURES), "coefficients": [0.0] * len(LID_FEATURES), "normalization": [{"mean": 0.0, "scale": 1.0} for _ in LID_FEATURES]}; lid_file.write_text(json.dumps(lid_payload), encoding="utf-8"); lid_sha = hashlib.sha256(lid_file.read_bytes()).hexdigest()
            for name in ("manifest_sha256", "labels_sha256", "split_manifest_sha256"): files[name] = root / f"{name}.json"; files[name].write_text(name, encoding="utf-8")
            validation_rows = [row(3, 1, "validation"), row(5, 0, "validation"), row(6, 1, "validation"), row(7, 0, "validation")]
            hidden_rows = [row(4, 1, "hidden_test"), row(8, 0, "hidden_test"), row(9, 1, "hidden_test"), row(10, 0, "hidden_test")]
            valid = evaluate(artifact, validation_rows, split="validation")
            hidden = evaluate(artifact, hidden_rows, split="hidden_test", run_id="h1")
            profile = promote_profile(profile_id="x", target_artifact={**artifact.to_dict(), "artifact_path": str(artifact_file), "artifact_sha256": sha}, final_anchor_artifact={**final_payload, "artifact_path": str(final_file), "artifact_sha256": final_sha}, lid_artifact={**lid_payload, "artifact_path": str(lid_file), "artifact_sha256": lid_sha}, validation=valid, hidden=hidden, validation_rows=validation_rows, hidden_rows=hidden_rows, dataset_files=files, identity={"backend_id":"b","model_id":"m","model_revision":"r","feature_schema_version":"char-alignment-v2","target_language":"de","source_language":"en","performance_modes":["NEUTRAL"]}, thresholds={"target_pass_probability":.8,"target_failure_probability":.2,"final_anchor_pass_probability":.8,"source_lid_probability":.8}, provenance={"code_commit":"a"*40,"runtime_lock_sha256":"1"*64,"models_lock_sha256":"2"*64}, output=root/"profile.json")
            self.assertEqual(profile["status"], "VALIDATED"); self.assertEqual(profile["dataset"]["manifest_sha256"], hashlib.sha256(b"manifest_sha256").hexdigest())

if __name__ == "__main__": unittest.main()
