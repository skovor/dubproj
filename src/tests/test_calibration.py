from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.calibration import TARGET_FEATURES, FeatureRow, export_draft, load_draft, train_calibrator
from dubbing_pipeline.calibration.lid_features import LID_FEATURES, LIDFeatureRow

def row(i, label, split="calibration"):
    return FeatureRow(f"c{i}", split, f"g{i}", label, {name: (1.0 if label else 0.0) for name in TARGET_FEATURES})

class CalibrationTests(unittest.TestCase):
    def test_hidden_test_is_sealed(self):
        with self.assertRaises(ValueError): train_calibrator([row(1, 1), row(2, 0, "hidden_test")], kind="target", features=TARGET_FEATURES, dataset_sha256="x")
    def test_draft_roundtrip_is_json(self):
        artifact = train_calibrator([row(1, 1), row(2, 0), row(3, 1), row(4, 0)], kind="target", features=TARGET_FEATURES, dataset_sha256="x")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draft.json"; digest = export_draft(artifact, path); self.assertEqual(len(digest), 64); loaded = load_draft(path); self.assertEqual(loaded["features"], list(TARGET_FEATURES)); self.assertEqual(loaded["status"], "DRAFT")
    def test_single_class_rejected(self):
        with self.assertRaises(ValueError): train_calibrator([row(1, 1), row(2, 1)], kind="target", features=TARGET_FEATURES, dataset_sha256="x")
    def test_lid_has_separate_row_schema_and_artifact_schema(self):
        values = {name: (0.9 if name.endswith("probability") else 0.5) for name in LID_FEATURES}
        rows = [LIDFeatureRow("a", "calibration", "g-a", 1, values), LIDFeatureRow("b", "calibration", "g-b", 0, values)]
        artifact = train_calibrator(rows, kind="lid", features=LID_FEATURES, dataset_sha256="x")
        self.assertEqual(artifact.feature_schema_version, "lid-fusion-v1")
        with self.assertRaises(ValueError):
            LIDFeatureRow("c", "calibration", "g-c", 1, {"lid_source_probability": 0.9})

if __name__ == "__main__": unittest.main()
