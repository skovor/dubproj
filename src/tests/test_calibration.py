from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.calibration import TARGET_FEATURES, FeatureRow, export_draft, load_draft, train_calibrator

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

if __name__ == "__main__": unittest.main()
