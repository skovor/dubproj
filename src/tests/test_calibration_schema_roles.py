from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dubbing_pipeline.qa_v2 import _CALIBRATOR_FEATURES, _FINAL_ANCHOR_FEATURES, _LID_FEATURES, calibration_profile_status


class CalibrationSchemaRoleTests(unittest.TestCase):
    def _profile(self, root: Path) -> dict:
        def artifact(schema: str, name: str) -> dict:
            path = root / name
            features = list(_FINAL_ANCHOR_FEATURES if schema == "final-anchor-v1" else (_LID_FEATURES if schema == "lid-fusion-v1" else _CALIBRATOR_FEATURES))
            path.write_text(json.dumps({"schema": "platt-calibrator-v1", "feature_schema_version": schema, "normalization_version": "alignment-text-normalization-v1", "features": features, "coefficients": [1.0] * len(features), "intercept": 0.0, "normalization": [{"mean": 0.0, "scale": 1.0} for _ in features]}), encoding="utf-8")
            return {
                "type": "platt", "engine": "builtin", "format": "json",
                "feature_schema_version": schema, "normalization_version": "alignment-text-normalization-v1",
                "artifact_path": str(path), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "feature_names": features,
            }
        return {
            "schema": "generic-dubbing-alignment-calibration-profile-v2", "status": "VALIDATED", "authority": True,
            "profile_id": "schema-test", "identity": {"backend_id": "b", "model_id": "m", "model_revision": "r", "feature_schema_version": "char-alignment-v2", "target_language": "de", "source_language": "en", "performance_modes": ["NEUTRAL"]},
            "thresholds": {"target_pass_probability": .8, "target_failure_probability": .2, "final_anchor_pass_probability": .8, "source_lid_probability": .8},
            "calibrators": {"target": artifact("char-alignment-v2", "target.json"), "final_anchor": artifact("final-anchor-v1", "anchor.json"), "lid": artifact("lid-fusion-v1", "lid.json")},
            "dataset": {"manifest_sha256": "0" * 64, "labels_sha256": "1" * 64, "split_manifest_sha256": "2" * 64, "calibration_count": 2, "validation_count": 2, "hidden_test_count": 2},
            "metrics": {"hidden_false_pass_count": 0, "hidden_false_fail_count": 0, "brier_score": .1, "expected_calibration_error": .1},
            "provenance": {"code_commit": "a" * 40, "runtime_lock_sha256": "3" * 64, "models_lock_sha256": "4" * 64, "created_at": "2025-01-01T00:00:00Z"},
        }

    def test_generated_role_profile_validates_schema(self):
        schema = json.loads(Path(__file__).parents[1].joinpath("config/calibration-profile.schema.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._profile(Path(tmp))
            errors = sorted(Draft202012Validator(schema).iter_errors(profile), key=lambda item: item.path)
            self.assertEqual(errors, [])

    def test_lid_is_required_and_language_id_is_rejected(self):
        kwargs = {"authority": True, "backend_id": "b", "model_id": "m", "model_revision": "r", "target_language": "de", "source_language": "en", "performance_mode": "NEUTRAL", "runtime_lock_sha256": "3" * 64, "models_lock_sha256": "4" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._profile(Path(tmp))
            profile["calibrators"].pop("lid")
            self.assertEqual(calibration_profile_status(profile, **kwargs), "BLOCKED_SCHEMA")
            profile = self._profile(Path(tmp))
            profile["calibrators"]["language_id"] = profile["calibrators"].pop("lid")
            self.assertEqual(calibration_profile_status(profile, **kwargs), "BLOCKED_SCHEMA")

    def test_role_schema_mismatch_is_rejected(self):
        kwargs = {"authority": True, "backend_id": "b", "model_id": "m", "model_revision": "r", "target_language": "de", "source_language": "en", "performance_mode": "NEUTRAL", "runtime_lock_sha256": "3" * 64, "models_lock_sha256": "4" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            profile = self._profile(Path(tmp))
            profile["calibrators"]["lid"]["feature_schema_version"] = "final-anchor-v1"
            self.assertEqual(calibration_profile_status(profile, **kwargs), "BLOCKED_CALIBRATOR_SCHEMA")


if __name__ == "__main__":
    unittest.main()
