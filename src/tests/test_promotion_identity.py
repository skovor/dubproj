from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dubbing_pipeline.calibration.identity import ModelIdentityError, resolve_alignment_identity
from dubbing_pipeline.config import CalibrationConfigurationError, PipelineConfig, QAConfig
from dubbing_pipeline.orchestration_v2 import run_scene_v2


class PromotionIdentityTests(unittest.TestCase):
    def test_production_preflight_blocks_before_runtime_or_tts(self):
        config = PipelineConfig(lab_mode=False, qa=QAConfig(calibration_authority=True))
        with self.assertRaises(CalibrationConfigurationError) as raised:
            run_scene_v2(None, config, runtime=None)
        self.assertEqual(raised.exception.status, "BLOCKED_EXPECTED_CODE_COMMIT_REQUIRED")
        config.qa.expected_calibration_code_commit = "not-a-sha"
        with self.assertRaises(CalibrationConfigurationError) as raised:
            run_scene_v2(None, config, runtime=None)
        self.assertEqual(raised.exception.status, "BLOCKED_CODE_COMMIT_MISMATCH")
        config.qa.expected_calibration_code_commit = "a" * 40
        with self.assertRaises(CalibrationConfigurationError) as raised:
            run_scene_v2(None, config, runtime=None)
        self.assertEqual(raised.exception.status, "BLOCKED_CODE_COMMIT_MISMATCH")

    def test_alignment_identity_requires_one_explicit_role_and_lock_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"role": "alignment", "backend_id": "ctc", "model_id": "m", "revision": "r1", "source_language": "en", "target_language": "de", "feature_schema_version": "char-alignment-v3", "performance_modes": ["NEUTRAL", "FAST"]}
            config = root / "config.json"; lock = root / "models.json"
            config.write_text(json.dumps({"models": [identity]}), encoding="utf-8")
            lock.write_text(json.dumps({"models": [identity]}), encoding="utf-8")
            self.assertEqual(resolve_alignment_identity(config, lock)["model_revision"], "r1")
            config.write_text(json.dumps({"models": [identity, {**identity, "model_id": "other"}]}), encoding="utf-8")
            with self.assertRaises(ModelIdentityError):
                resolve_alignment_identity(config, lock)
            config.write_text(json.dumps({"models": [identity]}), encoding="utf-8")
            lock.write_text(json.dumps({"models": [{**identity, "revision": "r2"}]}), encoding="utf-8")
            with self.assertRaises(ModelIdentityError):
                resolve_alignment_identity(config, lock)


if __name__ == "__main__":
    unittest.main()
