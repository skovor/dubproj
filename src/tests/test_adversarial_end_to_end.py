from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dubbing_pipeline.benchmark import BenchmarkManifest, validate_manifest
from dubbing_pipeline.calibration.lid_features import features as lid_features
from dubbing_pipeline.fmv_selector import select_local_scene
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore, HumanLabel
from dubbing_pipeline.performance import PerformanceMode, classify_performance
from dubbing_pipeline.qa_v2 import fold
from dubbing_pipeline.repair import FailureCause, apply_repair
from dubbing_pipeline.repair_planner import plan_repairs
from dubbing_pipeline.attempts import AttemptStore
from dubbing_pipeline.scene_qa import audit_scene_windows


class AdversarialEndToEndTests(unittest.TestCase):
    def test_german_and_lid_contracts_do_not_alias(self):
        self.assertNotEqual(fold("schön"), fold("schon"))
        value = lid_features({"ctc_target_raw_score": 2.0, "ctc_target_calibrated_probability": .2})
        self.assertEqual(value["ctc_target_raw_score"], 2.0)
        self.assertEqual(value["ctc_target_calibrated_probability"], .2)
        self.assertNotIn("ctc_target_probability", value)

    def test_hidden_seal_blocks_post_seal_label_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            store.add_clip(ClipRecord("h", "a" * 64, "scene", "line", "take", "speaker", "Hallo", split="hidden_test"))
            store.seal_hidden_test("operator")
            with self.assertRaises(ValueError):
                store.save_label(HumanLabel("h", "reviewer", "CORRECT_NEUTRAL"))
            self.assertTrue(store.verify_hidden_seal())
            store.close()

    def test_uncertain_repair_cannot_consume_tts_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AttemptStore(Path(tmp) / "attempts.sqlite")
            called = []
            outcome = apply_repair(plan_repairs(FailureCause.LANGUAGE_LEAK_SUSPECTED)[0], line_id="l", input_audio_sha256="a" * 64, reference_sha256=None, store=store, executor=lambda _: called.append(1))
            self.assertEqual(outcome.status, "HOLD_NO_TTS")
            self.assertEqual(called, [])
            store.close()

    def test_fmv_window_failure_is_local(self):
        audio = np.zeros(3000, dtype="float32"); audio[:1000] = .1; audio[2000:3000] = 1.0
        evidence = audit_scene_windows(audio, 1000, [{"line_id": "A", "start": 0, "end": 1000}, {"line_id": "B", "start": 2000, "end": 3000}])
        self.assertEqual(evidence["failed_line_ids"], ["B"])
        class Line:
            def __init__(self, line_id): self.id = line_id
        options = {"A": [{"candidate": type("C", (), {"candidate_id": "A1"})(), "eligible": True}], "B": [{"candidate": type("C", (), {"candidate_id": "B1"})(), "eligible": True}]}
        result = select_local_scene([Line("A"), Line("B")], options, [], max_candidates_per_line=1, max_iterations=1, mount_line=lambda value, line, option: value + [line.id], audit_scene=lambda value, index: (False, type("Audit", (), {"diagnostics": {"failed_line_ids": ["B"]}})()))
        self.assertFalse(result.passed)
        self.assertEqual(result.matrix, [])

    def test_unresolved_performance_is_not_neutral(self):
        self.assertEqual(classify_performance(metadata={}).mode, PerformanceMode.UNRESOLVED)

    def test_declared_real_audio_does_not_override_manifest_derivation(self):
        manifest = BenchmarkManifest("b", ("l1",), ("not_audio",), ("not_ref",), "models", "runtime", None, "config", "commit", "LINE_SEPARATED", True)
        validation = validate_manifest(manifest, require_files=False)
        self.assertFalse(validation["real_audio"])


if __name__ == "__main__":
    unittest.main()
