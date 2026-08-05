from __future__ import annotations

import unittest
import numpy as np

from dubbing_pipeline.scene_qa import audit_scene_windows


class SceneWindowQATests(unittest.TestCase):
    def test_silent_line_is_attributed_to_its_window(self):
        audio = np.zeros(2000, dtype="float32")
        audio[0:500] = .1
        result = audit_scene_windows(audio, 1000, [{"line_id": "A", "start": 0, "end": 500}, {"line_id": "B", "start": 500, "end": 1000}])
        self.assertEqual(result["failed_line_ids"], ["B"])
        self.assertEqual(result["line_gate_results"][0]["failed_gates"], [])

    def test_clipping_is_attributed_without_replacing_other_line(self):
        audio = np.zeros(2000, dtype="float32")
        audio[0:500] = .1
        audio[1000:1500] = 1.0
        result = audit_scene_windows(audio, 1000, [{"line_id": "A", "start": 0, "end": 500}, {"line_id": "B", "start": 1000, "end": 1500}])
        self.assertEqual(result["failed_line_ids"], ["B"])
        self.assertIn("line_clipping", result["line_gate_results"][1]["failed_gates"])

    def test_dialogue_channel_ignores_music_on_channel_zero(self):
        audio = np.zeros((1000, 2), dtype="float32")
        audio[:, 0] = .2  # music/bed
        result = audit_scene_windows(audio, 1000, [{"line_id": "A", "start": 0, "end": 1000}], dialogue_channel=1)
        self.assertEqual(result["dialogue_channel"], 1)
        self.assertEqual(result["failed_line_ids"], ["A"])
        self.assertIn("line_activity", result["line_gate_results"][0]["failed_gates"])

    def test_mounted_delta_is_required_only_for_generated_line(self):
        source = np.zeros((1000, 2), dtype="float32"); source[:, 0] = .2
        mounted = source.copy()
        result = audit_scene_windows(mounted, 1000, [{"line_id": "A", "start": 0, "end": 1000}], dialogue_channel=1, source_audio=source, require_mounted_delta_line_ids={"A"})
        self.assertEqual(result["failed_line_ids"], ["A"])
        self.assertIn("mounted_source_delta", result["line_gate_results"][0]["failed_gates"])
        mounted[:, 1] = .3
        passed = audit_scene_windows(mounted, 1000, [{"line_id": "A", "start": 0, "end": 1000}], dialogue_channel=1, source_audio=source, require_mounted_delta_line_ids={"A"})
        self.assertEqual(passed["failed_line_ids"], [])

    def test_invalid_dialogue_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_scene_windows(np.zeros((10, 1)), 1000, [{"line_id": "A", "start": 0, "end": 10}], dialogue_channel=1)


if __name__ == "__main__":
    unittest.main()
