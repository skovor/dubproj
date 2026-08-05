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


if __name__ == "__main__":
    unittest.main()
