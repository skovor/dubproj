from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from dubbing_pipeline.performance import PerformanceContract, PerformanceMode, classify_performance, extract_basic_features
from dubbing_pipeline.performance_policy import policy_for
from dubbing_pipeline.qa_v2 import evaluate_candidate_v2

class PerformanceTests(unittest.TestCase):
    def test_explicit_mode_wins(self):
        result=classify_performance(metadata={"performance_mode":"CRYING_SPEECH"},rms_dbfs=-10); self.assertEqual(result.mode,PerformanceMode.CRYING_SPEECH); self.assertEqual(result.confidence,1.0)
    def test_effort_does_not_require_final_word(self):
        self.assertFalse(policy_for(PerformanceMode.EFFORT).require_final_word); self.assertFalse(policy_for(PerformanceMode.EFFORT).require_content)
    def test_neutral_requires_semantics(self):
        self.assertTrue(policy_for("NEUTRAL").require_content); self.assertTrue(policy_for("NEUTRAL").require_final_word)
    def test_features_deterministic(self):
        values=extract_basic_features([-.5,0,.5,0],2); self.assertAlmostEqual(values["duration_seconds"],2.0); self.assertIn("rms_dbfs",values)
    def test_unknown_mode_is_not_silently_neutral(self):
        result = classify_performance(metadata={})
        self.assertEqual(result.mode, PerformanceMode.UNRESOLVED)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(policy_for(PerformanceMode.UNRESOLVED).notes, "performance unresolved; retain lexical gates and block promotion")

    def test_duration_policy_is_executed_as_a_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "line.wav"
            sf.write(path, np.ones(24000, dtype="float32") * .05, 24000)
            result = evaluate_candidate_v2(str(path), expected_text="", target_sample_rate=24000, reference_end=.5, hard_gates=["performance_duration"], require_asr=False, performance_mode="FAST", performance_max_duration_error_ms=10.0)
            self.assertEqual(result.gates["performance_duration"].status.value, "FAIL")

if __name__=="__main__": unittest.main()
