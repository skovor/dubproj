from __future__ import annotations
import unittest
from dubbing_pipeline.performance import PerformanceContract, PerformanceMode, classify_performance, extract_basic_features
from dubbing_pipeline.performance_policy import policy_for

class PerformanceTests(unittest.TestCase):
    def test_explicit_mode_wins(self):
        result=classify_performance(metadata={"performance_mode":"CRYING_SPEECH"},rms_dbfs=-10); self.assertEqual(result.mode,PerformanceMode.CRYING_SPEECH); self.assertEqual(result.confidence,1.0)
    def test_effort_does_not_require_final_word(self):
        self.assertFalse(policy_for(PerformanceMode.EFFORT).require_final_word); self.assertFalse(policy_for(PerformanceMode.EFFORT).require_content)
    def test_neutral_requires_semantics(self):
        self.assertTrue(policy_for("NEUTRAL").require_content); self.assertTrue(policy_for("NEUTRAL").require_final_word)
    def test_features_deterministic(self):
        values=extract_basic_features([-.5,0,.5,0],2); self.assertAlmostEqual(values["duration_seconds"],2.0); self.assertIn("rms_dbfs",values)

if __name__=="__main__": unittest.main()
