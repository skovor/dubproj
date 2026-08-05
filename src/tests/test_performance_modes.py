from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from dubbing_pipeline.performance import PerformanceContract, PerformanceMode, classify_performance, extract_basic_features, measure_audio
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

    def test_short_window_does_not_infer_fast(self):
        self.assertEqual(classify_performance(metadata={}, duration_seconds=.20).mode, PerformanceMode.UNRESOLVED)

    def test_measured_audio_is_separate_from_declared_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voiced.wav"
            samples = .2 * np.sin(2 * np.pi * 180 * np.arange(24000) / 24000).astype("float32")
            sf.write(path, samples, 24000)
            evidence = measure_audio(path, declared=classify_performance(metadata={"performance_mode": "FAST"}))
            self.assertEqual(evidence.mode, PerformanceMode.FAST)
            self.assertEqual(evidence.declared_mode, "FAST")
            self.assertTrue(evidence.measured)
            self.assertGreater(evidence.pitch_hz or 0.0, 100.0)
            self.assertGreater(evidence.speech_ratio or 0.0, .9)

    def test_noise_does_not_become_effort_without_explicit_metadata(self):
        rng = np.random.default_rng(12)
        result = classify_performance(metadata={}, rms_dbfs=-20.0, pitch_hz=None, speech_ratio=1.0, duration_seconds=1.0)
        self.assertEqual(result.mode, PerformanceMode.UNRESOLVED)

    def test_duration_policy_is_executed_as_a_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "line.wav"
            sf.write(path, np.ones(24000, dtype="float32") * .05, 24000)
            result = evaluate_candidate_v2(str(path), expected_text="", target_sample_rate=24000, reference_end=.5, hard_gates=["performance_duration"], require_asr=False, performance_mode="FAST", performance_max_duration_error_ms=10.0)
            self.assertEqual(result.gates["performance_duration"].status.value, "FAIL")

if __name__=="__main__": unittest.main()
