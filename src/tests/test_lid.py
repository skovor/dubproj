from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.lid import LIDPolicy, independent_lid, fuse_language_evidence
from dubbing_pipeline.calibration.lid_features import LID_FEATURES, features

class Backend:
    backend_id="speechbrain-ecapa"; model_id="voxlingua107"; model_revision="r1"
    def detect(self, path, sample_rate=16000): return {"language":"en","probabilities":{"en":.92,"de":.05}}

class LIDTests(unittest.TestCase):
    def test_short_clip_not_applicable(self):
        with tempfile.NamedTemporaryFile() as audio: evidence=independent_lid(Backend(), audio.name, policy=LIDPolicy(), duration_seconds=.1, speech_ratio=1, sample_rate=48000, audio_sha256="a"*64)
        self.assertEqual(evidence.status, "LID_NOT_APPLICABLE")
    def test_independent_backend_and_hash(self):
        with tempfile.NamedTemporaryFile() as audio: evidence=independent_lid(Backend(), audio.name, policy=LIDPolicy(), duration_seconds=1, speech_ratio=.8, sample_rate=48000, audio_sha256="a"*64)
        self.assertEqual(evidence.status, "LID_CONFIDENT"); self.assertEqual(evidence.backend_id, "speechbrain-ecapa"); self.assertEqual(len(evidence.evidence_hash), 64); self.assertEqual(evidence.record["evidence_family"], "AUDIO_LANGUAGE_ID")
    def test_concordance_confirms_leak(self):
        with tempfile.NamedTemporaryFile() as audio: evidence=independent_lid(Backend(), audio.name, policy=LIDPolicy(), duration_seconds=1, speech_ratio=.8, sample_rate=48000, audio_sha256="a"*64)
        result=fuse_language_evidence(whisper_language="en", whisper_probability=.9, lid=evidence, ctc_target_probability=.2, policy=LIDPolicy()); self.assertNotEqual(result["status"], "LANGUAGE_LEAK_CONFIRMED")
    def test_ctc_conflict_is_not_leak(self):
        with tempfile.NamedTemporaryFile() as audio: evidence=independent_lid(Backend(), audio.name, policy=LIDPolicy(), duration_seconds=1, speech_ratio=.8, sample_rate=48000, audio_sha256="a"*64)
        result=fuse_language_evidence(whisper_language="en", whisper_probability=.9, lid=evidence, ctc_target_probability=.9, policy=LIDPolicy()); self.assertNotEqual(result["status"], "EVIDENCE_CONFLICT")
        raw_only=fuse_language_evidence(whisper_language="en", whisper_probability=.9, lid=evidence, ctc_target_raw_score=.99, policy=LIDPolicy()); self.assertNotIn(raw_only["status"], {"LANGUAGE_LEAK_CONFIRMED", "EVIDENCE_CONFLICT"})
    def test_lid_feature_names_separate_raw_and_calibrated_ctc(self):
        value = features({"probabilities": {"en": .1, "de": .9}, "whisper_target_probability": .8, "ctc_target_raw_score": 2.5, "ctc_target_calibrated_probability": .7, "duration_seconds": 1, "speech_ratio": .8})
        self.assertEqual(set(value), set(LID_FEATURES))
        self.assertEqual(value["ctc_target_raw_score"], 2.5)
        self.assertNotIn("ctc_target_probability", value)
    def test_configured_language_pair_drives_whisper_slots(self):
        value = features({"probabilities": {"fr": .91, "ja": .04}, "language": "fr", "whisper_probability": .91}, source_language="fr", target_language="ja")
        self.assertEqual(value["lid_source_probability"], .91)
        self.assertEqual(value["lid_target_probability"], .04)
        self.assertEqual(value["whisper_source_probability"], 0.0)
        value = features({"probabilities": {"fr": .04, "ja": .91}, "whisper_source_probability": .0, "whisper_target_probability": .91}, source_language="fr", target_language="ja")
        self.assertEqual(value["lid_target_probability"], .91)
        self.assertEqual(value["whisper_target_probability"], .91)

if __name__ == "__main__": unittest.main()
