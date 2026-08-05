from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from dubbing_pipeline.calibration.goldset_bridge import extract_goldset_features
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore, HumanLabel


class BridgeTests(unittest.TestCase):
    def test_goldset_produces_three_separate_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            clip = ClipRecord("c", "a" * 64, "scene", "line", "take", "speaker", "Hallo", split="calibration")
            store.add_clip(clip)
            for reviewer in ("r1", "r2"):
                store.save_label(HumanLabel("c", reviewer, labels=("CORRECT_NEUTRAL",)))
            target = {"raw_target_score": .9, "expected_characters": 5, "expected_words": 1, "native_char_coverage": 1.0, "mean_char_score": .9, "minimum_char_score": .8, "p10_char_score": .85, "delete_count": 0, "substitute_count": 0, "insert_count": 0, "interpolated_count": 0, "compression_ratio": 1.0, "duration": 1.0, "final_anchor_evidence": {"coverage": 1.0, "minimum_score": .8, "mean_score": .9, "duration_ms": 200, "gap_to_active_speech_end_ms": 20, "deleted_characters": 0, "substituted_characters": 0, "insertions_inside_anchor": 0, "interpolated": False}}
            lid = {"probabilities": {"en": .05, "de": .9}, "language": "de", "probability": .9, "duration_seconds": 1.0, "speech_ratio": .8}
            result = extract_goldset_features(store, lambda _clip: {"target": target, "final": target, "lid": lid}, Path(tmp) / "features")
            self.assertEqual(result["counts"], {"target": 1, "final_anchor": 1, "lid": 1})
            self.assertEqual(len(Path(result["paths"]["target"]).read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(Path(result["paths_by_split"]["target"]["calibration"]).read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(Path(result["paths_by_split"]["target"]["hidden_test"]).read_text(encoding="utf-8"), "")
            self.assertEqual(result["schema"], "goldset-feature-bridge-v2")
            self.assertEqual(json.loads(Path(result["paths"]["lid"]).read_text(encoding="utf-8"))["feature_schema_version"], "lid-fusion-v1")
            store.close()

    def test_bridge_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            store.add_clip(ClipRecord("c", "a" * 64, "scene", "line", "take", "speaker", "Hallo", split="calibration"))
            for reviewer in ("r1", "r2"):
                store.save_label(HumanLabel("c", reviewer, labels=("CORRECT_NEUTRAL",)))
            with self.assertRaises(ValueError): extract_goldset_features(store, lambda _clip: {}, Path(tmp) / "features")
            store.close()

    def test_adjudicated_consensus_is_the_only_calibration_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            store.add_clip(ClipRecord("c", "a" * 64, "scene", "line", "take", "speaker", "Hallo", split="calibration"))
            store.save_label(HumanLabel("c", "r1", labels=("CORRECT_NEUTRAL",)))
            store.save_label(HumanLabel("c", "r2", labels=("LEXICAL_ERROR",)))
            store.adjudicate("c", "lead", ("CORRECT_NEUTRAL",))
            target = {"raw_target_score": .9, "expected_characters": 5, "expected_words": 1, "native_char_coverage": 1.0, "mean_char_score": .9, "minimum_char_score": .8, "p10_char_score": .85, "delete_count": 0, "substitute_count": 0, "insert_count": 0, "interpolated_count": 0, "compression_ratio": 1.0, "duration": 1.0, "final_anchor_evidence": {"coverage": 1.0, "minimum_score": .8, "mean_score": .9, "duration_ms": 200, "gap_to_active_speech_end_ms": 20, "deleted_characters": 0, "substituted_characters": 0, "insertions_inside_anchor": 0, "interpolated": False}}
            lid = {"probabilities": {"en": .05, "de": .9}, "duration_seconds": 1.0, "speech_ratio": .8}
            result = extract_goldset_features(store, lambda _clip: {"target": target, "final": target, "lid": lid}, Path(tmp) / "features")
            row = json.loads(Path(result["paths"]["target"]).read_text(encoding="utf-8"))
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["metadata"]["label_authority"], "adjudicated_consensus")
            self.assertEqual(len(store.effective_labels()), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
