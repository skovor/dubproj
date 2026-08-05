from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore, HumanLabel, stable_split, validate_goldset

SHA = "a" * 64

def clip(name="c1", line="l1", split=None):
    return ClipRecord(name, SHA, "scene", line, "take", "speaker", "Hallo", split_group=f"scene:{line}", split=split or "calibration")

class GoldsetTests(unittest.TestCase):
    def test_group_split_is_stable(self):
        self.assertEqual(stable_split("scene:l1"), stable_split("scene:l1"))

    def test_review_payload_has_no_automatic_evidence(self):
        value = clip().review_payload(); self.assertNotIn("generation_provenance", value); self.assertNotIn("score", value); self.assertEqual(value["expected_text"], "Hallo")

    def test_store_requires_human_double_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite"); store.add_clip(clip())
            store.save_label(HumanLabel("c1", "a", "CORRECT_NEUTRAL"))
            result = validate_goldset(store.clips(), store.labels()); self.assertFalse(result["valid"]); self.assertTrue(any("independent" in e for e in result["errors"])); store.close()

    def test_claim_allows_two_reviewers_but_not_duplicate_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite"); store.add_clip(clip())
            self.assertEqual(store.claim("reviewer-a").clip_id, "c1")
            self.assertIsNone(store.claim("reviewer-a"))
            self.assertEqual(store.claim("reviewer-b").clip_id, "c1")
            store.save_label(HumanLabel("c1", "reviewer-a", labels=("CORRECT_NEUTRAL", "TIMING_BAD")))
            self.assertIsNone(store.claim("reviewer-a"))
            store.close()

    def test_clip_content_is_immutable_and_adjudication_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite"); store.add_clip(clip())
            with self.assertRaises(ValueError):
                store.add_clip(ClipRecord("c1", SHA, "scene", "l1", "other-take", "speaker", "Different"))
            store.save_label(HumanLabel("c1", "a", "CORRECT_NEUTRAL")); store.save_label(HumanLabel("c1", "b", "LEXICAL_ERROR"))
            self.assertFalse(validate_goldset(store.clips(), store.labels())["valid"])
            store.adjudicate("c1", "lead", ("CORRECT_NEUTRAL",), comment="reviewed")
            result = validate_goldset(store.clips(), store.labels())
            self.assertTrue(result["valid"])
            store.close()

    def test_disagreement_requires_adjudication(self):
        c = clip(); labels = [HumanLabel("c1", "a", "CORRECT_NEUTRAL"), HumanLabel("c1", "b", "LEXICAL_ERROR")]
        result = validate_goldset([c], labels); self.assertFalse(result["valid"]); self.assertTrue(any("disagreement" in e for e in result["errors"]))

    def test_split_leakage_is_blocked(self):
        c1 = clip("c1", "l1", "calibration"); c2 = clip("c2", "l1", "validation")
        result = validate_goldset([c1, c2], []); self.assertTrue(any("line crosses" in e for e in result["errors"]))

if __name__ == "__main__": unittest.main()
