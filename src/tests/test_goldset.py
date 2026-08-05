from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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

    def test_hidden_membership_requires_one_time_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            hidden = clip("hidden", "hidden-line", "hidden_test")
            try:
                store.add_clip(hidden)
                self.assertFalse(validate_goldset(store.clips(), [], require_double_review=False)["hidden_test_sealed"])
                seal = store.seal_hidden_test("lead")
                self.assertTrue(seal["seal_id"])
                self.assertTrue(validate_goldset(store.clips(), [], require_double_review=False, hidden_sealed=True)["hidden_test_sealed"])
                store.mark_hidden_opened("lead")
                with self.assertRaises(ValueError): store.mark_hidden_opened("lead")
            finally:
                store.close()

    def test_hidden_seal_freezes_membership_labels_and_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.sqlite"
            store = GoldsetStore(path)
            hidden = clip("hidden", "hidden-line", "hidden_test")
            store.add_clip(hidden)
            seal = store.seal_hidden_test("lead")
            self.assertTrue(store.verify_hidden_seal())
            with self.assertRaises(ValueError):
                store.add_clip(clip("hidden-2", "hidden-line-2", "hidden_test"))
            with self.assertRaises(ValueError):
                store.save_label(HumanLabel("hidden", "reviewer", "CORRECT_NEUTRAL"))
            with self.assertRaises(ValueError):
                store.mark_hidden_opened("other")
            self.assertEqual(store.hidden_seal()["digest"], seal["digest"])
            store.close()

    def test_normal_claim_cannot_return_or_request_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            store.add_clip(clip("cal", "cal-line", "calibration"))
            store.add_clip(clip("hidden", "hidden-line", "hidden_test"))
            self.assertEqual(store.claim("reviewer", split="calibration").clip_id, "cal")
            with self.assertRaises(PermissionError):
                store.claim("reviewer", split="hidden_test")
            self.assertIsNone(store.claim("reviewer"))
            store.close()

    def test_hidden_evaluation_is_atomic_one_shot_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            store.add_clip(clip("hidden", "hidden-line", "hidden_test"))
            store.seal_hidden_test("operator")
            receipts = []
            def open_once():
                try:
                    receipts.append(store.open_hidden_evaluation("operator", "run-1"))
                except ValueError:
                    return
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _: open_once(), range(2)))
            self.assertEqual(len(receipts), 1)
            receipt = receipts[0]
            self.assertTrue(store.verify_hidden_evaluation_receipt(receipt))
            tampered = dict(receipt); tampered["clips"] = []
            self.assertFalse(store.verify_hidden_evaluation_receipt(tampered))
            with self.assertRaises(ValueError):
                store.open_hidden_evaluation("operator", "run-2")
            store.close()

    def test_hidden_finalization_is_authoritative_exact_and_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite")
            clip_value = clip("hidden", "hidden-line", "hidden_test")
            store.add_clip(clip_value); store.save_label(HumanLabel("hidden", "reviewer-a", labels=("CORRECT_NEUTRAL",))); store.save_label(HumanLabel("hidden", "reviewer-b", labels=("CORRECT_NEUTRAL",))); store.adjudicate("hidden", "lead", ("CORRECT_NEUTRAL",)); store.seal_hidden_test("operator")
            receipt = store.open_hidden_evaluation("operator", "run-final")
            def row(role):
                return {"clip_id": "hidden", "split": "hidden_test", "split_group": "hidden-line", "label": 0, "features": {"x": 0.1}, "metadata": {"audio_sha256": clip_value.audio_sha256, "label_hash": "a" * 64, "evidence_hash": "b" * 64, "role": role}}
            rows = {role: [row(role)] for role in ("target", "final_anchor", "lid")}
            reports = {role: {"run_id": f"{role}-run"} for role in rows}
            finalization = store.finalize_hidden_evaluation(receipt_id=receipt["receipt_id"], run_id="run-final", profile_id="profile", code_commit="a" * 40, role_hidden_rows=rows, role_hidden_reports=reports, hidden_jsonl_hashes={role: "c" * 64 for role in rows}, hidden_report_hashes={role: "d" * 64 for role in rows})
            verified = store.verify_hidden_evaluation_finalization(finalization["finalization_id"], profile_id="profile", code_commit="a" * 40)
            self.assertIsNone(verified["consumed_at"])
            consumed = store.consume_hidden_finalization(finalization["finalization_id"], profile_id="profile", code_commit="a" * 40)
            self.assertIsNotNone(consumed["consumed_at"])
            with self.assertRaises(ValueError):
                store.consume_hidden_finalization(finalization["finalization_id"], profile_id="profile", code_commit="a" * 40)
            store.close()

if __name__ == "__main__": unittest.main()
