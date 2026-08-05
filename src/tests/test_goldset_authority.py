from __future__ import annotations

import copy, json, tempfile, unittest
from pathlib import Path

from dubbing_pipeline.calibration.goldset_bridge import extract_goldset_features
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore, HumanLabel


def _evidence(_clip):
    target = {"expected_characters": 5, "expected_words": 1, "duration": 1.0, "raw_target_score": .9, "native_char_coverage": .9, "mean_char_score": .9, "minimum_char_score": .9, "p10_char_score": .9, "delete_count": 0, "substitute_count": 0, "insert_count": 0, "interpolated_count": 0, "compression_ratio": 1.0, "final_anchor_evidence": {"coverage": .9, "minimum_score": .9, "mean_score": .9, "duration_ms": 1000, "gap_to_active_speech_end_ms": 0, "deleted_characters": 0, "substituted_characters": 0, "insertions_inside_anchor": 0, "interpolated": False}}
    return {"target": target, "final": target, "lid": {"probabilities": {"en": .1, "de": .9}, "duration_seconds": 1.0, "speech_ratio": .9}}


class GoldsetAuthorityTests(unittest.TestCase):
    def _fixture(self, tmp: str):
        root = Path(tmp); store = GoldsetStore(root / "gold.sqlite")
        clip = ClipRecord("hidden", "a" * 64, "scene", "line", "take", "speaker", "Hallo", split="hidden_test")
        store.add_clip(clip); store.save_label(HumanLabel("hidden", "r1", labels=("CORRECT_NEUTRAL",))); store.save_label(HumanLabel("hidden", "r2", labels=("CORRECT_NEUTRAL",))); store.adjudicate("hidden", "lead", ("CORRECT_NEUTRAL",)); store.seal_hidden_test("operator")
        hidden = store.open_hidden_evaluation("operator", "run")
        bridge = extract_goldset_features(store, _evidence, root / "features", hidden_evaluation_receipt=hidden)
        rows = {}
        for role, path in (("target", bridge["paths_by_split"]["target"]["hidden_test"]), ("final_anchor", bridge["paths_by_split"]["final_anchor"]["hidden_test"]), ("lid", bridge["paths_by_split"]["lid"]["hidden_test"])):
            rows[role] = [json.loads(Path(path).read_text(encoding="utf-8").strip())]
        kwargs = {"receipt_id": hidden["receipt_id"], "run_id": "run", "profile_id": "profile", "code_commit": "a" * 40, "role_hidden_rows": rows, "role_hidden_reports": {role: {"run_id": f"{role}-run"} for role in rows}, "hidden_jsonl_hashes": {role: "c" * 64 for role in rows}, "hidden_report_hashes": {role: "d" * 64 for role in rows}}
        return store, kwargs

    def test_legitimate_bridge_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, kwargs = self._fixture(tmp); result = store.finalize_hidden_evaluation(**kwargs); self.assertTrue(result["bridge_receipts"]); store.close()

    def _reject_mutation(self, mutate):
        with tempfile.TemporaryDirectory() as tmp:
            store, kwargs = self._fixture(tmp); mutate(kwargs["role_hidden_rows"]); with_error = self.assertRaises(ValueError)
            with with_error: store.finalize_hidden_evaluation(**kwargs)
            store.close()

    def test_label_binary_mismatch_blocks(self): self._reject_mutation(lambda rows: rows["target"][0].__setitem__("label", 0))
    def test_arbitrary_label_hash_blocks(self): self._reject_mutation(lambda rows: rows["target"][0]["metadata"].__setitem__("label_payload_sha256", "a" * 64))
    def test_arbitrary_evidence_hash_blocks(self): self._reject_mutation(lambda rows: rows["target"][0]["metadata"].__setitem__("evidence_sha256", "b" * 64))
    def test_features_mutation_blocks(self): self._reject_mutation(lambda rows: rows["target"][0]["features"].__setitem__("target_score", 0.1))
    def test_audio_sha_mutation_blocks(self): self._reject_mutation(lambda rows: rows["target"][0]["metadata"].__setitem__("audio_sha256", "f" * 64))

    def test_receipt_from_other_run_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, kwargs = self._fixture(tmp); kwargs["run_id"] = "other-run"
            with self.assertRaises(ValueError): store.finalize_hidden_evaluation(**kwargs)
            store.close()

    def test_labels_changed_before_seal_change_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite"); clip = ClipRecord("h", "a" * 64, "s", "l", "t", "p", "Hallo", split="hidden_test"); store.add_clip(clip); store.save_label(HumanLabel("h", "r1", labels=("CORRECT_NEUTRAL",))); store.save_label(HumanLabel("h", "r2", labels=("CORRECT_NEUTRAL",))); first = store.seal_hidden_test("operator"); self.assertEqual(first["digest"], store.hidden_seal()["digest"]); store.close()

    def test_labels_changed_after_seal_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GoldsetStore(Path(tmp) / "gold.sqlite"); clip = ClipRecord("h", "a" * 64, "s", "l", "t", "p", "Hallo", split="hidden_test"); store.add_clip(clip); store.save_label(HumanLabel("h", "r1", labels=("CORRECT_NEUTRAL",))); store.save_label(HumanLabel("h", "r2", labels=("CORRECT_NEUTRAL",))); store.seal_hidden_test("operator")
            with self.assertRaises(ValueError): store.save_label(HumanLabel("h", "r1", labels=("TIMING_BAD",)))
            store.close()


if __name__ == "__main__": unittest.main()
