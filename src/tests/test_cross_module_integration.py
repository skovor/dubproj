from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dubbing_pipeline.benchmark import (
    BenchmarkManifest,
    run_benchmark,
    validate_manifest,
    verify_benchmark_row,
)
from dubbing_pipeline.calibration import (
    FINAL_ANCHOR_FEATURES,
    LID_FEATURES,
    TARGET_FEATURES,
    FeatureRow,
    LIDFeatureRow,
    extract_goldset_features,
    train_calibrator,
)
from dubbing_pipeline.calibration.promote import promote_profile
from dubbing_pipeline.calibration.validate import evaluate
from dubbing_pipeline.config import PipelineConfig, QAConfig
from dubbing_pipeline.fmv_selector import select_local_scene
from dubbing_pipeline.goldset import ClipRecord, GoldsetStore, HumanLabel
from dubbing_pipeline.hashing import contract_hash, sha256_file
from dubbing_pipeline.lid import LIDEvidence, LIDPolicy, fuse_language_evidence
from dubbing_pipeline.qa_v2 import (
    _feature_vector_hash,
    apply_independent_evidence,
    calibration_profile_status,
    decide_linguistic_evidence,
)
from dubbing_pipeline.scene_qa import audit_scene_windows
from scripts.promote_branch import execute_second_game_adapter


def _feature_evidence(clip: ClipRecord) -> dict:
    bad = clip.clip_id.endswith("bad")
    score = 0.1 if bad else 0.95
    return {
        "target": {
            "expected_characters": 5,
            "expected_words": 1,
            "duration": 1.0,
            "raw_target_score": score,
            "native_char_coverage": score,
            "mean_char_score": score,
            "minimum_char_score": score,
            "p10_char_score": score,
            "delete_ratio": 1.0 - score,
            "substitute_ratio": 1.0 - score,
            "insert_ratio": 1.0 - score,
            "interpolated_ratio": 0.0,
            "compression_ratio": score,
            "characters_per_second": 5.0,
            "words_per_second": 1.0,
        },
        "final": {
            "coverage": score,
            "minimum_score": score,
            "mean_score": score,
            "final_duration": 1.0,
            "gap_to_active_speech_end_ms": 0.0 if not bad else 200.0,
            "delete_count": 0 if not bad else 2,
            "substitute_count": 0 if not bad else 2,
            "insert_count": 0,
            "final_interpolated": 0.0,
        },
        "lid": {
            "probabilities": {"en": 0.95 if bad else 0.05, "de": 0.05 if bad else 0.95},
            "lid_source_probability": 0.95 if bad else 0.05,
            "lid_target_probability": 0.05 if bad else 0.95,
            "whisper_source_probability": 0.95 if bad else 0.05,
            "whisper_target_probability": 0.05 if bad else 0.95,
            "ctc_target_raw_score": score,
            "ctc_target_calibrated_probability": score,
            "duration_seconds": 1.0,
            "speech_ratio": 0.9,
            "source_language": "en",
            "target_language": "de",
        },
    }


def _rows(path: Path, row_type):
    result = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        result.append(
            row_type(
                value["clip_id"], value["split"], value["split_group"],
                int(value["label"]), dict(value["features"]),
                value.get("performance_mode", "NEUTRAL"), value.get("metadata"),
            )
        )
    return result


class CrossModuleIntegrationTests(unittest.TestCase):
    def test_goldset_bridge_train_validate_promote_runtime_and_receipt_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = GoldsetStore(root / "gold.sqlite")
            clips = []
            for split in ("calibration", "validation", "hidden_test"):
                for index in range(4):
                    suffix = "bad" if index % 2 else "good"
                    clip = ClipRecord(
                        f"{split}-{index}-{suffix}",
                        hashlib.sha256(f"{split}-{index}".encode()).hexdigest(),
                        f"scene-{split}-{index}", f"line-{split}-{index}",
                        f"take-{index}", "speaker", "Hallo", split_group=f"{split}-{index}", split=split,
                    )
                    clips.append(clip)
                    store.add_clip(clip)
                    label = "TIMING_BAD" if suffix == "bad" else "CORRECT_NEUTRAL"
                    if suffix == "bad":
                        labels = ("TIMING_BAD", "SOURCE_LANGUAGE_LEAK")
                    else:
                        labels = (label,)
                    store.save_label(HumanLabel(clip.clip_id, "reviewer-a", labels=labels))
                    store.save_label(HumanLabel(clip.clip_id, "reviewer-b", labels=labels))
                    store.adjudicate(clip.clip_id, "lead", labels)
            store.seal_hidden_test("operator")
            hidden_receipt = store.open_hidden_evaluation("operator", "integration-run-1")
            bridge = extract_goldset_features(
                store, _feature_evidence, root / "features",
                hidden_evaluation_receipt=hidden_receipt,
            )
            self.assertEqual(bridge["counts_by_split"]["target"]["calibration"], 4)
            self.assertTrue(bridge["hidden_evaluation_receipt"]["receipt_sha256"])

            target_rows = _rows(Path(bridge["paths"]["target"]), FeatureRow)
            anchor_rows = _rows(Path(bridge["paths"]["final_anchor"]), FeatureRow)
            lid_rows = _rows(Path(bridge["paths"]["lid"]), LIDFeatureRow)
            target_artifact = train_calibrator([row for row in target_rows if row.split == "calibration"], kind="target", features=TARGET_FEATURES, dataset_sha256=bridge["sha256"]["target"])
            anchor_artifact = train_calibrator([row for row in anchor_rows if row.split == "calibration"], kind="final_anchor", features=FINAL_ANCHOR_FEATURES, dataset_sha256=bridge["sha256"]["final_anchor"])
            lid_artifact = train_calibrator([row for row in lid_rows if row.split == "calibration"], kind="lid", features=LID_FEATURES, dataset_sha256=bridge["sha256"]["lid"])
            artifact_paths = {}
            for name, artifact in (("target", target_artifact), ("final", anchor_artifact), ("lid", lid_artifact)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
                artifact_paths[name] = path
            target_validation = [row for row in target_rows if row.split == "validation"]
            target_hidden = [row for row in target_rows if row.split == "hidden_test"]
            anchor_validation = [row for row in anchor_rows if row.split == "validation"]
            anchor_hidden = [row for row in anchor_rows if row.split == "hidden_test"]
            lid_validation = [row for row in lid_rows if row.split == "validation"]
            lid_hidden = [row for row in lid_rows if row.split == "hidden_test"]
            reports = {
                "target": (evaluate(target_artifact, target_validation, split="validation", run_id="target-v"), evaluate(target_artifact, target_hidden, split="hidden_test", run_id="target-h")),
                "final_anchor": (evaluate(anchor_artifact, anchor_validation, split="validation", run_id="anchor-v"), evaluate(anchor_artifact, anchor_hidden, split="hidden_test", run_id="anchor-h")),
                "lid": (evaluate(lid_artifact, lid_validation, split="validation", run_id="lid-v"), evaluate(lid_artifact, lid_hidden, split="hidden_test", run_id="lid-h")),
            }
            for role, (_validation, hidden) in reports.items():
                self.assertEqual(hidden.false_pass_count, 0, role)
            dataset_files = {}
            for name, key in (("manifest", "manifest_sha256"), ("labels", "labels_sha256"), ("splits", "split_manifest_sha256")):
                path = root / f"{name}.json"
                path.write_text(name, encoding="utf-8")
                dataset_files[key] = path
            code_commit = "a" * 40
            runtime_lock = root / "runtime.lock"; runtime_lock.write_bytes(b"runtime")
            models_lock = root / "models.lock"; models_lock.write_bytes(b"models")
            profile = promote_profile(
                profile_id="integration-profile",
                target_artifact={**target_artifact.to_dict(), "artifact_path": str(artifact_paths["target"]), "artifact_sha256": sha256_file(artifact_paths["target"])},
                final_anchor_artifact={**anchor_artifact.to_dict(), "artifact_path": str(artifact_paths["final"]), "artifact_sha256": sha256_file(artifact_paths["final"])},
                lid_artifact={**lid_artifact.to_dict(), "artifact_path": str(artifact_paths["lid"]), "artifact_sha256": sha256_file(artifact_paths["lid"])},
                validation=reports["target"][0], hidden=reports["target"][1],
                validation_rows=target_validation, hidden_rows=target_hidden,
                role_validation_rows={"target": target_validation, "final_anchor": anchor_validation, "lid": lid_validation},
                role_hidden_rows={"target": target_hidden, "final_anchor": anchor_hidden, "lid": lid_hidden},
                role_validation_reports={role: pair[0] for role, pair in reports.items()},
                role_hidden_reports={role: pair[1] for role, pair in reports.items()},
                dataset_files=dataset_files,
                identity={"backend_id": "integration", "model_id": "model", "model_revision": "1", "feature_schema_version": "char-alignment-v3", "target_language": "de", "source_language": "en", "performance_modes": ["NEUTRAL"]},
                thresholds={"target_pass_probability": .8, "target_failure_probability": .2, "final_anchor_pass_probability": .8, "source_lid_probability": .8},
                provenance={"code_commit": code_commit, "runtime_lock_sha256": sha256_file(runtime_lock), "models_lock_sha256": sha256_file(models_lock)},
                output=root / "profile.json", hidden_evaluation_receipt=hidden_receipt, require_hidden_evaluation_receipt=True,
            )
            self.assertEqual(profile["status"], "VALIDATED")
            matched = calibration_profile_status(profile, authority=True, backend_id="integration", model_id="model", model_revision="1", target_language="de", source_language="en", performance_mode="NEUTRAL", runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock), expected_code_commit=code_commit, calibrator_root=root, require_promotion_receipt=True)
            self.assertEqual(matched, "MATCHED_VALIDATED")
            base = decide_linguistic_evidence("Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99}, automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=__import__("dubbing_pipeline.qa_v2", fromlist=["LanguageProfile"]).LanguageProfile(), evidence_records=[{"evidence_family": "WHISPER_ASR", "evidence_hash": "a" * 64}])
            alignment = {"target_score": .95, "source_score": .10, "target": {"native_char_coverage": .95, "mean_char_score": .95, "minimum_char_score": .95, "p10_char_score": .95, "compression_ratio": .95, "characters_per_second": 5.0, "words_per_second": 1.0, "duration": 1.0, "char_segments": [{"expected_index": index} for index in range(5)], "final_anchor_evidence": {"coverage": .95, "minimum_score": .95, "mean_score": .95, "duration_ms": 1000.0, "gap_to_active_speech_end_ms": 0.0, "deleted_characters": 0, "substituted_characters": 0, "insertions_inside_anchor": 0, "interpolated": False, "status": "FINAL_ANCHOR_EVIDENCE_COLLECTED", "timing_valid": True, "expected_characters": 5}}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER", "evidence_hash": "b" * 64}]}
            lid_evidence = {"language": "de", "probability": .95, "probabilities": {"de": .95, "en": .05}, "record": {"evidence_family": "AUDIO_LANGUAGE_ID", "evidence_hash": "c" * 64}, "duration_seconds": 1.0, "speech_ratio": .9}
            decision = apply_independent_evidence(base, alignment, lid_evidence=lid_evidence, calibration_authority=True, calibration_profile=profile, calibration_profile_root=root, backend_id="integration", model_id="model", model_revision="1", target_language="de", source_language="en", performance_mode="NEUTRAL", runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock), expected_code_commit=code_commit, require_promotion_receipt=True)
            self.assertIsNotNone(decision.lid_feature_vector)
            self.assertEqual(decision.lid_feature_vector_hash, _feature_vector_hash(decision.lid_feature_vector, "lid-fusion-v3"))
            mutated = json.loads(json.dumps(profile)); mutated["thresholds"]["target_pass_probability"] = .01
            self.assertIn(calibration_profile_status(mutated, authority=True, backend_id="integration", model_id="model", model_revision="1", target_language="de", source_language="en", performance_mode="NEUTRAL", runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock), expected_code_commit=code_commit, calibrator_root=root, require_promotion_receipt=True), {"BLOCKED_INVALID_THRESHOLDS", "BLOCKED_PROMOTION_RECEIPT"})
            with self.assertRaises(ValueError):
                store.open_hidden_evaluation("operator", "integration-run-2")
            store.close()

    def test_lid_raw_score_is_diagnostic_until_calibrated(self):
        lid = LIDEvidence("LID_CONFIDENT", "en", {"en": .95, "de": .05}, "lid", "model", "1", "a" * 64, 1.0, 24000, .9, "b" * 64, record={"evidence_family": "AUDIO_LANGUAGE_ID"})
        raw_only = fuse_language_evidence(whisper_language="en", whisper_probability=.95, lid=lid, ctc_target_raw_score=.99, policy=LIDPolicy())
        calibrated = fuse_language_evidence(whisper_language="en", whisper_probability=.95, lid=lid, ctc_target_raw_score=.99, ctc_target_calibrated_probability=.10, policy=LIDPolicy())
        self.assertNotEqual(raw_only["status"], "LANGUAGE_LEAK_CONFIRMED")
        self.assertEqual(calibrated["status"], "LANGUAGE_LEAK_CONFIRMED")
        self.assertIsNone(raw_only["ctc_target_calibrated_probability"])

    def test_fmv_channel_repair_benchmark_and_adapter_contracts_cross_module(self):
        audio = np.zeros((2400, 2), dtype="float32"); audio[:, 0] = .01; audio[100:900, 1] = .2
        source = np.zeros_like(audio); source[:, 0] = .01
        scene_audit = audit_scene_windows(audio, 24000, [{"line_id": "L1", "start": 0, "end": 2400}], dialogue_channel=1, source_audio=source, require_mounted_delta_line_ids={"L1"})
        self.assertTrue(scene_audit["all_lines_active"])
        class Line:
            def __init__(self, line_id): self.id = line_id
        lines = [Line("A"), Line("B")]
        options = {"A": [{"candidate": type("C", (), {"candidate_id": "A1"})(), "eligible": True}, {"candidate": type("C", (), {"candidate_id": "A2"})(), "eligible": True}], "B": [{"candidate": type("C", (), {"candidate_id": "B1"})(), "eligible": True}, {"candidate": type("C", (), {"candidate_id": "B2"})(), "eligible": True}]}
        def audit(value, _index):
            selected = dict(value)
            passed = selected == {"A": "A2", "B": "B1"}
            return passed, type("Audit", (), {"diagnostics": {} if passed else {"failed_line_ids": ["B"]}})()
        fmv = select_local_scene(lines, options, [], max_candidates_per_line=2, max_iterations=4, mount_line=lambda value, line, option: {**dict(value), line.id: option["candidate"].candidate_id}, audit_scene=audit)
        self.assertTrue(fmv.passed); self.assertEqual(fmv.selected["A"]["candidate"].candidate_id, "A2")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / "output.wav"; reference = root / "reference.wav"; model_lock = root / "models.lock"; runtime_lock = root / "runtime.lock"
            sf.write(output, audio, 24000); sf.write(reference, audio, 24000); model_lock.write_bytes(b"models"); runtime_lock.write_bytes(b"runtime")
            manifest = BenchmarkManifest.from_paths(benchmark_id="integration", line_ids=("L1",), audio_paths=(str(output),), reference_paths=(str(reference),), model_lock=str(model_lock), runtime_lock=str(runtime_lock), calibration_profile=None, config_hash="config", commit="commit", topology="LINE_SEPARATED")
            self.assertTrue(validate_manifest(manifest)["real_audio"])
            report_path = root / "report.json"; report = {"lines": [{"id": "L1", "status": "FINAL_PASS", "output": str(output)}]}; report_path.write_text(json.dumps(report), encoding="utf-8")
            row = {"status": "FINAL_PASS", "report_path": str(report_path), "output_path": str(output), "report_sha256": sha256_file(report_path), "output_sha256": sha256_file(output), "report_contract_hash": contract_hash("benchmark-report-v1", {"line_id": "L1", "report": report}, files=[output])}
            self.assertTrue(verify_benchmark_row("L1", row)["valid"])
            second_manifest = root / "second.json"; second_manifest.write_text(json.dumps({"scenes": [{"scene_id": "s", "game_id": "dq3", "audio_path": str(output), "reference_path": str(reference), "audio_sha256": sha256_file(output), "reference_sha256": sha256_file(reference), "timing": {"start": 0, "end": 1}, "extraction_status": "VERIFIED"}]}), encoding="utf-8")
            result = execute_second_game_adapter({"game_id": "dq3", "adapter_entrypoint": "adapters.second_game_template:SecondGameAdapter", "manifest_path": str(second_manifest)}, Path(__file__).resolve().parents[1])
            self.assertTrue(result["executed"]); self.assertTrue(result["valid"]); self.assertTrue(result["content_verified"])
            with self.assertRaises(ValueError):
                run_benchmark(manifest, lambda *_: {"status": "FINAL_PASS"}, require_files=True, require_trusted_runner=True, repository_root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
