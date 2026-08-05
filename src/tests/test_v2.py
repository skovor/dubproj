from __future__ import annotations

import json
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from dubbing_pipeline.alignment import AlignmentCache, AlignmentTextPolicy, _character_evidence, contrastive_align, normalize_alignment_text
from dubbing_pipeline.alignment import _extract_mfa_words, _raw_char_segments
from dubbing_pipeline.contracts import ContractError, DeliveryWindow, EvidenceFamily, GateStatus, ReferenceEvidence
from dubbing_pipeline.contracts.manifest import validate_manifest_value
from dubbing_pipeline.deploy_v2 import DeploymentError, PackageEntry, deploy_atomic_v2, stage_files_v2
from dubbing_pipeline.hashing import atomic_json, contract_hash, sha256_file
from dubbing_pipeline.models import Line
from dubbing_pipeline.montage import mount_surgical
from dubbing_pipeline.asr import ASRCache, prepare_whisperx_escalation, transcribe_dual
from dubbing_pipeline.qa_v2 import LanguageProfile, apply_independent_evidence, decide_linguistic_evidence, evaluate_candidate_v2, final_word, load_safe_calibrator, ordered_content, predict_probability, source_language_leak
from dubbing_pipeline.scheduler import run_cohorts
from dubbing_pipeline.telemetry import TelemetryCollector
from dubbing_pipeline.config import PipelineConfig, QAConfig
from dubbing_pipeline.generation_v2 import GenerationRuntimeV2
from dubbing_pipeline.generation_v2 import GenerationRequest
from dubbing_pipeline.models import Scene
from dubbing_pipeline.orchestration_v2 import run_scene_v2
from dubbing_pipeline.orchestration_v2 import _line_linguistic_summary
from dubbing_pipeline.post_qa import audit_candidate_stage, audit_scene_stage, persist_audio_atomic
from dubbing_pipeline.reference import materialize_reference


def _validated_profile(root: Path, *, model_revision: str = "test") -> tuple[dict, Path, Path]:
    artifact = root / "target_calibrator.json"
    final_artifact = root / "final_anchor_calibrator.json"
    lid_artifact = root / "lid_calibrator.json"
    runtime_lock = root / "runtime.lock"
    models_lock = root / "models.lock"
    target_features = ["target_score", "native_char_coverage", "mean_char_score", "minimum_char_score", "p10_char_score", "delete_ratio", "substitute_ratio", "insert_ratio", "interpolated_ratio", "compression_ratio", "characters_per_second", "words_per_second", "duration", "performance_mode"]
    final_features = ["final_coverage", "final_minimum_score", "final_mean_score", "final_duration", "gap_to_active_speech_end_ms", "final_delete_count", "final_substitute_count", "insertions_inside_anchor", "final_interpolated"]
    normalization = [{"mean": 0.0, "scale": 1.0} for _ in target_features]
    final_normalization = [{"mean": 0.0, "scale": 1.0} for _ in final_features]
    artifact.write_text(json.dumps({
        "schema": "platt-calibrator-v1",
        "feature_schema_version": "char-alignment-v2",
        "normalization_version": "alignment-text-normalization-v2",
        "features": target_features,
        "coefficients": [1.2, .8, .7, .7, .4, -1.0, -1.0, -1.0, -1.0, -.3, 0.0, 0.0, 0.0, 0.0],
        "intercept": -1.1,
        "normalization": normalization,
    }), encoding="utf-8")
    final_artifact.write_text(json.dumps({
        "schema": "platt-calibrator-v1", "feature_schema_version": "final-anchor-v1", "normalization_version": "alignment-text-normalization-v2",
        "features": final_features, "coefficients": [1.2, .8, .7, .7, -.01, -1.0, -1.0, -1.0, -1.0], "intercept": -1.1, "normalization": final_normalization,
    }), encoding="utf-8")
    lid_features = ["lid_source_probability", "lid_target_probability", "whisper_source_probability", "whisper_target_probability", "ctc_target_raw_score", "ctc_target_calibrated_probability", "duration_seconds", "speech_ratio", "performance_mode"]
    lid_artifact.write_text(json.dumps({
        "schema": "platt-calibrator-v1", "feature_schema_version": "lid-fusion-v1", "normalization_version": "alignment-text-normalization-v2",
        "features": lid_features, "coefficients": [1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "intercept": 0.0,
        "normalization": [{"mean": 0.0, "scale": 1.0} for _ in lid_features],
    }), encoding="utf-8")
    runtime_lock.write_bytes(b"runtime-lock-for-test")
    models_lock.write_bytes(b"models-lock-for-test")
    profile = {
        "schema": "generic-dubbing-alignment-calibration-profile-v2",
        "status": "VALIDATED",
        "authority": True,
        "profile_id": "de-neutral-whisperx-001",
        "identity": {
            "backend_id": "fake-ctc",
            "model_id": "fake-german",
            "model_revision": model_revision,
            "feature_schema_version": "char-alignment-v2",
            "target_language": "de",
            "source_language": "en",
            "performance_modes": ["NEUTRAL"],
        },
        "thresholds": {
            "target_pass_probability": .65,
            "target_failure_probability": .50,
            "final_anchor_pass_probability": .65,
            "source_lid_probability": .70,
        },
        "calibrators": {
        "target": {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": "char-alignment-v2", "normalization_version": "alignment-text-normalization-v2", "feature_names": target_features, "artifact_path": str(artifact), "artifact_sha256": sha256_file(artifact)},
        "final_anchor": {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": "final-anchor-v1", "normalization_version": "alignment-text-normalization-v2", "feature_names": final_features, "artifact_path": str(final_artifact), "artifact_sha256": sha256_file(final_artifact)},
        "lid": {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": "lid-fusion-v1", "normalization_version": "alignment-text-normalization-v2", "feature_names": lid_features, "artifact_path": str(lid_artifact), "artifact_sha256": sha256_file(lid_artifact)},
        },
        "dataset": {
            "manifest_sha256": "a" * 64,
            "labels_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
            "calibration_count": 180,
            "validation_count": 60,
            "hidden_test_count": 60,
        },
        "metrics": {"hidden_false_pass_count": 0, "hidden_false_fail_count": 1, "brier_score": .03, "expected_calibration_error": .02},
        "provenance": {
            "code_commit": "d" * 40,
            "runtime_lock_sha256": sha256_file(runtime_lock),
            "models_lock_sha256": sha256_file(models_lock),
            "created_at": "2026-08-04T00:00:00Z",
        },
    }
    return profile, runtime_lock, models_lock


class V2ContractsTests(unittest.TestCase):
    def test_strict_manifest_rejects_unknown_and_overlap(self):
        base = {"id": "scene", "topology": "EMBEDDED_FMV", "movie_identity_verified": True, "duration_seconds": 3.0, "lines": []}
        base["lines"] = [{"id": "a", "speaker": "A", "source_text": "Hello", "target_text": "Hallo", "start": 0.0, "end": 1.5, "subtitle_authorized": True, "movie_identity_verified": True, "card_identity_verified": True, "card_timebase_verified": True}, {"id": "b", "speaker": "A", "source_text": "Bye", "target_text": "Tschüss", "start": 1.0, "end": 2.0, "subtitle_authorized": True, "movie_identity_verified": True, "card_identity_verified": True, "card_timebase_verified": True}]
        with self.assertRaises(ContractError): validate_manifest_value(base)
        base["lines"][1]["start"] = 1.5
        validate_manifest_value(base)
        base["extra"] = True
        with self.assertRaises(ContractError): validate_manifest_value(base)

    def test_reference_text_uses_segment_transcript(self):
        line = Line("x", "A", "Wrong full stem", "Richtig", reference_segments=[])
        line.reference_segments = [__import__("dubbing_pipeline.models", fromlist=["ReferenceSegment"]).ReferenceSegment("ref.wav", text="Exact segment")]
        self.assertEqual(line.reference_text, "Exact segment")

    def test_legacy_metadata_roundtrip_is_not_nested(self):
        value = Line.from_dict({"id": "x", "speaker": "A", "source_text": "a", "target_text": "b", "metadata": {"flag": True}}).to_dict()
        self.assertEqual(value["metadata"], {"flag": True})

    def test_canonical_hash_does_not_depend_on_path(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.bin"; second = Path(directory) / "b.bin"; first.write_bytes(b"same"); second.write_bytes(b"same")
            self.assertEqual(contract_hash("x", {"a": 1}, [first]), contract_hash("x", {"a": 1}, [second]))

    def test_reference_materialization_pairs_segment_audio_and_text(self):
        import soundfile as sf
        from dubbing_pipeline.models import ReferenceSegment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "long.wav"; value = np.column_stack([np.zeros(4800, dtype="float32"), np.ones(4800, dtype="float32") * .2]); sf.write(source, value, 24000)
            line = Line("L", "Speaker", "Full sentence", "Ziel", reference_segments=[ReferenceSegment(str(source), start=.05, end=.10, text="Exact", channel=1)])
            evidence = materialize_reference(line, root, root / "cache")
            self.assertEqual(evidence.exact_transcript, "Exact"); self.assertEqual(evidence.native_sample_rate, 24000); self.assertEqual(evidence.samples, 1200); self.assertEqual(evidence.channel, 0)

    def test_generation_hash_covers_sampling_parameters(self):
        reference = ReferenceEvidence("r", "ref.wav", "a" * 64, 24000, 1, 100, 0, 100, None, "Hello", "en", "Speaker", "L", "test", "1", "b" * 64)
        line = Line("L", "Speaker", "Hello", "Hallo", subtitle_authorized=True)
        common = dict(line=line, reference=reference, target_language="de", model_id="model", model_revision="rev", backend_version="backend", native_sample_rate=24000, guidance_scale=2.0, seed=1, temperature=None, t_shift=None, postprocess_output="none", text_normalization_version="v1")
        first = GenerationRequest(generation_steps=32, **common); second = GenerationRequest(generation_steps=33, **common)
        self.assertNotEqual(first.generation_hash, second.generation_hash)


class V2QATests(unittest.TestCase):
    def _wav(self, directory: str, name: str = "audio.wav") -> Path:
        import soundfile as sf
        path = Path(directory) / name; sf.write(path, np.ones(2400, dtype="float32") * .1, 24000); return path

    def test_final_word_requires_suffix_order(self):
        self.assertFalse(final_word("Warum nicht", "nicht Warum")[0])
        self.assertFalse(final_word("Warum nicht", "Warum")[0])
        self.assertTrue(final_word("Warum nicht", "Warum nicht")[0])

    def test_content_is_ordered_and_english_leak_is_rejected(self):
        self.assertFalse(ordered_content("Was machst du", "du machst Was")[0])
        self.assertFalse(source_language_leak("Was machst du", "Nearby enemies detected", "en", .99, LanguageProfile())[0])

    def test_qa_has_no_fake_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            result = evaluate_candidate_v2(str(path), expected_text="Hallo dort", source_text="Hello there", target_sample_rate=24000, target_frames=2400, transcript=None)
            self.assertFalse(result.passed)
            self.assertEqual(result.gates["content"].status, GateStatus.NOT_RUN)
            self.assertEqual(result.gates["serialization_contract"].status, GateStatus.PASS)
            passed = evaluate_candidate_v2(str(path), expected_text="Hallo dort", source_text="Hello there", target_sample_rate=24000, target_frames=2400, transcript="Hallo dort", language="de", language_probability=.99)
            self.assertTrue(passed.passed)

    def test_unicode_normalization_keeps_german_content_deterministic(self):
        self.assertEqual(ordered_content("Für nächste", "FÜR NÄCHSTE")[0], True)
        self.assertEqual(ordered_content("Für nächste", "Fuer naechste")[0], False)

    def test_dual_asr_marks_disagreement_uncertain_without_relaxing_content(self):
        from dubbing_pipeline.post_qa import persist_audio_atomic
        with tempfile.TemporaryDirectory() as directory:
            path = persist_audio_atomic(Path(directory) / "audio.wav", np.ones(2400, dtype="float32") * .1, 24000)
            class ASR:
                def __init__(self): self.calls = []
                def transcribe(self, _path, *, language=None):
                    self.calls.append(language)
                    return {"text": "Keine Sorge" if language == "de" else "Keine Zorge", "language": "de", "probability": .99}
            backend = ASR(); cache = ASRCache(Path(directory) / "cache")
            evidence = transcribe_dual(backend, path, source_language="en", target_language="de", cache=cache, semantic_key="speech-equivalent")
            self.assertEqual(backend.calls, ["de", None])
            self.assertEqual(evidence.forced_transcript, "Keine Sorge")
            self.assertEqual({item["evidence_family"] for item in evidence.to_dict()["evidence_records"]}, {"WHISPER_ASR"})
            result = evaluate_candidate_v2(
                str(path), expected_text="Keine Sorge", source_text="Don't worry",
                target_sample_rate=24000, target_frames=2400,
                linguistic_evidence=evidence.to_dict(),
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.failure_class.value, "ASR_UNCERTAIN")
            self.assertEqual(result.diagnostics["linguistic_decision"]["status"], "ASR_UNCERTAIN")
            # Reusing the same artifact must not perform either decode again.
            second = transcribe_dual(backend, path, source_language="en", target_language="de", cache=cache, semantic_key="speech-equivalent")
            self.assertEqual(len(backend.calls), 2)
            self.assertTrue(second.forced_target.cache_hit)
            self.assertTrue(second.automatic.cache_hit)
            disk_cache = ASRCache(Path(directory) / "cache")
            third = transcribe_dual(backend, path, source_language="en", target_language="de", cache=disk_cache, semantic_key="speech-equivalent")
            self.assertEqual(len(backend.calls), 2)
            self.assertTrue(third.forced_target.cache_hit)
            transformed = persist_audio_atomic(Path(directory) / "resampled.wav", np.ones(2400, dtype="float32") * .11, 24000)
            transcribe_dual(backend, transformed, source_language="en", target_language="de", cache=cache, semantic_key="speech-equivalent")
            self.assertEqual(len(backend.calls), 2)

    def test_english_audio_forced_as_german_is_language_leak_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            evidence = {
                "target_language": "de",
                "evidence_hashes": ["a" * 64, "b" * 64],
                "forced_target": {"mode": "forced_target", "text": "Dann wurde", "language": "de", "probability": .99},
                "automatic": {"mode": "automatic", "text": "Don't worry", "language": "en", "probability": .99},
            }
            result = evaluate_candidate_v2(
                str(path), expected_text="Keine Sorge", source_text="Don't worry",
                target_sample_rate=24000, target_frames=2400,
                linguistic_evidence=evidence,
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.diagnostics["linguistic_decision"]["status"], "LANGUAGE_LEAK_SUSPECTED")
            self.assertEqual(result.gates["source_language"].status, GateStatus.FAIL)

    def test_two_correlated_whisper_failures_are_uncertain_not_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            evidence = {
                "target_language": "de",
                "evidence_hashes": ["a" * 64, "b" * 64],
                "forced_target": {"text": "falscher Inhalt", "language": "de", "probability": .99},
                "automatic": {"text": "falscher Inhalt", "language": "de", "probability": .99},
            }
            result = evaluate_candidate_v2(str(path), expected_text="Keine Sorge", source_text="Don't worry", target_sample_rate=24000, target_frames=2400, linguistic_evidence=evidence)
            self.assertFalse(result.passed)
            self.assertEqual(result.diagnostics["linguistic_decision"]["status"], "ASR_UNCERTAIN")

    def test_ctc_confirms_phonetic_content_after_whisper_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            evidence = {
                "target_language": "de",
                "evidence_hashes": ["a" * 64, "b" * 64],
                "evidence_records": [{"evidence_family": "WHISPER_ASR", "evidence_hash": "a" * 64}],
                "forced_target": {"text": "Keine Zorge", "language": "de", "probability": .99},
                "automatic": {"text": "Keine Zorge", "language": "de", "probability": .99},
            }
            alignment = {
                "target_score": .90,
                "source_score": .10,
                "margin": .80,
                "target": {"score": .90, "final_anchor_present": True},
                "evidence_records": [{"evidence_id": "c", "evidence_family": "CTC_FORCED_ALIGNER", "backend_id": "test-ctc", "model_id": "de", "model_revision": "1", "mode": "expected_text_alignment", "audio_sha256": "d" * 64, "semantic_key": None, "output": {"score": .90}, "confidence": .90, "evidence_hash": "c" * 64}],
            }
            result = evaluate_candidate_v2(str(path), expected_text="Keine Sorge", source_text="Don't worry", target_sample_rate=24000, target_frames=2400, linguistic_evidence=evidence, alignment_evidence=alignment)
            self.assertFalse(result.passed)
            self.assertEqual(result.diagnostics["linguistic_decision"]["status"], "TARGET_ALIGNMENT_SUPPORT")
            self.assertEqual(result.diagnostics["linguistic_decision"]["evidence_families"], ["CTC_FORCED_ALIGNER", "WHISPER_ASR"])
            self.assertEqual(result.diagnostics["linguistic_decision"]["cross_language_margin"], .80)
            self.assertEqual(result.failure_class.value, "ASR_UNCERTAIN")

    def test_ctc_target_can_overrule_whisper_language_misread(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            evidence = {
                "target_language": "de",
                "evidence_hashes": ["a" * 64, "b" * 64],
                "evidence_records": [{"evidence_family": "WHISPER_ASR", "evidence_hash": "a" * 64}],
                "forced_target": {"text": "Keine Sorge", "language": "de", "probability": .80},
                "automatic": {"text": "Don't worry", "language": "en", "probability": .99},
            }
            alignment = {
                "target_score": .90,
                "source_score": .10,
                "margin": .80,
                "target": {"score": .90, "final_anchor_present": True},
                "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}],
            }
            result = evaluate_candidate_v2(str(path), expected_text="Keine Sorge", source_text="Don't worry", target_sample_rate=24000, target_frames=2400, linguistic_evidence=evidence, alignment_evidence=alignment)
            self.assertFalse(result.passed)
            self.assertEqual(result.diagnostics["linguistic_decision"]["status"], "EVIDENCE_CONFLICT")
            self.assertEqual(result.gates["source_language"].status, GateStatus.FAIL)
            self.assertFalse(result.diagnostics["linguistic_decision"]["calibration_authority"])

    def test_cross_language_ctc_margin_is_diagnostic_only(self):
        base = decide_linguistic_evidence(
            "Keine Sorge", "Don't worry", forced_target={"text": "Keine Sorge", "language": "de", "probability": .99},
            automatic={"text": "Keine Sorge", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(), evidence_hashes=["a" * 64],
        )
        result = apply_independent_evidence(
            base,
            {"target_score": .90, "source_score": .90, "margin": 0.0, "target": {"score": .90, "final_anchor_present": True}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
        )
        self.assertEqual(result.status, "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT")
        self.assertEqual(result.cross_language_margin, 0.0)

    def test_uncalibrated_target_score_never_emits_hard_pass(self):
        base = decide_linguistic_evidence(
            "Keine Sorge", "Don't worry", forced_target={"text": "Keine Sorge", "language": "de", "probability": .99},
            automatic={"text": "Keine Sorge", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
        )
        decision = apply_independent_evidence(
            base,
            {"target_score": .95, "source_score": .10, "target": {"final_anchor_present": True}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
        )
        self.assertEqual(decision.status, "PASS_SCREENED_WITH_ALIGNMENT_SUPPORT")
        self.assertFalse(decision.confirmed)
        self.assertFalse(decision.calibration_authority)

    def test_mismatched_calibration_profile_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, runtime_lock, models_lock = _validated_profile(root, model_revision="wrong-revision")
            base = decide_linguistic_evidence(
                "Keine Sorge", "Don't worry", forced_target={"text": "Keine Sorge", "language": "de", "probability": .99},
                automatic={"text": "Keine Sorge", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
            )
            decision = apply_independent_evidence(
                base,
                {"target_score": .95, "source_score": .10, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
                calibration_authority=True, calibration_profile=profile, calibration_profile_root=root,
                backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
                runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock),
            )
            self.assertEqual(decision.status, "BLOCKED")
            self.assertEqual(decision.calibration_profile_status, "BLOCKED_IDENTITY_MISMATCH")
            self.assertFalse(decision.confirmed)

    def test_matching_calibration_profile_can_confirm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, runtime_lock, models_lock = _validated_profile(root)
            base = decide_linguistic_evidence(
                "Keine Sorge", "Don't worry", forced_target={"text": "Keine Sorge", "language": "de", "probability": .99},
                automatic={"text": "Keine Sorge", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
            )
            target_chars = [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate("KeineSorge")]
            target_metrics = _character_evidence("Keine Sorge", {"char_segments": target_chars}, [])
            decision = apply_independent_evidence(
                base,
                {"target_score": .95, "source_score": .10, "target": {**target_metrics, "char_segments": target_chars}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
                calibration_authority=True, calibration_profile=profile, calibration_profile_root=root,
                backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
                runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock),
            )
            self.assertEqual(decision.status, "PASS_CONFIRMED")
            self.assertTrue(decision.confirmed)
            self.assertEqual(decision.calibration_profile_status, "MATCHED_VALIDATED")
            self.assertIsNotNone(decision.calibrated_target_probability)
            self.assertIsNotNone(decision.calibrated_final_anchor_probability)
            self.assertNotEqual(decision.raw_target_score, decision.calibrated_target_probability)

    def test_inert_calibrator_cannot_grant_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, runtime_lock, models_lock = _validated_profile(root)
            profile["calibrators"]["target"]["engine"] = "external-python"
            base = decide_linguistic_evidence(
                "Keine Sorge", "Don't worry", forced_target={"text": "Keine Sorge", "language": "de", "probability": .99},
                automatic={"text": "Keine Sorge", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
            )
            target_chars = [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate("KeineSorge")]
            target_metrics = _character_evidence("Keine Sorge", {"char_segments": target_chars}, [])
            decision = apply_independent_evidence(
                base,
                {"target_score": .99, "target": {**target_metrics, "char_segments": target_chars}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
                calibration_authority=True, calibration_profile=profile, calibration_profile_root=root,
                backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
                runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock),
            )
            self.assertEqual(decision.status, "BLOCKED")
            self.assertEqual(decision.calibration_profile_status, "BLOCKED_CALIBRATOR_NOT_EXECUTABLE")
            self.assertIsNone(decision.calibrated_target_probability)

    def test_safe_calibrator_import_and_probability_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, _runtime_lock, _models_lock = _validated_profile(root)
            artifact = load_safe_calibrator(
                root / "target_calibrator.json",
                profile["calibrators"]["target"]["artifact_sha256"],
                "char-alignment-v2",
            )
            vector = {name: 0.0 for name in artifact["features"]}
            self.assertGreaterEqual(predict_probability(artifact, vector), 0.0)
            self.assertLessEqual(predict_probability(artifact, vector), 1.0)
            with self.assertRaises(ValueError):
                predict_probability(artifact, {"target_score": 0.99})
            broken = dict(artifact)
            broken["coefficients"] = [float("nan")] + list(artifact["coefficients"])[1:]
            with self.assertRaises(ValueError):
                predict_probability(broken, vector)

    def test_alignment_normalization_is_versioned_and_shared(self):
        self.assertEqual(normalize_alignment_text("geht\u2019s"), normalize_alignment_text("geht's"))
        self.assertEqual(normalize_alignment_text("F\u00fcr n\u00e4chste!"), normalize_alignment_text("F\u00fcrn\u00e4chste"))
        self.assertEqual(AlignmentTextPolicy().version, "alignment-text-normalization-v2")

    def test_sequence_character_alignment_preserves_insertions_deletions_and_apostrophes(self):
        def rows(value: str):
            return [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate(value)]

        apostrophe = _character_evidence("geht's", {"char_segments": rows("geht's")}, [])
        self.assertEqual(apostrophe["native_char_coverage"], 1.0)
        self.assertEqual(apostrophe["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")
        self.assertNotIn("'", apostrophe["unaligned_characters"])

        umlaut = _character_evidence("Für nächste!", {"char_segments": rows("Fürnächste!")}, [])
        self.assertEqual(umlaut["native_char_coverage"], 1.0)
        self.assertEqual(umlaut["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")
        self.assertIn("ü", [row["expected_char"] for row in umlaut["char_segments"]])
        self.assertIn("ä", [row["expected_char"] for row in umlaut["char_segments"]])

        deleted = _character_evidence("Sorge", {"char_segments": rows("Soge")}, [])
        expected_rows = [row for row in deleted["char_segments"] if row.get("expected_index") is not None]
        self.assertEqual(next(row for row in expected_rows if row["expected_char"] == "r")["operation"], "DELETE")
        self.assertEqual([row["operation"] for row in expected_rows if row["expected_char"] in {"g", "e"}], ["MATCH", "MATCH"])

        inserted = _character_evidence("Sorge", {"char_segments": rows("Sorxge")}, [])
        self.assertTrue(any(row.get("operation") == "INSERT" and row.get("observed_char") == "x" for row in inserted["char_segments"]))
        expected_rows = [row for row in inserted["char_segments"] if row.get("expected_index") is not None]
        self.assertEqual([row["operation"] for row in expected_rows if row["expected_char"] in {"g", "e"}], ["MATCH", "MATCH"])
        self.assertEqual(inserted["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")

        interpolated_rows = rows("Sorge")
        interpolated_rows[2]["interpolated"] = True
        interpolated = _character_evidence("Sorge", {"char_segments": interpolated_rows}, [])
        self.assertEqual(interpolated["char_segments"][2]["operation"], "INTERPOLATED")
        self.assertEqual(interpolated["interpolated_count"], 1)
        self.assertTrue(interpolated["alignment_operation_hash"])

        punctuation = _character_evidence("Komm zurück...", {"char_segments": rows("Komm zurück")}, [])
        self.assertEqual(punctuation["native_char_coverage"], 1.0)
        self.assertEqual(punctuation["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")
        cutoff = _character_evidence("Komm zurück", {"char_segments": rows("Komm zurü")}, [])
        self.assertNotEqual(cutoff["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")
        self.assertLess(cutoff["final_anchor_evidence"]["coverage"], 1.0)

    def test_minimal_profile_cannot_grant_authority(self):
        base = decide_linguistic_evidence(
            "Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99},
            automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
        )
        decision = apply_independent_evidence(
            base,
            {"target_score": .95, "source_score": .10, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
            calibration_authority=True,
            calibration_profile={"schema": "generic-dubbing-calibration-profile-v1", "authority": True, "model_id": "fake-german", "model_revision": "test", "target_language": "de", "source_language": "en", "performance_mode": "NEUTRAL"},
            backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
            runtime_lock_sha256="a" * 64, models_lock_sha256="b" * 64,
        )
        self.assertEqual(decision.status, "BLOCKED")
        self.assertEqual(decision.calibration_profile_status, "BLOCKED_SCHEMA")

    def test_calibration_artifact_and_provenance_contract_is_fail_closed(self):
        base = decide_linguistic_evidence(
            "Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99},
            automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
        )
        alignment = {"target_score": .95, "source_score": .10, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, runtime_lock, models_lock = _validated_profile(root)
            common = dict(
                calibration_authority=True, calibration_profile_root=root,
                backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
                runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock),
            )
            cases = [
                ("missing calibrator", lambda value: value["calibrators"]["target"].update({"type": "platt", "artifact_path": str(root / "missing.bin"), "artifact_sha256": "a" * 64}), "BLOCKED_CALIBRATOR_ARTIFACT"),
                ("bad calibrator hash", lambda value: value["calibrators"]["target"].update({"artifact_sha256": "a" * 64}), "BLOCKED_CALIBRATOR_HASH"),
                ("feature schema", lambda value: value["identity"].update({"feature_schema_version": "word-alignment-v1"}), "BLOCKED_FEATURE_SCHEMA"),
                ("runtime provenance", lambda value: value["provenance"].update({"runtime_lock_sha256": "f" * 64}), "BLOCKED_RUNTIME_MODEL_MISMATCH"),
                ("missing threshold", lambda value: value["thresholds"].pop("target_pass_probability"), "BLOCKED_INCOMPLETE_PROFILE"),
                ("draft profile", lambda value: value.update({"status": "DRAFT"}), "BLOCKED_PROFILE_STATUS"),
            ]
            for name, mutate, expected in cases:
                candidate = copy.deepcopy(profile)
                mutate(candidate)
                decision = apply_independent_evidence(base, alignment, calibration_profile=candidate, **common)
                self.assertEqual(decision.status, "BLOCKED", name)
                self.assertEqual(decision.calibration_profile_status, expected, name)

    def test_calibrated_content_without_target_lid_is_held_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._wav(directory)
            profile, runtime_lock, models_lock = _validated_profile(root)
            evidence = {
                "target_language": "de",
                "evidence_records": [{"evidence_family": "WHISPER_ASR"}],
                "forced_target": {"text": "Keine Sorge", "language": "de", "probability": .99},
                "automatic": {"text": "Don't worry", "language": "en", "probability": .99},
            }
            chars = [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate(char for char in "Keine Sorge" if not char.isspace())]
            char_metrics = _character_evidence("Keine Sorge", {"char_segments": chars}, [])
            result = evaluate_candidate_v2(
                str(path), expected_text="Keine Sorge", source_text="Don't worry", target_sample_rate=24000, target_frames=2400,
                linguistic_evidence=evidence,
                alignment_evidence={"target_score": .95, "source_score": .10, "target": {**char_metrics, "char_segments": chars}, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
                calibration_authority=True, calibration_profile=profile, calibration_profile_root=root,
                backend_id="fake-ctc", model_id="fake-german", model_revision="test", performance_mode="NEUTRAL",
                runtime_lock_sha256=sha256_file(runtime_lock), models_lock_sha256=sha256_file(models_lock),
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.gates["source_language"].status, GateStatus.FAIL)
            self.assertEqual(result.failure_class.value, "ASR_UNCERTAIN")

    def test_alignment_score_without_whisper_family_cannot_hard_confirm(self):
        base = decide_linguistic_evidence(
            "Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99},
            automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
        )
        result = apply_independent_evidence(
            replace(base, evidence_families=[]),
            {"target_score": .95, "source_score": .05, "margin": .90, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
        )
        self.assertEqual(result.status, "ALIGNMENT_UNCERTAIN")
        self.assertEqual(result.evidence_families, ["CTC_FORCED_ALIGNER"])

    def test_ctc_cannot_rescue_missing_final_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._wav(directory)
            evidence = {
                "target_language": "de",
                "evidence_hashes": ["a" * 64],
                "evidence_records": [{"evidence_family": "WHISPER_ASR", "evidence_hash": "a" * 64}],
                "forced_target": {"text": "Keine", "language": "de", "probability": .99},
                "automatic": {"text": "Keine", "language": "de", "probability": .99},
            }
            alignment = {
                "target_score": .90,
                "source_score": .10,
                "margin": .80,
                "target": {"score": .90, "final_anchor_present": False},
                "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}],
            }
            result = evaluate_candidate_v2(str(path), expected_text="Keine Sorge", source_text="Don't worry", target_sample_rate=24000, target_frames=2400, linguistic_evidence=evidence, alignment_evidence=alignment)
            self.assertFalse(result.passed)
            self.assertEqual(result.gates["final_word"].status, GateStatus.FAIL)

    def test_alignment_from_different_audio_is_not_fused(self):
        base = decide_linguistic_evidence(
            "Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99},
            automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(),
            evidence_records=[{"evidence_family": "WHISPER_ASR", "audio_sha256": "a" * 64}], audio_sha256="a" * 64,
        )
        result = apply_independent_evidence(
            base,
            {"target_score": .95, "source_score": .05, "margin": .90, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER", "audio_sha256": "b" * 64}]},
        )
        self.assertEqual(result.status, "ALIGNMENT_UNCERTAIN")
        self.assertIn("different audio artifact", result.reason)

    def test_ctc_source_preference_needs_independent_lid_for_confirmed_leak(self):
        base = decide_linguistic_evidence(
            "Keine Sorge", "Don't worry", forced_target={"text": "Dann wurde", "language": "de", "probability": .99},
            automatic={"text": "Don't worry", "language": "en", "probability": .99}, target_language="de", profile=LanguageProfile(), evidence_hashes=["a" * 64],
        )
        alignment = {"target_score": .20, "source_score": .90, "margin": -.70, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]}
        suspected = apply_independent_evidence(base, alignment, source_language="en")
        self.assertEqual(suspected.status, "LANGUAGE_LEAK_STRONG_SUSPICION")
        self.assertFalse(suspected.confirmed)
        confirmed = apply_independent_evidence(base, alignment, lid_evidence={"language": "en", "probability": .95, "record": {"evidence_family": "AUDIO_LANGUAGE_ID"}}, source_language="en")
        self.assertEqual(confirmed.status, "LANGUAGE_LEAK_STRONG_SUSPICION")
        self.assertFalse(confirmed.confirmed)

    def test_ctc_rejects_target_without_inventing_source_leak(self):
        base = decide_linguistic_evidence(
            "Keine Sorge", "Don't worry", forced_target={"text": "falscher Inhalt", "language": "de", "probability": .99},
            automatic={"text": "falscher Inhalt", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(), evidence_hashes=["a" * 64],
        )
        result = apply_independent_evidence(
            base,
            {"target_score": .20, "source_score": .10, "margin": .10, "evidence_records": [{"evidence_family": "CTC_FORCED_ALIGNER"}]},
            source_language="en",
        )
        self.assertEqual(result.status, "LEXICAL_FAILURE_SUSPECTED")
        self.assertFalse(result.confirmed)

    def test_missing_alignment_family_is_a_hold(self):
        base = decide_linguistic_evidence(
            "Hallo", "Hello", forced_target={"text": "Hallo", "language": "de", "probability": .99},
            automatic={"text": "Hallo", "language": "de", "probability": .99}, target_language="de", profile=LanguageProfile(), evidence_hashes=["a" * 64],
        )
        self.assertEqual(base.status, "PASS_SCREENED")
        self.assertEqual(apply_independent_evidence(base, None).status, "ALIGNER_NOT_APPLICABLE")

    def test_alignment_cache_runs_each_hypothesis_once(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"; sf.write(path, np.ones(2400, dtype="float32") * .1, 24000)
            class Aligner:
                evidence_family = EvidenceFamily.CTC_FORCED_ALIGNER; backend_id = "ctc"; model_id = "de"; model_revision = "1"
                def __init__(self): self.calls = []
                def align(self, _path, *, text, language):
                    self.calls.append((text, language)); return {"score": .9 if language == "de" else .1, "coverage": 1.0, "final_anchor_present": True}
            backend = Aligner(); cache = AlignmentCache(Path(directory) / "cache", "ctc", "de", "1")
            first = contrastive_align(backend, path, target_text="Keine Sorge", source_text="Don't worry", cache=cache)
            second = contrastive_align(backend, path, target_text="Keine Sorge", source_text="Don't worry", cache=cache)
            self.assertEqual(len(backend.calls), 2)
            self.assertTrue(second.target.cache_hit and second.source.cache_hit)
            self.assertEqual(first.margin, .8)

    def test_mfa_json_word_tier_is_normalised(self):
        rows = _extract_mfa_words({"tiers": {"words": {"entries": [[0.0, 0.2, "Keine"], [0.2, 0.4, "Sorge"]]}}})
        self.assertEqual([row["word"] for row in rows], ["Keine", "Sorge"])
        self.assertEqual(rows[-1]["end"], .4)

    def test_whisperx_adapter_passes_segments_not_outer_result(self):
        from dubbing_pipeline.alignment import WhisperXCTCAligner
        class FakeWhisperX:
            def load_audio(self, _path):
                return np.zeros(16000, dtype="float32")
            def align(self, segments, _model, _metadata, _audio, _device, **_kwargs):
                if not isinstance(segments, list):
                    raise AssertionError("outer result object passed to WhisperX.align")
                return {"word_segments": [{"start": 0.1, "end": 0.3, "word": "Sorge", "score": .9, "chars": [{"char": char, "start": 0.1 + index * .03, "end": 0.1 + (index + 1) * .03, "score": .9} for index, char in enumerate("Sorge")]}]}
        adapter = WhisperXCTCAligner(device="cpu")
        fake = FakeWhisperX()
        adapter._model = lambda _language: (fake, object(), {})
        result = adapter.align("candidate.wav", text="Sorge", language="de")
        self.assertEqual(result["final_anchor_evidence"]["status"], "FINAL_ANCHOR_EVIDENCE_COLLECTED")
        self.assertEqual(result["final_anchor_evidence"]["authority"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["native_char_coverage"], 1.0)
        self.assertEqual(len(result["char_segments"]), 5)
        self.assertNotIn("final_anchor_present", result)

    def test_whisperx_official_segment_chars_are_used_before_word_fallback(self):
        segment_chars = [{"char": char, "start": index * .04, "end": (index + 1) * .04, "score": .8} for index, char in enumerate("Sorge")]
        value = {
            "segments": [{"start": 0.0, "end": .2, "chars": segment_chars}],
            "word_segments": [{"word": "Sorge", "chars": [{"char": "X", "start": 0.0, "end": .1, "score": .1}]}],
        }
        rows = _raw_char_segments(value, list(value["word_segments"]))
        self.assertEqual("".join(row["char"] for row in rows), "Sorge")
        self.assertEqual(rows[0]["start"], 0.0)

    def test_speechbrain_log_score_is_exponentiated(self):
        from dubbing_pipeline.alignment import SpeechBrainVoxLingua107
        class FakeClassifier:
            def classify_file(self, _path):
                return ([[0.0]], [-0.2231435513142097], [0], ["en"])
        adapter = SpeechBrainVoxLingua107()
        adapter._classifier = FakeClassifier()
        value = adapter.detect("audio.wav")
        self.assertAlmostEqual(value["probability"], .8, places=6)
        self.assertLess(value["log_probability"], 0.0)

    def test_line_summary_does_not_use_first_candidate_as_authority(self):
        class Candidate:
            def __init__(self, value): self.candidate_id = value
        class Audit:
            def __init__(self, status): self.diagnostics = {"linguistic_decision": {"status": status, "evidence_families": ["WHISPER_ASR"], "missing_tokens": [], "final_anchor_present": True}, "asr": {"evidence_hashes": []}}; self.artifact_path = "x.wav"; self.gates = {}
        row = {}
        options = [{"candidate": Candidate("c1"), "raw_audit": Audit("ASR_UNCERTAIN"), "mounted_audit": Audit("ASR_UNCERTAIN"), "eligible": False, "alignment_status": "ALIGNMENT_UNCERTAIN"}, {"candidate": Candidate("c2"), "raw_audit": Audit("PASS_SCREENED"), "mounted_audit": Audit("PASS_CONFIRMED"), "eligible": True, "alignment_status": "PASS_CONFIRMED"}]
        _line_linguistic_summary(row, options, expected_text="Hallo")
        self.assertEqual(row["line_linguistic_summary"]["eligible_count"], 1)
        self.assertEqual(row["candidate_linguistic_decisions"][1]["status"], "PASS_CONFIRMED")

    def test_line_summary_holds_when_independent_aligner_is_unavailable(self):
        class Candidate:
            candidate_id = "c1"
        class Audit:
            artifact_path = "c1.wav"
            gates = {}
            diagnostics = {"linguistic_decision": {"status": "PASS_SCREENED", "evidence_families": ["WHISPER_ASR"], "missing_tokens": [], "final_anchor_present": True}, "asr": {"evidence_hashes": []}}
        row = {}
        _line_linguistic_summary(
            row,
            [{"candidate": Candidate(), "raw_audit": Audit(), "mounted_audit": Audit(), "eligible": False, "alignment_status": "ALIGNER_NOT_APPLICABLE"}],
            expected_text="Hallo",
        )
        self.assertEqual(row["candidate_linguistic_decisions"][0]["status"], "ALIGNER_NOT_APPLICABLE")

    def test_whisperx_escalation_is_only_a_serializable_request(self):
        request = prepare_whisperx_escalation("candidate.wav", candidate_id="line:r1:t1", expected_text="Keine Sorge", source_text="Don't worry", evidence_hashes=["a" * 64])
        self.assertEqual(request.to_dict()["status"], "PENDING")
        self.assertEqual(request.to_dict()["backend"], "whisperx_or_mfa")
        self.assertEqual(request.to_dict()["candidate_id"], "line:r1:t1")


class V2PostTransformTests(unittest.TestCase):
    def test_post_transform_audit_is_fail_closed_and_reopens_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = persist_audio_atomic(Path(directory) / "processed.wav", np.ones(2400, dtype="float32") * .1, 24000)
            missing_asr = audit_candidate_stage(
                path,
                stage="PROCESSED_QA",
                expected_text="Hallo",
                target_sample_rate=24000,
                target_frames=2400,
                channels=1,
                transcript=None,
            )
            self.assertFalse(missing_asr.passed)
            self.assertEqual(missing_asr.gates["content"].status, GateStatus.NOT_RUN)
            good = audit_candidate_stage(
                path,
                stage="SERIALIZED_QA",
                expected_text="Hallo",
                target_sample_rate=24000,
                target_frames=2400,
                channels=1,
                transcript="Hallo",
                language="de",
                language_probability=.99,
            )
            self.assertTrue(good.passed)
            self.assertEqual(good.artifact_sha256, __import__("dubbing_pipeline.hashing", fromlist=["sha256_file"]).sha256_file(path))

    def test_scene_audit_checks_protected_and_untouched_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.column_stack([np.ones(2400, dtype="float32") * .1, np.ones(2400, dtype="float32") * .2])
            target = source.copy(); target[100:200, 0] = 0
            path = persist_audio_atomic(Path(directory) / "scene.wav", target, 24000)
            failed = audit_scene_stage(path, expected_sample_rate=24000, expected_frames=2400, expected_channels=2, protected_intervals_ok=False, untouched_channels_ok=True)
            self.assertFalse(failed.passed)
            self.assertEqual(failed.gates["preserved_intervals"].status, GateStatus.FAIL)
            self.assertEqual(failed.gates["scene_clipping"].status, GateStatus.PASS)
            self.assertEqual(failed.gates["scene_peak"].status, GateStatus.PASS)


class V2MountDeployTests(unittest.TestCase):
    def test_mount_preserves_channels_and_effort(self):
        stem = np.zeros((24000, 2), dtype="float32"); stem[:2400, 0] = .2; stem[:, 1] = .07; stem[12000:13000, 0] = .15
        generated = np.zeros(2400, dtype="float32"); generated[100:1100] = .3
        window = DeliveryWindow("scene", "line", 0, 24000, 2400, 12000, ((0, 500),), 13000, 0, "tb", "owner")
        result, metrics = mount_surgical(stem, generated, 24000, window, 24000, empalme_b=True)
        self.assertTrue(np.array_equal(stem[:, 1], result[:, 1]))
        self.assertTrue(np.array_equal(stem[:500, 0], result[:500, 0]))
        self.assertTrue(np.array_equal(stem[13000:, 0], result[13000:, 0]))
        self.assertEqual(metrics.preserved_hash_before, metrics.preserved_hash_after)

    def test_deploy_rollback_removes_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stage = root / "stage"; runtime = root / "runtime"; backups = root / "backups"; stage.mkdir(); runtime.mkdir()
            (stage / "one.bin").write_bytes(b"one"); (stage / "two.bin").write_bytes(b"two")
            entries = [PackageEntry(stage / "one.bin", Path("one.bin")), PackageEntry(stage / "two.bin", Path("two.bin"))]
            def fail(index, _destination):
                if index == 1: raise RuntimeError("injected")
            with self.assertRaises(RuntimeError): deploy_atomic_v2(entries, stage, runtime, backups, failure_injector=fail, lab_mode=False)
            self.assertFalse((runtime / "one.bin").exists()); self.assertFalse((runtime / "two.bin").exists())
            with self.assertRaises(DeploymentError): stage_files_v2([PackageEntry(stage / "one.bin", Path("..") / "escape")], root / "badstage")


class V2SchedulerTests(unittest.TestCase):
    class FakeCTC:
        evidence_family = "CTC_FORCED_ALIGNER"
        backend_id = "fake-ctc"
        model_id = "fake-german"
        model_revision = "test"
        def align(self, _path, *, text, language):
            chars = [{"char": char, "start": index * .02, "end": (index + 1) * .02, "score": .9} for index, char in enumerate(char for char in text if not char.isspace())]
            return {"score": .90 if language == "de" else .10, "coverage": 1.0, "words": [{"word": token, "score": .9} for token in text.split()], "char_segments": chars}

    def _calibrated_qa(self, root: Path) -> tuple[QAConfig, Path, Path]:
        profile, runtime_lock, models_lock = _validated_profile(root)
        return QAConfig(
            calibration_authority=True,
            calibration_profile=profile, calibration_profile_root=root,
        ), runtime_lock, models_lock

    def test_qa_is_after_initial_cohort(self):
        order = []; telemetry = TelemetryCollector("test-run")
        items = ["a", "b", "c"]
        def generate(values, round_index): order.append(("generate", round_index, tuple(values))); return {item: [item] for item in values}
        class Result:
            passed = True; failure_class = None
        def evaluate(item): order.append(("qa", item)); return Result()
        report = run_cohorts(items, item_id=lambda value: value, generate=generate, evaluate=evaluate, telemetry=telemetry)
        self.assertEqual(order[0][0], "generate"); self.assertEqual(order[1][0], "qa"); self.assertEqual(report.retry_ids, [])
        self.assertNotIn("RUNTIME_SMOKE", report.phases)
        self.assertNotIn("MOUNT_SCENES", report.phases)

    def test_asr_uncertain_is_held_without_regeneration(self):
        class Result:
            passed = False
            failure_class = __import__("dubbing_pipeline.contracts", fromlist=["FailureClass"]).FailureClass.ASR_UNCERTAIN
        generated = []
        def generate(values, round_index):
            generated.append(round_index)
            return {value: [value] for value in values}
        report = run_cohorts(["uncertain"], item_id=lambda value: value, generate=generate, evaluate=lambda _value: Result())
        self.assertEqual(generated, [1])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.blockers[0]["reason"], "ASR_UNCERTAIN_HOLD")

    def test_lexical_suspicion_is_held_without_regeneration(self):
        class Result:
            passed = False
            failure_class = __import__("dubbing_pipeline.contracts", fromlist=["FailureClass"]).FailureClass.ASR_UNCERTAIN
        generated = []
        def generate(values, round_index):
            generated.append(round_index)
            return {value: [value] for value in values}
        report = run_cohorts(["lexical-suspect"], item_id=lambda value: value, generate=generate, evaluate=lambda _value: Result())
        self.assertEqual(generated, [1])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.blockers[0]["reason"], "ASR_UNCERTAIN_HOLD")

    def test_blocked_calibration_is_deterministic_and_not_a_retry(self):
        class Result:
            passed = False
            failure_class = __import__("dubbing_pipeline.contracts", fromlist=["FailureClass"]).FailureClass.DETERMINISTIC_CALIBRATION
        generated = []
        def generate(values, round_index):
            generated.append(round_index)
            return {value: [value] for value in values}
        report = run_cohorts(["blocked-profile"], item_id=lambda value: value, generate=generate, evaluate=lambda _value: Result())
        self.assertEqual(generated, [1])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.blockers[0]["reason"], "DETERMINISTIC_CALIBRATION_HOLD")

    def test_fmv_scene_uses_surgical_mount(self):
        import soundfile as sf
        class Backend:
            def generate_batch(self, payload):
                audio = np.zeros(2400, dtype="float32"); audio[100:1100] = .05
                return [audio for _ in payload]
        class ASR:
            def transcribe(self, _path): return {"text": "Hallo", "language": "de", "probability": .99}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ref = root / "ref.wav"; stem = root / "stem.wav"
            sf.write(ref, np.ones(2400, dtype="float32") * .04, 24000)
            original = np.column_stack([np.ones(24000, dtype="float32") * .02, np.ones(24000, dtype="float32") * .07]); sf.write(stem, original, 24000)
            qa, runtime_lock, models_lock = self._calibrated_qa(root)
            config = PipelineConfig(project_root=root, output_root=root / "out", cache_root=root / "cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root / "sandbox", initial_takes=1, retry_takes=0, qa=qa, runtime_lock=runtime_lock, models_lock=models_lock)
            line = __import__("dubbing_pipeline.models", fromlist=["Line"]).Line("L1", "A", "Hello", "Hallo", 0, 1, topology="EMBEDDED_FMV", subtitle_authorized=True, reference_audio=str(ref), movie_identity_verified=True, card_identity_verified=True, card_timebase_verified=True, preserved_source_intervals=[{"start": 0.0, "end": 0.05}], source_resume=.7, speech_start=.1, speech_end=.7)
            report = run_scene_v2(Scene("S", "EMBEDDED_FMV", [line], source_stem=str(stem), movie_identity_verified=True), config, runtime=GenerationRuntimeV2(Backend(), backend_version="test"), asr=ASR(), alignment_backend=self.FakeCTC())
            self.assertTrue(report["pass"]); self.assertTrue(Path(report["mounted_output"]).is_file())
            self.assertEqual(report["performance_by_line"]["L1"]["mode"], "NEUTRAL")
            self.assertTrue(report["qa_routes"]["L1"])
            self.assertTrue(report["model_pool"]["loaded"])
            self.assertTrue((root / "out" / "S" / "state").exists())

    def test_mfa_is_invoked_only_when_cost_route_requests_it(self):
        import soundfile as sf
        class Backend:
            def generate_batch(self, payload):
                audio = np.zeros(2400, dtype="float32"); audio[100:1100] = .05
                return [audio for _ in payload]
        class ASR:
            def transcribe(self, _path): return {"text": "Hallo", "language": "de", "probability": .99}
        class LowCTC(self.FakeCTC):
            def align(self, _path, *, text, language): return {"score": .10, "coverage": 0.2, "final_anchor_present": False}
        class MFA:
            def __init__(self): self.calls=0
            def align(self, _path, *, text, language): self.calls += 1; return {"score": .2, "coverage": .2, "final_anchor_present": False, "words": []}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); ref=root/"ref.wav"; sf.write(ref, np.ones(2400,dtype="float32")*.04,24000)
            qa, runtime_lock, models_lock=self._calibrated_qa(root)
            config=PipelineConfig(project_root=root, output_root=root/"out", cache_root=root/"cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root/"sandbox", initial_takes=1, retry_takes=0, qa=qa, runtime_lock=runtime_lock, models_lock=models_lock)
            line=Line("L1","A","Hello","Hallo",0,1,topology="LINE_SEPARATED",subtitle_authorized=True,reference_audio=str(ref),metadata={"mfa_requested":True})
            mfa=MFA(); report=run_scene_v2(Scene("S","LINE_SEPARATED",[line]),config,runtime=GenerationRuntimeV2(Backend(),backend_version="test"),asr=ASR(),alignment_backend=LowCTC(),mfa_backend=mfa)
            self.assertGreater(mfa.calls,0); self.assertGreater(report["mfa"]["executed"],0)

    def test_final_selection_happens_after_processed_and_mounted_qa(self):
        import soundfile as sf

        class Backend:
            def generate_batch(self, payload):
                long_audio = np.zeros(48000, dtype="float32"); long_audio[100:47000] = .05
                short_audio = np.zeros(2400, dtype="float32"); short_audio[100:1100] = .05
                return [long_audio, short_audio][:len(payload)]

        class ASR:
            def transcribe(self, _path): return {"text": "Hallo", "language": "de", "probability": .99}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ref = root / "ref.wav"; stem = root / "stem.wav"
            sf.write(ref, np.ones(2400, dtype="float32") * .04, 24000)
            original = np.column_stack([np.ones(24000, dtype="float32") * .02, np.ones(24000, dtype="float32") * .07]); sf.write(stem, original, 24000)
            qa, runtime_lock, models_lock = self._calibrated_qa(root)
            config = PipelineConfig(project_root=root, output_root=root / "out", cache_root=root / "cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root / "sandbox", fmv_initial_takes=2, fmv_retry_takes=0, seed=11, qa=qa, runtime_lock=runtime_lock, models_lock=models_lock)
            line = __import__("dubbing_pipeline.models", fromlist=["Line"]).Line("L1", "A", "Hello", "Hallo", 0, 1, topology="EMBEDDED_FMV", subtitle_authorized=True, reference_audio=str(ref), movie_identity_verified=True, card_identity_verified=True, card_timebase_verified=True, preserved_source_intervals=[{"start": 0.0, "end": 0.05}], source_resume=.7, speech_start=.1, speech_end=.7)
            report = run_scene_v2(Scene("S", "EMBEDDED_FMV", [line], source_stem=str(stem), movie_identity_verified=True), config, runtime=GenerationRuntimeV2(Backend(), backend_version="test"), asr=ASR(), alignment_backend=self.FakeCTC())
            self.assertTrue(report["pass"], report)
            self.assertEqual(report["lines"][0]["candidate_id"], "L1:r1:t2")
            self.assertEqual(report["lines"][0]["status"], "FINAL_PASS")
            self.assertTrue(report["scene_qa"]["passed"])
            self.assertEqual(report["stage_evidence"]["SCENE_QA"]["status"], "EXECUTED")
            self.assertTrue(Path(report["mounted_output"]).is_file())

    def test_second_screened_candidate_is_not_hidden_by_first_uncertain_candidate(self):
        import soundfile as sf
        class Backend:
            def generate_batch(self, payload):
                outputs = []
                for index, _item in enumerate(payload):
                    audio = np.zeros(2400, dtype="float32"); audio[100:1100] = .05 + index * .001
                    outputs.append(audio)
                return outputs
        class ASR:
            def transcribe(self, path, *, language=None):
                path_text = str(path)
                correct = "t02" in path_text or "_t2" in path_text
                return {"text": "Hallo" if correct else "falscher Inhalt", "language": "de", "probability": .99}
        class SelectiveCTC(self.FakeCTC):
            def align(self, path, *, text, language):
                if "t01" in str(path) or "_t1" in str(path):
                    return {"score": .20 if language == "de" else .10, "coverage": 1.0, "final_anchor_present": False}
                return super().align(path, text=text, language=language)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ref = root / "ref.wav"
            sf.write(ref, np.ones(2400, dtype="float32") * .04, 24000)
            qa, runtime_lock, models_lock = self._calibrated_qa(root)
            config = PipelineConfig(project_root=root, output_root=root / "out", cache_root=root / "cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root / "sandbox", initial_takes=2, retry_takes=0, seed=11, qa=qa, runtime_lock=runtime_lock, models_lock=models_lock)
            line = Line("L1", "A", "Hello", "Hallo", 0, 1, topology="LINE_SEPARATED", subtitle_authorized=True, reference_audio=str(ref))
            report = run_scene_v2(Scene("S", "LINE_SEPARATED", [line]), config, runtime=GenerationRuntimeV2(Backend(), backend_version="test"), asr=ASR(), alignment_backend=SelectiveCTC())
            self.assertTrue(report["pass"], report)
            self.assertEqual(report["lines"][0]["candidate_id"], "L1:r1:t2")
            decisions = report["lines"][0]["candidate_linguistic_decisions"]
            self.assertEqual({item["status"] for item in decisions}, {"ASR_UNCERTAIN", "PASS_CONFIRMED"})
            self.assertEqual(report["lines"][0]["line_linguistic_summary"]["eligible_count"], 1)
            self.assertTrue(any(item["outcome"] == "HOLD_NO_TTS" for item in report["repair_attempts"]))


if __name__ == "__main__":
    unittest.main()
