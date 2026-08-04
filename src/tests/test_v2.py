from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dubbing_pipeline.contracts import ContractError, DeliveryWindow, GateStatus, ReferenceEvidence
from dubbing_pipeline.contracts.manifest import validate_manifest_value
from dubbing_pipeline.deploy_v2 import DeploymentError, PackageEntry, deploy_atomic_v2, stage_files_v2
from dubbing_pipeline.hashing import atomic_json, contract_hash
from dubbing_pipeline.models import Line
from dubbing_pipeline.montage import mount_surgical
from dubbing_pipeline.qa_v2 import LanguageProfile, evaluate_candidate_v2, final_word, ordered_content, source_language_leak
from dubbing_pipeline.scheduler import run_cohorts
from dubbing_pipeline.telemetry import TelemetryCollector
from dubbing_pipeline.config import PipelineConfig
from dubbing_pipeline.generation_v2 import GenerationRuntimeV2
from dubbing_pipeline.generation_v2 import GenerationRequest
from dubbing_pipeline.models import Scene
from dubbing_pipeline.orchestration_v2 import run_scene_v2
from dubbing_pipeline.post_qa import audit_candidate_stage, audit_scene_stage, persist_audio_atomic
from dubbing_pipeline.reference import materialize_reference


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
            config = PipelineConfig(project_root=root, output_root=root / "out", cache_root=root / "cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root / "sandbox", initial_takes=1, retry_takes=0)
            line = __import__("dubbing_pipeline.models", fromlist=["Line"]).Line("L1", "A", "Hello", "Hallo", 0, 1, topology="EMBEDDED_FMV", subtitle_authorized=True, reference_audio=str(ref), movie_identity_verified=True, card_identity_verified=True, card_timebase_verified=True, preserved_source_intervals=[{"start": 0.0, "end": 0.05}], source_resume=.7, speech_start=.1, speech_end=.7)
            report = run_scene_v2(Scene("S", "EMBEDDED_FMV", [line], source_stem=str(stem), movie_identity_verified=True), config, runtime=GenerationRuntimeV2(Backend(), backend_version="test"), asr=ASR())
            self.assertTrue(report["pass"]); self.assertTrue(Path(report["mounted_output"]).is_file())

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
            config = PipelineConfig(project_root=root, output_root=root / "out", cache_root=root / "cache", sample_rate=24000, native_sample_rate=24000, lab_mode=True, sandbox_root=root / "sandbox", fmv_initial_takes=2, fmv_retry_takes=0, seed=11)
            line = __import__("dubbing_pipeline.models", fromlist=["Line"]).Line("L1", "A", "Hello", "Hallo", 0, 1, topology="EMBEDDED_FMV", subtitle_authorized=True, reference_audio=str(ref), movie_identity_verified=True, card_identity_verified=True, card_timebase_verified=True, preserved_source_intervals=[{"start": 0.0, "end": 0.05}], source_resume=.7, speech_start=.1, speech_end=.7)
            report = run_scene_v2(Scene("S", "EMBEDDED_FMV", [line], source_stem=str(stem), movie_identity_verified=True), config, runtime=GenerationRuntimeV2(Backend(), backend_version="test"), asr=ASR())
            self.assertTrue(report["pass"])
            self.assertEqual(report["lines"][0]["candidate_id"], "L1:r1:t2")
            self.assertEqual(report["lines"][0]["status"], "FINAL_PASS")
            self.assertTrue(report["scene_qa"]["passed"])
            self.assertEqual(report["stage_evidence"]["SCENE_QA"]["status"], "EXECUTED")
            self.assertTrue(Path(report["mounted_output"]).is_file())


if __name__ == "__main__":
    unittest.main()
