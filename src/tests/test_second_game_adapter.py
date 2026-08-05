from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
from dubbing_pipeline.benchmark import BenchmarkManifest, build_invocation_receipt, run_benchmark, trusted_runner_identity, validate_manifest, verify_benchmark_row, _ast_calls_run_scene
from dubbing_pipeline.hashing import contract_hash, sha256_file
from adapters.second_game_template import SecondGameAdapter
from dubbing_pipeline.attestation import sign_attestation, verify_attestation, subject_digest

class FinalValidationTests(unittest.TestCase):
    def test_manifest_identity_and_rate(self):
        manifest=BenchmarkManifest("b",("l1",),("audio",),("ref",),"models","runtime","profile","config","commit","LINE_SEPARATED",False); self.assertEqual(len(manifest.digest()),64)
        result=run_benchmark(manifest,lambda *_:{"status":"PASS"},require_files=False); self.assertEqual(result.passed,1); self.assertGreater(result.lines_per_minute,0)
    def test_second_game_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); audio=root/"audio.wav"; reference=root/"reference.wav"; audio.write_bytes(b"audio"); reference.write_bytes(b"reference")
            import hashlib
            result=SecondGameAdapter("other").validate({"scenes":[{"scene_id":"s","game_id":"other","audio_path":str(audio),"reference_path":str(reference),"audio_sha256":hashlib.sha256(audio.read_bytes()).hexdigest(),"reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),"timing":{"start":0,"end":1},"extraction_status":"VERIFIED"}]})
            self.assertTrue(result["valid"]); self.assertTrue(result["independent_adapter"]); self.assertTrue(result["content_verified"])
    def test_manifest_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            files={name: root/name for name in ("audio.wav","ref.wav","models.lock","runtime.lock")}
            import soundfile as sf
            import numpy as np
            sf.write(files["audio.wav"], np.zeros(2400, dtype="float32"), 24000)
            sf.write(files["ref.wav"], np.ones(2400, dtype="float32") * .01, 24000)
            files["models.lock"].write_bytes(b"models-lock")
            files["runtime.lock"].write_bytes(b"runtime-lock")
            manifest=BenchmarkManifest.from_paths(benchmark_id="b",line_ids=("l1",),audio_paths=(str(files["audio.wav"]),),reference_paths=(str(files["ref.wav"]),),model_lock=str(files["models.lock"]),runtime_lock=str(files["runtime.lock"]),calibration_profile=None,config_hash="c",commit="d",topology="LINE_SEPARATED",real_audio=True)
            from dubbing_pipeline.benchmark import validate_manifest
            self.assertTrue(validate_manifest(manifest)["valid"])
            files["audio.wav"].write_bytes(b"changed")
            self.assertFalse(validate_manifest(manifest)["valid"])

    def test_trusted_benchmark_rejects_fixed_runner(self):
        manifest = BenchmarkManifest("b", ("l1",), ("audio",), ("ref",), "models", "runtime", None, "config", "commit", "LINE_SEPARATED", True)
        with self.assertRaises(ValueError):
            run_benchmark(manifest, lambda *_: {"status": "PASS"}, require_files=False, require_trusted_runner=True)
        self.assertFalse(validate_manifest(manifest, require_files=False)["real_audio"])

    def test_manifest_rejects_arbitrary_wav_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); audio=root/"audio.wav"; reference=root/"ref.wav"; models=root/"models.lock"; runtime=root/"runtime.lock"
            audio.write_bytes(b"not-a-wave"); reference.write_bytes(b"also-not-a-wave"); models.write_bytes(b"m"); runtime.write_bytes(b"r")
            manifest=BenchmarkManifest.from_paths(benchmark_id="b", line_ids=("l1",), audio_paths=(str(audio),), reference_paths=(str(reference),), model_lock=str(models), runtime_lock=str(runtime), calibration_profile=None, config_hash="c", commit="d", topology="LINE_SEPARATED")
            self.assertFalse(validate_manifest(manifest)["valid"])

    def test_runner_ast_ignores_docstrings_and_dead_branches(self):
        self.assertFalse(_ast_calls_run_scene('''def runner():\n    "run_scene_v2"\n    return {}\n'''))
        self.assertFalse(_ast_calls_run_scene('''def runner():\n    if False:\n        return run_scene_v2()\n    return {}\n'''))
        self.assertTrue(_ast_calls_run_scene('''def runner():\n    return run_scene_v2()\n'''))

    def test_report_output_pair_is_reopened_and_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/"output.wav"; report_path=root/"report.json"
            import soundfile as sf
            import numpy as np
            sf.write(output, np.ones(2400, dtype="float32") * .01, 24000)
            report={"scene_id":"s","run_id":"r","lines":[{"id":"l1","status":"FINAL_PASS","output":str(output)}]}
            report["invocation_receipt"] = build_invocation_receipt(report, code_commit="a" * 40)
            report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
            row={"status":"FINAL_PASS","report_path":str(report_path),"output_path":str(output),"report_sha256":sha256_file(report_path),"output_sha256":sha256_file(output),"report_contract_hash":contract_hash("benchmark-report-v1", {"line_id":"l1","report":report}, files=[output])}
            self.assertTrue(verify_benchmark_row("l1", row)["valid"])
            report["lines"][0]["status"]="FAIL"; report_path.write_text(__import__("json").dumps(report), encoding="utf-8"); row["report_sha256"]=sha256_file(report_path)
            self.assertFalse(verify_benchmark_row("l1", row)["valid"])

    def test_fabricated_second_game_json_cannot_be_promoted(self):
        from scripts.promote_branch import execute_second_game_adapter
        result=execute_second_game_adapter({"valid":True,"independent_adapter":True,"content_verified":True}, Path(__file__).resolve().parents[1])
        self.assertFalse(result["valid"])

    def test_benchmark_boolean_claim_is_not_an_attestation(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        import base64
        private = Ed25519PrivateKey.generate(); public = base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii"); private_b64 = base64.b64encode(private.private_bytes_raw()).decode("ascii")
        subject = {"schema": "benchmark-attestation-subject-v1", "manifest_digest": "m", "code_commit": "a" * 40, "benchmark_payload_sha256": "b" * 64}
        signed = sign_attestation(subject, private_b64, key_id="ci")
        self.assertTrue(verify_attestation(signed, public, expected_subject=subject, expected_commit="a" * 40))
        self.assertFalse(verify_attestation({"verified": True}, public, expected_subject=subject, expected_commit="a" * 40))

if __name__=="__main__": unittest.main()
