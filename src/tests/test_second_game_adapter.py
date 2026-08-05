from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
from dubbing_pipeline.benchmark import BenchmarkManifest, run_benchmark, trusted_runner_identity, validate_manifest
from adapters.second_game_template import SecondGameAdapter

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
            for path in files.values(): path.write_bytes(path.name.encode())
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

if __name__=="__main__": unittest.main()
