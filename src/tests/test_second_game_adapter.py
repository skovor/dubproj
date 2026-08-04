from __future__ import annotations
import unittest
from dubbing_pipeline.benchmark import BenchmarkManifest, run_benchmark
from adapters.second_game_template import SecondGameAdapter

class FinalValidationTests(unittest.TestCase):
    def test_manifest_identity_and_rate(self):
        manifest=BenchmarkManifest("b",("l1",),("audio",),("ref",),"models","runtime","profile","config","commit","LINE_SEPARATED",False); self.assertEqual(len(manifest.digest()),64)
        result=run_benchmark(manifest,lambda *_:{"status":"PASS"},require_files=False); self.assertEqual(result.passed,1); self.assertGreater(result.lines_per_minute,0)
    def test_second_game_is_explicit(self):
        result=SecondGameAdapter("other").validate({"scenes":["s"]}); self.assertTrue(result["valid"]); self.assertTrue(result["independent_adapter"])

if __name__=="__main__": unittest.main()
