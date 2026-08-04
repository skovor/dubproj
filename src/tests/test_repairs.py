from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.attempts import AttemptStore
from dubbing_pipeline.repair import FailureCause, apply_repair
from dubbing_pipeline.repair_planner import plan_repairs

class RepairTests(unittest.TestCase):
    def test_uncertain_never_calls_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=AttemptStore(Path(tmp)/"a.sqlite"); action=plan_repairs(FailureCause.ASR_UNCERTAIN)[0]; called=[]; result=apply_repair(action,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=lambda _: called.append(1)); self.assertEqual(result.status,"HOLD_NO_TTS"); self.assertFalse(called); store.close()
    def test_signature_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=AttemptStore(Path(tmp)/"a.sqlite"); action=plan_repairs(FailureCause.SEAM_FAIL)[0]; first=apply_repair(action,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=lambda _: {"status":"PASS"}); second=apply_repair(action,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=lambda _: {"status":"PASS"}); self.assertEqual(first.status,"PASS"); self.assertEqual(second.status,"DUPLICATE_ATTEMPT"); store.close()
    def test_causal_actions_are_bounded(self):
        self.assertEqual(plan_repairs(FailureCause.LANGUAGE_LEAK_CONFIRMED)[0].max_attempts,2); self.assertEqual(plan_repairs(FailureCause.DETERMINISTIC_CALIBRATION)[0].max_attempts,0)

if __name__=="__main__": unittest.main()
