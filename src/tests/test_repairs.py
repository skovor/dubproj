from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from dubbing_pipeline.attempts import AttemptStore
from dubbing_pipeline.repair import FailureCause, apply_repair, re_audit_repair
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
    def test_missing_executor_is_blocked_and_causal_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=AttemptStore(Path(tmp)/"a.sqlite")
            action=plan_repairs(FailureCause.LANGUAGE_LEAK_CONFIRMED)[0]
            first=apply_repair(action,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=None)
            self.assertEqual(first.status,"BLOCKED_NO_EXECUTOR")
            varied=action.__class__(action.strategy,action.cause,{**action.parameters,"variant":1},action.allows_tts,action.max_attempts,action.rationale)
            second=apply_repair(varied,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=lambda _: {"status":"PASS"})
            self.assertEqual(second.status,"PASS")
            exhausted=action.__class__(action.strategy,action.cause,{**action.parameters,"variant":2},action.allows_tts,action.max_attempts,action.rationale)
            self.assertEqual(apply_repair(exhausted,line_id="l",input_audio_sha256="a"*64,reference_sha256=None,store=store,executor=lambda _: {"status":"PASS"}).status,"BUDGET_EXHAUSTED")
            store.close()

    def test_suspected_language_leak_is_held_without_tts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AttemptStore(Path(tmp) / "a.sqlite")
            action = plan_repairs(FailureCause.LANGUAGE_LEAK_SUSPECTED)[0]
            called = []
            result = apply_repair(action, line_id="l", input_audio_sha256="a" * 64, reference_sha256=None, store=store, executor=lambda _: called.append(1))
            self.assertEqual(result.status, "HOLD_NO_TTS")
            self.assertFalse(called)
            store.close()

    def test_reaudit_reopens_and_hashes_repair_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AttemptStore(Path(tmp) / "a.sqlite")
            output = Path(tmp) / "repair.wav"
            action = plan_repairs(FailureCause.SEAM_FAIL)[0]
            def execute(_):
                output.write_bytes(b"audio")
                return {"status": "PASS", "output_audio_path": str(output)}
            outcome = apply_repair(action, line_id="l", input_audio_sha256="a" * 64, reference_sha256=None, store=store, executor=execute)
            self.assertEqual(outcome.status, "PASS")
            checked = re_audit_repair(outcome, output_audio_path=output, auditor=lambda path: {"passed": True, "path": path})
            self.assertEqual(checked.status, "REPAIR_QA_PASS")
            self.assertEqual(len(checked.output_audio_sha256), 64)
            store.close()

if __name__=="__main__": unittest.main()
