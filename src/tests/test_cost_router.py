from __future__ import annotations
import unittest
from dubbing_pipeline.model_pool import ModelIdentity, ModelPool
from dubbing_pipeline.scheduler import QALevel, route_qa

class CostRouterTests(unittest.TestCase):
    def test_technical_failure_stops_at_level_zero(self): self.assertEqual(route_qa(technical_passed=False),[QALevel.TECHNICAL])
    def test_uncertain_escalates_without_generation(self): self.assertEqual(route_qa(technical_passed=True,provisional_status="ASR_UNCERTAIN"),[1,2])
    def test_language_conflict_uses_lid_even_for_one_candidate(self): self.assertEqual(route_qa(technical_passed=True, provisional_status="LANGUAGE_LEAK_SUSPECTED", lid_available=True), [1, 2, 3])
    def test_pool_loads_once(self):
        pool=ModelPool(); identity=ModelIdentity("asr","m","r","cuda:0"); loads=[]; loader=lambda: loads.append(1) or object(); pool.get(identity,loader); pool.get(identity,loader); self.assertEqual(len(loads),1); self.assertEqual(pool.load_counts()[identity],1); pool.close()

if __name__=="__main__": unittest.main()
