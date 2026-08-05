from __future__ import annotations
import unittest
from dubbing_pipeline.fmv_selector import select_local_scene

class Line:
    def __init__(self, id): self.id=id

class FMVTests(unittest.TestCase):
    def test_local_selector_does_not_enumerate_product(self):
        calls=[]
        lines=[Line("a"),Line("b")]; options={line.id:[{"candidate":type("C",(),{"candidate_id":line.id+str(i)})(),"eligible":True} for i in range(3)] for line in lines}; source=[]
        def mount(value,line,option): calls.append((line.id,option["candidate"].candidate_id)); return list(value)+[line.id]
        result=select_local_scene(lines,options,source,max_candidates_per_line=3,max_iterations=2,mount_line=mount,audit_scene=lambda value,index:(True,{"index":index}))
        self.assertTrue(result.passed); self.assertEqual(result.attempts,1); self.assertEqual(set(result.selected),{"a","b"}); self.assertLess(len(calls),9)
    def test_blocked_candidate_is_visible(self):
        line=Line("a"); options={"a":[{"candidate":type("C",(),{"candidate_id":"bad"})(),"eligible":True}]}; result=select_local_scene([line],options,[],max_candidates_per_line=1,max_iterations=1,mount_line=lambda *args: (_ for _ in ()).throw(ValueError("seam")),audit_scene=lambda *_:(False,None)); self.assertFalse(result.passed); self.assertTrue(result.matrix)
    def test_scene_failure_substitutes_only_attributed_line(self):
        lines=[Line("a"),Line("b")]
        options={"a":[{"candidate":type("C",(),{"candidate_id":"a1"})(),"eligible":True},{"candidate":type("C",(),{"candidate_id":"a2"})(),"eligible":True}],"b":[{"candidate":type("C",(),{"candidate_id":"b1"})(),"eligible":True}]}
        audits=[]
        def audit(_value,index):
            audits.append(index)
            return (index == 2, type("Audit",(),{"diagnostics":{"failed_line_ids":["a"]}})())
        result=select_local_scene(lines,options,[],max_candidates_per_line=2,max_iterations=3,mount_line=lambda value,line,option:value+[(line.id,option["candidate"].candidate_id)],audit_scene=audit)
        self.assertTrue(result.passed)
        self.assertEqual(result.selected["a"]["candidate"].candidate_id,"a2")
        self.assertEqual(result.selected["b"]["candidate"].candidate_id,"b1")
        self.assertTrue(any(row["action"] == "SUBSTITUTE_ATTRIBUTED_LINE" and row["line_id"] == "a" for row in result.matrix))

if __name__=="__main__": unittest.main()
