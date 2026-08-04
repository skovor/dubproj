"""Scene-level candidate matrix used by FMV local repair reports."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

def build_candidate_matrix(lines: Sequence[Any], options_by_line: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows=[]
    for line in lines:
        for option in options_by_line.get(line.id, ()):
            candidate=option.get("candidate")
            rows.append({"line_id":line.id,"candidate_id":getattr(candidate,"candidate_id",None),"stages":{"generated":True,"raw_qa":option.get("raw_audit") is not None,"processed":option.get("processed_audit") is not None,"mounted":option.get("mounted_audit") is not None,"serialized":option.get("serialized_audit") is not None},"eligible":bool(option.get("eligible")),"blocker":option.get("error") or option.get("alignment_status"),"recommended_action":None if option.get("eligible") else "REVIEW_OR_CAUSAL_REPAIR"})
    return rows

__all__=["build_candidate_matrix"]
