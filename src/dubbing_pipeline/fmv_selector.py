"""Deterministic local FMV candidate selection; never enumerates a Cartesian product."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

@dataclass(frozen=True)
class LocalSelection:
    passed: bool
    working: Any | None
    selected: dict[str, dict[str, Any]]
    audit: Any | None
    attempts: int
    matrix: list[dict[str, Any]] = field(default_factory=list)

def select_local_scene(lines: Sequence[Any], options_by_line: Mapping[str, Sequence[dict[str, Any]]], source: Any, *, max_candidates_per_line: int, max_iterations: int, mount_line: Callable[[Any, Any, dict[str, Any]], Any], audit_scene: Callable[[Any, int], tuple[bool, Any]], rank: Callable[[dict[str, Any]], Any] | None = None) -> LocalSelection:
    """Greedy + bounded one-line substitutions; each trial remounts from source."""
    rank = rank or (lambda option: 0)
    lists={line.id: sorted([option for option in options_by_line.get(line.id, ()) if option.get("eligible")], key=rank, reverse=True)[:max(1,max_candidates_per_line)] for line in lines}
    selected={}; cursors={line.id: 0 for line in lines}; matrix=[]; last_audit=None; attempts=0
    if any(not lists.get(line.id) for line in lines):
        return LocalSelection(False, None, selected, None, 0, [{"line_id": line.id, "candidate_count": len(lists.get(line.id, [])), "blocker": "NO_LOCAL_CANDIDATE", "action": "HOLD"} for line in lines if not lists.get(line.id)])
    for iteration in range(max(1,max_iterations)):
        for line in lines:
            selected[line.id] = lists[line.id][cursors[line.id]]
        working=source.copy() if hasattr(source,"copy") else source
        mount_failure=None
        for line in lines:
            try:
                working=mount_line(working,line,selected[line.id])
            except Exception as exc:
                mount_failure=(line.id, str(exc)); break
        attempts += 1
        if mount_failure:
            line_id, error=mount_failure
            matrix.append({"line_id":line_id,"candidate_id":getattr(selected[line_id].get("candidate"),"candidate_id",None),"eligible":True,"blocker":error,"action":"SUBSTITUTE_ATTRIBUTED_LINE"})
            if cursors[line_id] + 1 < len(lists[line_id]):
                cursors[line_id] += 1
                continue
            break
        passed,last_audit=audit_scene(working,attempts)
        if passed: return LocalSelection(True,working,dict(selected),last_audit,attempts,matrix)
        diagnostics=getattr(last_audit,"diagnostics",{}) or {}
        culprit_ids=list(diagnostics.get("failed_line_ids") or ([diagnostics["failed_line_id"]] if diagnostics.get("failed_line_id") else []))
        if not culprit_ids:
            culprit_ids=[line.id for line in lines if cursors[line.id] + 1 < len(lists[line.id])]
        culprit=next((line_id for line_id in culprit_ids if line_id in cursors and cursors[line_id] + 1 < len(lists[line_id])), None)
        if culprit is None:
            break
        old=selected[culprit]; cursors[culprit] += 1
        matrix.append({"line_id":culprit,"candidate_id":getattr(old.get("candidate"),"candidate_id",None),"blocker":diagnostics.get("failed_gates") or diagnostics.get("reason") or "SCENE_QA_FAILED","action":"SUBSTITUTE_ATTRIBUTED_LINE","next_candidate_id":getattr(lists[culprit][cursors[culprit]].get("candidate"),"candidate_id",None)})
    return LocalSelection(False,None,dict(selected),last_audit,attempts,matrix)

__all__=["LocalSelection","select_local_scene"]
