"""Deterministic local FMV candidate selection; never enumerates a Cartesian product."""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
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
    """Bounded best-first search over visited candidate-index states.

    The search never enumerates the Cartesian product.  It expands the
    attributed failed lines first, then all remaining one-step neighbors when
    attribution is absent or exhausted.  This permits returning to an earlier
    option for a different line without repeating a state indefinitely.
    """
    rank = rank or (lambda option: 0)
    lists={line.id: sorted([option for option in options_by_line.get(line.id, ()) if option.get("eligible")], key=rank, reverse=True)[:max(1,max_candidates_per_line)] for line in lines}
    selected={}; matrix=[]; last_audit=None; attempts=0
    if any(not lists.get(line.id) for line in lines):
        return LocalSelection(False, None, selected, None, 0, [{"line_id": line.id, "candidate_count": len(lists.get(line.id, [])), "blocker": "NO_LOCAL_CANDIDATE", "action": "HOLD"} for line in lines if not lists.get(line.id)])
    line_ids=[line.id for line in lines]
    initial=tuple(0 for _ in line_ids)
    queue: list[tuple[float, int, tuple[int, ...]]] = []
    counter=0
    def priority(state: tuple[int, ...]) -> float:
        return sum(float(rank(lists[line_id][index])) for line_id, index in zip(line_ids, state))
    heapq.heappush(queue, (-priority(initial), counter, initial)); visited={initial}
    while queue and attempts < max(1, max_iterations):
        _score, _counter, state = heapq.heappop(queue)
        selected={line_id: lists[line_id][index] for line_id, index in zip(line_ids, state)}
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
            failed_ids=[line_id]
            diagnostics={"failed_line_ids": failed_ids, "reason": error}
        else:
            passed,last_audit=audit_scene(working,attempts)
            if passed: return LocalSelection(True,working,dict(selected),last_audit,attempts,matrix)
            diagnostics=(last_audit.get("diagnostics", last_audit) if isinstance(last_audit, Mapping) else getattr(last_audit,"diagnostics",{})) if last_audit is not None else {}
            diagnostics=diagnostics if isinstance(diagnostics, Mapping) else {}
            culprit_ids=list(diagnostics.get("failed_line_ids") or ([diagnostics["failed_line_id"]] if diagnostics.get("failed_line_id") else []))
            if not culprit_ids:
                culprit_ids=[str(row.get("line_id")) for row in (diagnostics.get("failed_line_results") or diagnostics.get("line_gate_results") or []) if row.get("passed") is False]
            failed_ids=[line_id for line_id in culprit_ids if line_id in line_ids]
            blocker = diagnostics.get("failed_gates") or diagnostics.get("reason") or "SCENE_QA_FAILED"
        # Failed attribution is a priority hint, not an exclusive lock.  A
        # stalled culprit must not prevent a valid alternative on another line.
        ordered_ids=[]
        for line_id in failed_ids + line_ids:
            if line_id not in ordered_ids: ordered_ids.append(line_id)
        for line_id in ordered_ids:
            position=line_ids.index(line_id); next_index=state[position] + 1
            if next_index >= len(lists[line_id]): continue
            next_state=list(state); next_state[position]=next_index; next_state=tuple(next_state)
            if next_state in visited: continue
            visited.add(next_state); counter += 1
            old_option=selected[line_id]
            matrix.append({"line_id": line_id, "candidate_id": getattr(old_option.get("candidate"), "candidate_id", None), "failed_line_ids": failed_ids, "blocker": blocker, "action": "SUBSTITUTE_ATTRIBUTED_LINE" if line_id in failed_ids else "SUBSTITUTE_FALLBACK_LINE", "next_candidate_id": getattr(lists[line_id][next_index].get("candidate"), "candidate_id", None), "state": list(state)})
            heapq.heappush(queue, (-priority(next_state), counter, next_state))
    return LocalSelection(False,None,dict(selected),last_audit,attempts,matrix)

__all__=["LocalSelection","select_local_scene"]
