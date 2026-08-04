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
    selected={}; matrix=[]; last_audit=None; attempts=0
    for iteration in range(max(1,max_iterations)):
        changed=False; working=source.copy() if hasattr(source,"copy") else source
        for line in lines:
            options=lists.get(line.id,[]); choices=[]
            if line.id in selected: choices.append(selected[line.id])
            choices.extend(option for option in options if option not in choices)
            chosen=None
            for option in choices:
                trial=source.copy() if hasattr(source,"copy") else source
                try:
                    for prior in lines:
                        if prior.id == line.id: break
                        if prior.id in selected: trial=mount_line(trial,prior,selected[prior.id])
                    trial=mount_line(trial,line,option); chosen=(option,trial); break
                except Exception as exc:
                    matrix.append({"line_id":line.id,"candidate_id":getattr(option.get("candidate"),"candidate_id",None),"eligible":True,"blocker":str(exc),"action":"TRY_NEXT_LOCAL"})
            if chosen is None:
                matrix.append({"line_id":line.id,"candidate_count":len(options),"blocker":"NO_LOCAL_CANDIDATE","action":"HOLD"}); continue
            if selected.get(line.id) is not chosen[0]: changed=True
            selected[line.id]=chosen[0]; working=chosen[1]
        attempts += 1
        if len(selected)==len(lines):
            # Rebuild once in line order so selected options are all mounted.
            working=source.copy() if hasattr(source,"copy") else source
            try:
                for line in lines: working=mount_line(working,line,selected[line.id])
            except Exception as exc:
                matrix.append({"blocker":str(exc),"action":"LOCAL_REPAIR"}); changed=True; continue
            passed,last_audit=audit_scene(working,attempts)
            if passed: return LocalSelection(True,working,dict(selected),last_audit,attempts,matrix)
        if not changed: break
    return LocalSelection(False,None,dict(selected),last_audit,attempts,matrix)

__all__=["LocalSelection","select_local_scene"]
