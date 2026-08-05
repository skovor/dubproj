"""Deterministic cause-to-strategy table."""
from __future__ import annotations
from typing import Any
from .repair import FailureCause, RepairAction

def plan_repairs(cause: FailureCause | str, diagnostics: dict[str, Any] | None = None) -> list[RepairAction]:
    cause=FailureCause(cause); diagnostics=diagnostics or {}
    if cause is FailureCause.FINAL_ANCHOR_MISSING: return [RepairAction("regenerate_final_anchor",cause,{"append_ellipsis":True,"prompt_variant":"final-anchor"},True,2,"directed final-word retry")]
    if cause is FailureCause.ACTIVE_BODY_OVERFLOW: return [RepairAction("duration_corrective_atempo",cause,{"max_ratio_deviation":.15},False,1,"bounded duration correction")]
    if cause is FailureCause.VOWEL_TOO_SHORT: return [RepairAction("localized_vowel_extension",cause,{"max_ms":120},False,1,"localized repair")]
    if cause is FailureCause.SEAM_FAIL: return [RepairAction("surgical_crossfade",cause,{"max_ms":80},False,1,"recompute seam")]
    if cause is FailureCause.REFERENCE_LIMITED: return [RepairAction("materialize_new_reference",cause,{"source":"validated-source"},False,1,"reference evidence first")]
    if cause is FailureCause.LANGUAGE_LEAK_CONFIRMED: return [RepairAction("regenerate_target_language",cause,{"language":"de","append_ellipsis":True},True,2,"independent leak evidence")]
    if cause is FailureCause.LANGUAGE_LEAK_SUSPECTED: return [RepairAction("hold_language_review",cause,{"requires":"independent_lid_or_human"},False,0,"suspected leak is not sufficient evidence for TTS")]
    return [RepairAction("hold_for_evidence",cause,{"reason":"no TTS retry"},False,0,"configuration/evidence issue")]

__all__=["plan_repairs"]
