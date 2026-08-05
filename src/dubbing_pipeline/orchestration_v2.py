"""V2 orchestration with post-transform and post-serialization QA.

The raw TTS candidate is only a probe.  A line becomes deliverable after its
processed file, mounted delivery, serialized artifact and (for FMV) complete
scene have all passed their own evidence-backed audits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .audio import read
from .alignment import AlignmentCache, AlignmentUnavailable, contrastive_align, language_id_evidence
from .lid import LIDPolicy, independent_lid, fuse_language_evidence
from .asr import ASRCache, DualASREvidence, prepare_whisperx_escalation, transcribe_dual
from .contracts import DeliveryWindow, FailureClass, RunState
from .contracts.manifest import validate_scene_value
from .generation_v2 import GenerationRuntimeV2, generate_cohort_v2
from .hashing import atomic_json, contract_hash, sha256_bytes, sha256_file
from .lab import SandboxLayout
from .mapping import validate_scene
from .models import Line, Scene
from .montage import mount_surgical
from .policy import BLOCKED, KEEP_ORIGINAL, classify_line
from .post_qa import StageAudit, audit_candidate_stage, audit_scene_stage, persist_audio_atomic
from .processing import process_candidate
from .qa_v2 import LanguageProfile, QAResultV2, evaluate_candidate_v2, is_provisional_result, linguistic_status, rank_candidate_v2, rank_provisional_v2, select_passed_v2
from .reference import materialize_reference
from .runtime_lock import assert_backend_matches_lock, assert_reproducible
from .scheduler import run_cohorts
from .scheduler import route_qa
from .fmv_selector import select_local_scene
from .scene_qa import build_candidate_matrix
from .performance import PerformanceEvidence, classify_performance
from .performance_policy import policy_for
from .model_pool import ModelIdentity, ModelPool
from .state import StateStore
from .attempts import AttemptStore
from .repair import FailureCause, apply_repair
from .repair_planner import plan_repairs


def _line_window(line: Line, sample_rate: int, scene_id: str, channel: int) -> DeliveryWindow:
    start, end = round(float(line.start) * sample_rate), round(float(line.end) * sample_rate)
    speech_start = round(float(line.speech_start if line.speech_start is not None else line.start) * sample_rate)
    speech_end = round(float(line.speech_end if line.speech_end is not None else line.end) * sample_rate)
    parsed_intervals = []
    for item in line.preserved_source_intervals:
        if isinstance(item, dict):
            start_value, end_value = item.get("start"), item.get("end")
        else:
            start_value, end_value = (item[0], item[1]) if len(item) >= 2 else (None, None)
        if start_value is not None and end_value is not None:
            parsed_intervals.append((round(float(start_value) * sample_rate), round(float(end_value) * sample_rate)))
    intervals = tuple(parsed_intervals)
    resume = round(float(line.source_resume) * sample_rate) if line.source_resume is not None else None
    return DeliveryWindow(scene_id, line.id, start, end, speech_start, speech_end, intervals, resume, channel, contract_hash("timebase", {"scene": scene_id, "line": line.id, "start": start, "end": end}))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "artifact"


def _transcribe_evidence(
    asr: Any,
    path: str | Path,
    cache: ASRCache,
    *,
    source_language: str,
    target_language: str,
    semantic_key: str | None = None,
) -> DualASREvidence | None:
    """Return dual evidence, or no linguistic evidence when ASR is disabled."""
    if asr is None:
        return None
    return transcribe_dual(
        asr,
        path,
        source_language=source_language,
        target_language=target_language,
        cache=cache,
        semantic_key=semantic_key,
    )


def _transcribe_path(asr: Any, path: str | Path, cache: ASRCache | dict[str, Any]) -> tuple[str | None, str | None, float | None]:
    """Compatibility shim exposing the forced-target reading only."""
    if asr is None:
        return None, None, None
    if isinstance(cache, ASRCache):
        evidence = _transcribe_evidence(asr, path, cache, source_language="en", target_language="de")
        if evidence is None:
            return None, None, None
        return evidence.forced_target.text, evidence.forced_target.language, evidence.forced_target.probability
    # Older callers supplied a plain dictionary; preserve its one-read API.
    key = sha256_file(path)
    if key not in cache:
        value = asr.transcribe(str(path))
        cache[key] = (value.get("text"), value.get("language"), value.get("probability")) if isinstance(value, dict) else (str(value), None, None)
    return cache[key]


def _profile(config: Any, supplied: LanguageProfile | None) -> LanguageProfile:
    if supplied is not None:
        return supplied
    return LanguageProfile(
        source_language=config.source_language,
        target_language=config.target_language,
        source_markers=tuple(getattr(config.qa, "english_markers", ()) or ()),
        strong_source_words=tuple(getattr(config.qa, "strong_source_words", ()) or ()),
    )


def _calibration_kwargs(config: Any, alignment_backend: Any, *, performance_mode: str | None = None) -> dict[str, Any]:
    """Bind QA authority to the exact active alignment runtime identity."""
    runtime_lock = getattr(config, "runtime_lock", None)
    models_lock = getattr(config, "models_lock", None)
    return {
        "calibration_authority": bool(getattr(config.qa, "calibration_authority", False)),
        "calibration_profile": getattr(config.qa, "calibration_profile", None),
        "calibration_profile_root": getattr(config.qa, "calibration_profile_root", None),
        "feature_schema_version": "char-alignment-v2",
        "backend_id": str(getattr(alignment_backend, "backend_id", "unknown")) if alignment_backend is not None else None,
        "runtime_lock_sha256": sha256_file(runtime_lock) if runtime_lock is not None and Path(runtime_lock).is_file() else None,
        "models_lock_sha256": sha256_file(models_lock) if models_lock is not None and Path(models_lock).is_file() else None,
        "model_id": str(getattr(alignment_backend, "model_id", "unknown")) if alignment_backend is not None else None,
        "model_revision": str(getattr(alignment_backend, "model_revision", "unknown")) if alignment_backend is not None else None,
        "performance_mode": str(performance_mode or getattr(config.qa, "performance_mode", "NEUTRAL")),
    }


def _line_performance(line: Line, config: Any) -> PerformanceEvidence:
    """Resolve performance independently for every line.

    Explicit metadata is authoritative for routing; duration is only a
    diagnostic fallback and never changes lexical content.  Keeping this
    decision line-local prevents a scene-level mode from leaking into a
    different speaker or delivery.
    """
    metadata = dict(getattr(line, "metadata", {}) or {})
    if not metadata.get("performance_mode") and getattr(config.qa, "performance_mode", None):
        metadata["performance_mode"] = getattr(config.qa, "performance_mode")
    duration = max(0.0, float(line.end) - float(line.start)) if line.end > line.start else None
    return classify_performance(metadata=metadata, duration_seconds=duration)


def _line_hard_gates(config: Any, performance: PerformanceEvidence) -> list[str]:
    """Apply only the mode-specific hard gates while retaining diagnostics."""
    gates = list(getattr(config.qa, "hard_gates", ()) or ())
    policy = policy_for(performance.mode)
    if not policy.require_content:
        gates = [gate for gate in gates if gate != "content"]
    if not policy.require_final_word:
        gates = [gate for gate in gates if gate != "final_word"]
    if not policy.require_loudness:
        gates = [gate for gate in gates if gate != "active_loudness"]
    return gates


def _result_or_failed(audit: StageAudit) -> QAResultV2:
    if audit.result is not None:
        return audit.result
    return QAResultV2(False, audit.gates, audit.diagnostics, audit.failure_class)


def _audit_can_escalate(audit: StageAudit | None) -> bool:
    """Technical PASS plus a provisional linguistic state may be aligned."""
    if audit is None or audit.result is None:
        return False
    result = audit.result
    technical = ("not_empty", "finite_audio", "sample_rate", "channels", "frames", "clipping", "active_loudness", "tail", "serialization_contract")
    if any(result.gates.get(name) is not None and result.gates[name].status.name == "FAIL" for name in technical):
        return False
    return is_provisional_result(result)


def _line_linguistic_summary(row: dict[str, Any], options: list[dict[str, Any]], *, expected_text: str, source_text: str = "", target_language: str = "de") -> None:
    decisions: list[dict[str, Any]] = []
    escalations: list[dict[str, Any]] = []
    for option in options:
        candidate = option.get("candidate")
        raw = option.get("raw_audit")
        mounted = option.get("mounted_audit")
        raw_decision = (raw.diagnostics if raw is not None else {}).get("linguistic_decision") or {}
        mounted_decision = (mounted.diagnostics if mounted is not None else {}).get("linguistic_decision") or {}
        decision = mounted_decision or raw_decision
        alignment_status = option.get("alignment_status")
        # A screened Whisper result is never presented as a final verdict
        # when the independent family is unavailable or still ambiguous.
        effective_status = decision.get("status")
        if not option.get("eligible") and alignment_status in {
            "ALIGNER_NOT_APPLICABLE", "ALIGNMENT_UNCERTAIN", "ALIGNMENT_ERROR", "ASR_EVIDENCE_MISSING",
        }:
            effective_status = "ASR_UNCERTAIN" if alignment_status == "ASR_EVIDENCE_MISSING" else ("ALIGNMENT_UNCERTAIN" if alignment_status == "ALIGNMENT_ERROR" else alignment_status)
        item = {
            "candidate_id": getattr(candidate, "candidate_id", None),
            "raw_status": raw_decision.get("status"),
            "mounted_status": mounted_decision.get("status"),
            "status": effective_status,
            "eligible": bool(option.get("eligible", False)),
            "alignment_status": alignment_status,
            "evidence_families": decision.get("evidence_families", []),
            "evidence_hashes": decision.get("evidence_hashes", []),
            "detected_language": decision.get("detected_language"),
            "language_probability": decision.get("language_probability"),
            "expected_alignment_score": decision.get("expected_alignment_score"),
            "source_alignment_score": decision.get("source_alignment_score"),
            "alignment_margin": decision.get("alignment_margin"),
            "calibration_authority": decision.get("calibration_authority", False),
            "calibration_profile_status": decision.get("calibration_profile_status", "DISABLED"),
            "native_char_coverage": decision.get("native_char_coverage"),
            "mean_char_score": decision.get("mean_char_score"),
            "minimum_char_score": decision.get("minimum_char_score"),
            "p10_char_score": decision.get("p10_char_score"),
            "unaligned_characters": decision.get("unaligned_characters", []),
            "interpolated_characters": decision.get("interpolated_characters", []),
            "compression_ratio": decision.get("compression_ratio"),
            "final_anchor_evidence": decision.get("final_anchor_evidence"),
            "missing_tokens": decision.get("missing_tokens", []),
            "final_anchor_present": decision.get("final_anchor_present"),
            "reason": decision.get("reason", ""),
        }
        decisions.append(item)
        if raw_decision.get("status") in {
            "ASR_UNCERTAIN", "LANGUAGE_LEAK_SUSPECTED", "LANGUAGE_LEAK_STRONG_SUSPICION",
            "LEXICAL_FAILURE_SUSPECTED", "ALIGNMENT_UNCERTAIN", "ALIGNER_NOT_APPLICABLE",
            "TARGET_ALIGNMENT_SUPPORT", "TARGET_ALIGNMENT_WEAK", "EVIDENCE_CONFLICT",
        }:
            evidence = raw.diagnostics.get("asr", {}) if raw is not None else {}
            escalations.append(prepare_whisperx_escalation(
                raw.artifact_path if raw is not None and raw.artifact_path else "",
                candidate_id=getattr(candidate, "candidate_id", None),
                expected_text=expected_text,
                source_text=source_text,
                language=target_language,
                reason=str(raw_decision.get("status")),
                evidence_hashes=list(raw_decision.get("evidence_hashes") or evidence.get("evidence_hashes") or []),
            ).to_dict())
    row["candidate_linguistic_decisions"] = decisions
    row["line_linguistic_summary"] = {
        "candidate_count": len(decisions),
        "status_counts": {status: sum(1 for item in decisions if item.get("status") == status) for status in sorted({item.get("status") for item in decisions if item.get("status")})},
        "eligible_count": sum(1 for item in decisions if item.get("eligible")),
    }
    if escalations:
        row["whisperx_escalations"] = escalations


def _line_delivery_clip(mounted, line: Line, scene: Scene, sample_rate: int):
    import numpy as np

    value = np.asarray(mounted, dtype="float32")
    start = round(float(line.speech_start if line.speech_start is not None else line.start) * sample_rate)
    end = round(float(line.speech_end if line.speech_end is not None else line.end) * sample_rate)
    if start < 0 or end <= start or end > len(value):
        raise ValueError(f"line delivery clip outside mounted scene: {line.id} {start}:{end}/{len(value)}")
    if scene.dialogue_channel < 0 or scene.dialogue_channel >= value.shape[1]:
        raise ValueError(f"dialogue channel outside mounted scene: {scene.dialogue_channel}")
    return value[start:end, scene.dialogue_channel], end - start


def _scene_integrity(source, mounted, lines: list[Line], scene: Scene, sample_rate: int) -> tuple[bool, bool, dict[str, Any]]:
    """Check protected samples and non-dialogue channels against the source."""
    import numpy as np

    before = np.asarray(source, dtype="float32")
    after = np.asarray(mounted, dtype="float32")
    details: dict[str, Any] = {"source_shape": list(before.shape), "mounted_shape": list(after.shape), "protected_intervals": []}
    untouched = before.shape == after.shape
    if untouched:
        for channel in range(before.shape[1]):
            if channel != scene.dialogue_channel and not np.array_equal(before[:, channel], after[:, channel]):
                untouched = False
                break
    protected = True
    if before.shape != after.shape or scene.dialogue_channel >= before.shape[1]:
        protected = False
    else:
        channel = scene.dialogue_channel
        for line in lines:
            for item in line.preserved_source_intervals:
                if isinstance(item, dict):
                    start_value, end_value = item.get("start"), item.get("end")
                else:
                    start_value, end_value = (item[0], item[1]) if len(item) >= 2 else (None, None)
                if start_value is None or end_value is None:
                    continue
                start, end = round(float(start_value) * sample_rate), round(float(end_value) * sample_rate)
                ok = 0 <= start < end <= len(before) and np.array_equal(before[start:end, channel], after[start:end, channel])
                details["protected_intervals"].append({"line_id": line.id, "start": start, "end": end, "passed": bool(ok)})
                protected = protected and ok
    details["protected_ok"] = bool(protected)
    details["untouched_channels_ok"] = bool(untouched)
    return protected, untouched, details


def _stage_bundle(option: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": option["raw_audit"].to_dict() if option.get("raw_audit") is not None else None,
        "processed": option["processed_audit"].to_dict() if option.get("processed_audit") is not None else None,
        "mounted": option["mounted_audit"].to_dict() if option.get("mounted_audit") is not None else None,
        "serialized": option["serialized_audit"].to_dict() if option.get("serialized_audit") is not None else None,
        "alignment": option.get("alignment"),
        "lid": option.get("lid"),
        "lid_fusion": option.get("lid_fusion"),
        "mfa": option.get("mfa"),
        "mfa_status": option.get("mfa_status"),
        "qa_route": option.get("qa_route"),
        "alignment_status": option.get("alignment_status"),
        "error": option.get("error"),
    }


def _aggregate_line_scene_audit(rows: list[dict[str, Any]]) -> StageAudit:
    passed = all(row.get("status") in {"FINAL_PASS", KEEP_ORIGINAL} for row in rows)
    qa_hash = contract_hash("scene-qa-v2", {"topology": "LINE_SEPARATED", "statuses": [row.get("status") for row in rows]})
    return StageAudit(stage="SCENE_QA", passed=passed, qa_hash=qa_hash, diagnostics={"topology": "LINE_SEPARATED", "line_count": len(rows)})


def run_scene_v2(scene: Scene, config: Any, *, runtime: GenerationRuntimeV2, asr: Any = None, stem_path: str | Path | None = None, output_dir: str | Path | None = None, language_profile: LanguageProfile | None = None, alignment_backend: Any = None, lid_backend: Any = None, mfa_backend: Any = None, model_pool: ModelPool | None = None, state_store: StateStore | None = None, repair_executor: Any = None) -> dict[str, Any]:
    """Run one scene and select only candidates that survive delivery QA.

    Raw QA is deliberately a filter, not the final decision.  Every surviving
    candidate is processed, mounted into a disposable scene, reopened and
    audited.  FMV winners are selected only from combinations that pass the
    complete scene audit.
    """
    if not bool(getattr(config, "lab_mode", True)):
        # Direct API callers receive the same fail-closed guarantee as the
        # CLI. Lab mode explicitly reports LAB_UNPINNED instead.
        assert_reproducible(config, strict=True)
        if config.models_lock is None:
            raise ValueError("production V2 run requires models_lock")
        assert_backend_matches_lock(runtime.backend, config.models_lock, role="generation", expected_model_id=config.model_id, expected_backend_version=runtime.backend_version)
        if asr is not None:
            assert_backend_matches_lock(asr, config.models_lock, role="asr")
        if alignment_backend is not None:
            assert_backend_matches_lock(alignment_backend, config.models_lock, role="alignment")
        if lid_backend is not None:
            assert_backend_matches_lock(lid_backend, config.models_lock, role="lid")
        if mfa_backend is not None:
            assert_backend_matches_lock(mfa_backend, config.models_lock, role="mfa")
    validate_scene(scene)
    if config.lab_mode:
        if config.sandbox_root is None:
            raise ValueError("V2 lab mode requires sandbox_root")
        SandboxLayout.create(config.sandbox_root).ensure_safe()
    out = Path(output_dir or config.output_root) / scene.id
    out.mkdir(parents=True, exist_ok=True)
    owned_pool = model_pool is None
    pool = model_pool or ModelPool()
    # The generation runtime owns the already-loaded model.  Registering it
    # in the run pool still makes the identity and load count observable and
    # prevents orchestration from silently constructing a second instance.
    pool.get(ModelIdentity("generation", str(getattr(runtime, "model_id", getattr(config, "model_id", "unknown"))), str(getattr(runtime, "model_revision", getattr(config, "model_revision", "unknown"))), str(getattr(config, "device", "cpu")), str(getattr(config, "dtype", "float32"))), lambda: runtime.backend)
    for role, backend in (("asr", asr), ("alignment", alignment_backend), ("lid", lid_backend), ("mfa", mfa_backend)):
        if backend is not None:
            pool.get(ModelIdentity(role, str(getattr(backend, "model_id", "unknown")), str(getattr(backend, "model_revision", "unknown")), str(getattr(config, "device", "cpu")), str(getattr(config, "dtype", "float32"))), lambda backend=backend: backend)
    run_state = state_store or StateStore(out / "state", f"{scene.id}-{contract_hash('run', {'scene': scene.id, 'config': config.to_dict()})[:16]}")
    run_state.commit(RunState(scene.id, "PREFLIGHT"), {"event": "scene_started", "scene_id": scene.id})
    repair_store = AttemptStore(out / "repair_attempts.sqlite")
    profile = _profile(config, language_profile)
    refs: dict[str, Any] = {}
    lines = [line for line in scene.lines if classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment).policy not in {KEEP_ORIGINAL, BLOCKED}]
    performance_by_line = {line.id: _line_performance(line, config) for line in scene.lines}
    for line in lines:
        refs[line.id] = materialize_reference(line, config.project_root, config.cache_root, language=config.source_language)

    asr_cache = ASRCache(
        Path(config.cache_root) / "asr-v4",
        backend_id=str(getattr(asr, "backend_id", "unknown")),
        model_id=str(getattr(asr, "model_id", "unknown")),
        model_revision=str(getattr(asr, "model_revision", "unknown")),
    ) if asr is not None else ASRCache(backend_id="disabled")
    alignment_cache = AlignmentCache(
        Path(config.cache_root) / "alignment-v1",
        backend_id=str(getattr(alignment_backend, "backend_id", "unknown")),
        model_id=str(getattr(alignment_backend, "model_id", "unknown")),
        model_revision=str(getattr(alignment_backend, "model_revision", "unknown")),
    ) if alignment_backend is not None else None
    raw_audits: dict[str, StageAudit] = {}
    line_by_id = {line.id: line for line in lines}

    def generate(items: list[Line], round_index: int):
        takes = (config.fmv_initial_takes if scene.topology == "EMBEDDED_FMV" else config.initial_takes) if round_index == 1 else (config.fmv_retry_takes if scene.topology == "EMBEDDED_FMV" else config.retry_takes)
        take_count = int(takes)
        if take_count <= 0:
            return {line.id: [] for line in items}
        return generate_cohort_v2(runtime, [(line, refs[line.id], take_count) for line in items], config, round_index=round_index, cache_root=config.cache_root)

    def evaluate(candidate):
        line = line_by_id[candidate.line_id]
        performance = performance_by_line[line.id]
        evidence = _transcribe_evidence(
            asr,
            candidate.raw_audio,
            asr_cache,
            source_language=config.source_language,
            target_language=config.target_language,
            semantic_key=f"{candidate.generation_hash}:{line.id}:{line.effective_target_text}",
        )
        transcript = evidence.forced_target.text if evidence is not None else None
        language = evidence.automatic.language if evidence is not None else None
        probability = evidence.automatic.probability if evidence is not None else None
        audit = audit_candidate_stage(
            candidate.raw_audio,
            stage="RAW_TECHNICAL_QA",
            expected_text=line.effective_target_text,
            source_text=line.source_text,
            target_sample_rate=int(getattr(config, "native_sample_rate", 24000)),
            target_frames=None,
            channels=1,
            # The raw producer is not yet aligned to a delivery window.  Tail
            # and frame constraints belong to PROCESSED_QA onward.
            reference_end=None,
            transcript=transcript,
            language=language,
            language_probability=probability,
            profile=profile,
            hard_gates=_line_hard_gates(config, performance),
            final_word_min_tokens=config.qa.final_word_min_tokens,
            tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
            require_asr=True,
            linguistic_evidence=evidence.to_dict() if evidence is not None else None,
        )
        raw_audits[candidate.candidate_id] = audit
        return _result_or_failed(audit)

    cohort = run_cohorts(lines, item_id=lambda item: item.id, generate=generate, evaluate=evaluate, max_retry_rounds=1)
    report: dict[str, Any] = {
        "schema": "scene-report-v2",
        "scene": scene.id,
        "topology": scene.topology,
        "run_id": cohort.run_id,
        "lines": [],
        "blockers": [],
        "raw_retry_ids": list(cohort.retry_ids),
        "phases": list(cohort.phases),
        "stage_evidence": {},
        "performance_by_line": {line_id: evidence.to_dict() for line_id, evidence in performance_by_line.items()},
        "qa_routes": {},
        "model_pool": {},
        "repair_attempts": [],
        "mfa": {"available": mfa_backend is not None, "requested": 0, "executed": 0},
    }
    stage_counts = {name: 0 for name in ("RAW_TECHNICAL_QA", "PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA", "LINGUISTIC_ALIGNMENT", "SCENE_QA")}
    alignment_count = 0

    def _record_repair(line: Line, option: dict[str, Any]) -> None:
        """Record one bounded causal action for a rejected candidate."""
        audit = option.get("mounted_audit") or option.get("processed_audit") or option.get("raw_audit")
        decision = ((audit.diagnostics if audit is not None else {}).get("linguistic_decision") or {})
        status = str(decision.get("status") or option.get("alignment_status") or "")
        if status in {"ASR_UNCERTAIN", "ALIGNMENT_UNCERTAIN", "ASR_EVIDENCE_MISSING"}:
            cause = FailureCause.ASR_UNCERTAIN
        elif status in {"LANGUAGE_LEAK_SUSPECTED", "LANGUAGE_LEAK_STRONG_SUSPICION", "EVIDENCE_CONFLICT"}:
            cause = FailureCause.LANGUAGE_LEAK_CONFIRMED
        elif not bool(decision.get("final_anchor_present", True)):
            cause = FailureCause.FINAL_ANCHOR_MISSING
        else:
            cause = FailureCause.DETERMINISTIC_CALIBRATION if status == "BLOCKED" else FailureCause.SEAM_FAIL
        action = plan_repairs(cause, decision)
        if not action:
            return
        source_path = option.get("mounted_path") or option.get("processed_path") or getattr(option.get("candidate"), "raw_audio", None)
        if not source_path or not Path(source_path).is_file():
            return
        reference_hash = refs.get(line.id).audio_sha256 if refs.get(line.id) is not None else None
        outcome = apply_repair(action[0], line_id=line.id, input_audio_sha256=sha256_file(source_path), reference_sha256=reference_hash, store=repair_store, executor=repair_executor)
        report["repair_attempts"].append({"line_id": line.id, "candidate_id": getattr(option.get("candidate"), "candidate_id", None), "cause": cause.value, "action": action[0].strategy, "outcome": outcome.status, "attempt_id": outcome.attempt_id, "diagnostics": outcome.diagnostics})
    stage_counts["RAW_TECHNICAL_QA"] = sum(len(value) for value in cohort.evaluations.values())
    report["stage_evidence"]["RAW_TECHNICAL_QA"] = {"status": "EXECUTED", "artifact_count": stage_counts["RAW_TECHNICAL_QA"]}

    source_array = None
    stem_rate = None
    scene_line_windows: list[dict[str, Any]] | None = None
    if scene.topology == "EMBEDDED_FMV":
        source_path = Path(stem_path or scene.source_stem or "")
        if not source_path.is_file():
            raise ValueError("FMV scene requires a source stem")
        source_array, stem_rate = read(source_path, always_2d=True)
        source_array = source_array.copy()
        stem_rate = int(stem_rate)
        scene_line_windows = [{"line_id": line.id, "start": round(float(line.start) * stem_rate), "end": round(float(line.end) * stem_rate)} for line in scene.lines]

    options_by_line: dict[str, list[dict[str, Any]]] = {}
    row_by_id: dict[str, dict[str, Any]] = {}
    for line in scene.lines:
        decision = classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment)
        performance = performance_by_line[line.id]
        row: dict[str, Any] = {"id": line.id, "policy": decision.policy, "policy_reason": decision.reason, "performance": performance.to_dict(), "stages": {}}
        row_by_id[line.id] = row
        if decision.policy in {KEEP_ORIGINAL, BLOCKED}:
            row["status"] = decision.policy
            report["lines"].append(row)
            if decision.policy == BLOCKED:
                report["blockers"].append({"line_id": line.id, "reason": decision.reason or "BLOCKED"})
            continue
        evaluations = list(zip(cohort.candidates.get(line.id, []), cohort.evaluations.get(line.id, [])))
        raw_rows = []
        stage_rows: list[dict[str, Any]] = []
        for candidate, _raw_result in evaluations:
            audit = raw_audits.get(candidate.candidate_id)
            if audit is not None:
                raw_rows.append(audit.to_dict())
        row["stages"]["RAW_TECHNICAL_QA"] = raw_rows
        options: list[dict[str, Any]] = []
        for candidate, raw_result in evaluations:
            raw_audit = raw_audits.get(candidate.candidate_id)
            if raw_audit is None or not _audit_can_escalate(raw_audit):
                continue
            target_rate = int(stem_rate or config.sample_rate)
            candidate_root = out / "candidates" / _safe_name(line.id) / _safe_name(candidate.candidate_id)
            try:
                processed = process_candidate(
                    candidate.raw_audio,
                    target_sample_rate=target_rate,
                    reference_end=(line.end - line.start) if scene.topology == "EMBEDDED_FMV" else None,
                    ffmpeg=config.ffmpeg,
                    tmpdir=out / "processing",
                )
                processed_path = persist_audio_atomic(candidate_root / "processed.wav", processed.audio, processed.sample_rate)
                stage_counts["PROCESSED_QA"] += 1
                processing_steps = set(processed.diagnostics.get("steps", []))
                semantic_key = f"{candidate.generation_hash}:{line.id}:{line.effective_target_text}"
                semantic_alias = semantic_key if processing_steps.issubset({"explicit_resample"}) else None
                processed_evidence = _transcribe_evidence(
                    asr,
                    processed_path,
                    asr_cache,
                    source_language=config.source_language,
                    target_language=config.target_language,
                    semantic_key=semantic_alias,
                )
                processed_transcript = processed_evidence.forced_target.text if processed_evidence is not None else None
                processed_language = processed_evidence.automatic.language if processed_evidence is not None else None
                processed_probability = processed_evidence.automatic.probability if processed_evidence is not None else None
                processed_audit = audit_candidate_stage(
                    processed_path,
                    stage="PROCESSED_QA",
                    expected_text=line.effective_target_text,
                    source_text=line.source_text,
                    target_sample_rate=target_rate,
                    target_frames=None,
                    channels=1,
                    reference_end=(line.end - line.start) if scene.topology == "EMBEDDED_FMV" else None,
                    transcript=processed_transcript,
                    language=processed_language,
                    language_probability=processed_probability,
                    profile=profile,
                    hard_gates=_line_hard_gates(config, performance),
                    final_word_min_tokens=config.qa.final_word_min_tokens,
                    tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                    require_asr=True,
                    linguistic_evidence=processed_evidence.to_dict() if processed_evidence is not None else None,
                )
                if not _audit_can_escalate(processed_audit):
                    stage_rows.append({"raw_audit": raw_audit, "processed_audit": processed_audit, "mounted_audit": None, "serialized_audit": None})
                    continue
                if scene.topology == "EMBEDDED_FMV":
                    window = _line_window(line, stem_rate, scene.id, scene.dialogue_channel)
                    mounted_array, mount_metrics = mount_surgical(source_array, read(processed_path, always_2d=True)[0], target_rate, window, stem_rate, empalme_b=bool(line.preserved_source_intervals or line.source_resume is not None))
                    mounted_path = persist_audio_atomic(candidate_root / "mounted.wav", mounted_array, stem_rate)
                    delivery_clip, clip_frames = _line_delivery_clip(mounted_array, line, scene, stem_rate)
                    clip_path = persist_audio_atomic(candidate_root / "mounted_line.wav", delivery_clip, stem_rate)
                    mounted_evidence = _transcribe_evidence(
                        asr,
                        clip_path,
                        asr_cache,
                        source_language=config.source_language,
                        target_language=config.target_language,
                    )
                    mounted_transcript = mounted_evidence.forced_target.text if mounted_evidence is not None else None
                    mounted_language = mounted_evidence.automatic.language if mounted_evidence is not None else None
                    mounted_probability = mounted_evidence.automatic.probability if mounted_evidence is not None else None
                    stage_counts["MOUNTED_QA"] += 1
                    mounted_audit = audit_candidate_stage(
                        clip_path,
                        stage="MOUNTED_QA",
                        expected_text=line.effective_target_text,
                        source_text=line.source_text,
                        target_sample_rate=stem_rate,
                        target_frames=clip_frames,
                        channels=1,
                        reference_end=clip_frames / stem_rate,
                        transcript=mounted_transcript,
                        language=mounted_language,
                        language_probability=mounted_probability,
                        profile=profile,
                        hard_gates=_line_hard_gates(config, performance),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        preserved_ok=mount_metrics.preserved_hash_before == mount_metrics.preserved_hash_after,
                        require_asr=True,
                        linguistic_evidence=mounted_evidence.to_dict() if mounted_evidence is not None else None,
                    )
                    protected_ok, untouched_ok, _integrity = _scene_integrity(source_array, mounted_array, [line], scene, stem_rate)
                    stage_counts["SERIALIZED_QA"] += 1
                    serialized_audit = audit_scene_stage(
                        mounted_path,
                        stage="SERIALIZED_QA",
                        expected_sample_rate=stem_rate,
                        expected_frames=len(source_array),
                        expected_channels=source_array.shape[1],
                        protected_intervals_ok=protected_ok,
                        untouched_channels_ok=untouched_ok,
                    )
                else:
                    mounted_path = persist_audio_atomic(candidate_root / "mounted.wav", read(processed_path, always_2d=True)[0], target_rate)
                    delivery_clip, clip_frames = read(mounted_path, always_2d=True)
                    mounted_evidence = _transcribe_evidence(
                        asr,
                        mounted_path,
                        asr_cache,
                        source_language=config.source_language,
                        target_language=config.target_language,
                        semantic_key=semantic_alias,
                    )
                    mounted_transcript = mounted_evidence.forced_target.text if mounted_evidence is not None else None
                    mounted_language = mounted_evidence.automatic.language if mounted_evidence is not None else None
                    mounted_probability = mounted_evidence.automatic.probability if mounted_evidence is not None else None
                    stage_counts["MOUNTED_QA"] += 1
                    mounted_audit = audit_candidate_stage(
                        mounted_path,
                        stage="MOUNTED_QA",
                        expected_text=line.effective_target_text,
                        source_text=line.source_text,
                        target_sample_rate=target_rate,
                        target_frames=len(delivery_clip),
                        channels=1,
                        reference_end=len(delivery_clip) / target_rate,
                        transcript=mounted_transcript,
                        language=mounted_language,
                        language_probability=mounted_probability,
                        profile=profile,
                        hard_gates=_line_hard_gates(config, performance),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        require_asr=True,
                        linguistic_evidence=mounted_evidence.to_dict() if mounted_evidence is not None else None,
                    )
                    stage_counts["SERIALIZED_QA"] += 1
                    # SERIALIZED_QA reopens the exact mounted artifact.  The
                    # SHA cache therefore reuses both ASR readings; no second
                    # linguistic decode is warranted for serialization alone.
                    serialized_evidence = _transcribe_evidence(
                        asr,
                        mounted_path,
                        asr_cache,
                        source_language=config.source_language,
                        target_language=config.target_language,
                    )
                    serialized_transcript = serialized_evidence.forced_target.text if serialized_evidence is not None else None
                    serialized_language = serialized_evidence.automatic.language if serialized_evidence is not None else None
                    serialized_probability = serialized_evidence.automatic.probability if serialized_evidence is not None else None
                    serialized_audit = audit_candidate_stage(
                        mounted_path,
                        stage="SERIALIZED_QA",
                        expected_text=line.effective_target_text,
                        source_text=line.source_text,
                        target_sample_rate=target_rate,
                        target_frames=len(delivery_clip),
                        channels=1,
                        reference_end=len(delivery_clip) / target_rate,
                        transcript=serialized_transcript,
                        language=serialized_language,
                        language_probability=serialized_probability,
                        profile=profile,
                        hard_gates=_line_hard_gates(config, performance),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        require_asr=True,
                        linguistic_evidence=serialized_evidence.to_dict() if serialized_evidence is not None else None,
                    )
                    mount_metrics = None
                options.append({
                    "candidate": candidate,
                    "raw_audit": raw_audit,
                    "processed_audit": processed_audit,
                    "mounted_audit": mounted_audit,
                    "serialized_audit": serialized_audit,
                    "processed_path": str(processed_path),
                    "mounted_path": str(mounted_path),
                    "mount_metrics": mount_metrics,
                    "processing_hash": processed.processing_hash,
                    "eligible": False,
                })
                stage_rows.append(options[-1])
            except Exception as exc:
                # Processing/mounting failures are evidence for this candidate,
                # not permission to select the earlier raw PASS.
                row.setdefault("transform_errors", []).append({"candidate_id": candidate.candidate_id, "error": str(exc)})
                stage_rows.append({"raw_audit": raw_audit, "processed_audit": None, "mounted_audit": None, "serialized_audit": None, "error": str(exc)})
        options_by_line[line.id] = options
        row["candidate_stages"] = [_stage_bundle(option) for option in stage_rows]
        _line_linguistic_summary(row, options, expected_text=line.effective_target_text, source_text=line.source_text, target_language=config.target_language)
        report["lines"].append(row)

    for stage in ("PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA"):
        report["stage_evidence"][stage] = {"status": "EXECUTED" if stage_counts[stage] else "NOT_RUN", "artifact_count": stage_counts[stage]}

    # Selective second-family QA: at most one provisional winner per line is
    # aligned initially; a fallback candidate is aligned only if that winner
    # is rejected or remains ambiguous.  No Cartesian product is involved.
    for line in scene.lines:
        options = options_by_line.get(line.id, [])
        row = row_by_id.get(line.id)
        if row is None or not options:
            continue
        performance = performance_by_line[line.id]
        provisional = [option for option in options if _audit_can_escalate(option.get("mounted_audit"))]
        provisional.sort(key=lambda option: rank_provisional_v2(option["mounted_audit"].result), reverse=True)
        for option in provisional:
            option["alignment_status"] = "NOT_SELECTED"
        for option in provisional:
            option["alignment_status"] = "ALIGNMENT_PENDING"
            mounted_audit = option["mounted_audit"]
            provisional_status = linguistic_status(mounted_audit.result) or "ASR_UNCERTAIN"
            qa_route = route_qa(
                technical_passed=True,
                provisional_status=provisional_status,
                candidate_count=len(provisional),
                lid_available=lid_backend is not None,
                mfa_requested=bool(mfa_backend is not None and (getattr(line, "metadata", {}) or {}).get("mfa_requested", False)),
            )
            option["qa_route"] = qa_route
            if 4 in qa_route:
                report["mfa"]["requested"] += 1
            report["qa_routes"].setdefault(line.id, []).append({"candidate_id": option["candidate"].candidate_id, "levels": list(qa_route), "performance_mode": performance.mode.value})
            if 2 not in qa_route:
                option["alignment_status"] = "TECHNICAL_ONLY_HOLD"
                continue
            asr_evidence = mounted_audit.diagnostics.get("asr") if mounted_audit is not None else None
            if alignment_backend is None:
                option["alignment_status"] = "ALIGNER_NOT_APPLICABLE"
                break
            if not asr_evidence:
                option["alignment_status"] = "ASR_EVIDENCE_MISSING"
                continue
            try:
                alignment = contrastive_align(
                    alignment_backend,
                    mounted_audit.artifact_path or "",
                    target_text=line.effective_target_text,
                    source_text=line.source_text,
                    target_language=config.target_language,
                    source_language=config.source_language,
                    cache=alignment_cache,
                    semantic_key=f"{option['candidate'].generation_hash}:{line.id}:{line.effective_target_text}",
                )
                alignment_count += 1
                stage_counts["LINGUISTIC_ALIGNMENT"] += 1
                alignment_dict = alignment.to_dict()
                lid = None
                lid_fusion = None
                source_suspected = alignment.source_score is not None and alignment.source_score >= float(getattr(config.qa, "alignment_source_leak_score", .75))
                asr_snapshot = mounted_audit.diagnostics.get("asr") if mounted_audit is not None else None
                automatic_snapshot = (asr_snapshot or {}).get("automatic") if isinstance(asr_snapshot, dict) else None
                source_suspected = source_suspected or (isinstance(automatic_snapshot, dict) and str(automatic_snapshot.get("language", "")).casefold().startswith(str(config.source_language).casefold()))
                if source_suspected and lid_backend is not None:
                    import numpy as np
                    lid_audio, lid_rate = read(mounted_audit.artifact_path or "", always_2d=True)
                    speech_ratio = float(np.mean(np.abs(lid_audio[:, 0]) > 0.01)) if len(lid_audio) else 0.0
                    lid_obj = independent_lid(
                        lid_backend,
                        mounted_audit.artifact_path or "",
                        policy=LIDPolicy(source_language=config.source_language, target_language=config.target_language),
                        duration_seconds=len(lid_audio) / max(1, int(lid_rate)),
                        speech_ratio=speech_ratio,
                        sample_rate=int(lid_rate),
                        audio_sha256=sha256_file(mounted_audit.artifact_path or ""),
                    )
                    lid = lid_obj.to_dict()
                    lid_fusion = fuse_language_evidence(
                        whisper_language=(automatic_snapshot or {}).get("language") if isinstance(automatic_snapshot, dict) else None,
                        whisper_probability=(automatic_snapshot or {}).get("probability") if isinstance(automatic_snapshot, dict) else None,
                        lid=lid_obj,
                        ctc_target_raw_score=alignment.target_score,
                        # Alignment adapters currently expose the calibrated
                        # target score under the historical field; keep this
                        # explicit until a separate CTC calibrator is wired.
                        ctc_target_calibrated_probability=alignment.target_score,
                        policy=LIDPolicy(source_language=config.source_language, target_language=config.target_language),
                    )
                    lid["fusion"] = lid_fusion
                mounted_audio, mounted_rate = read(mounted_audit.artifact_path or "", always_2d=True)
                regraded = audit_candidate_stage(
                    mounted_audit.artifact_path or "",
                    stage="MOUNTED_QA",
                    expected_text=line.effective_target_text,
                    source_text=line.source_text,
                    target_sample_rate=int(mounted_rate),
                    target_frames=len(mounted_audio),
                    channels=int(mounted_audio.shape[1]),
                    reference_end=len(mounted_audio) / mounted_rate,
                    profile=profile,
                    hard_gates=_line_hard_gates(config, performance),
                    final_word_min_tokens=config.qa.final_word_min_tokens,
                    tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                    preserved_ok=(mounted_audit.gates.get("preserved_intervals").measured_value if mounted_audit.gates.get("preserved_intervals") is not None else None),
                    require_asr=True,
                    linguistic_evidence=asr_evidence,
                    alignment_evidence=alignment_dict,
                    lid_evidence=lid,
                    alignment_min_target_score=float(getattr(config.qa, "alignment_min_target_score", .65)),
                    alignment_min_margin=float(getattr(config.qa, "alignment_min_margin", .20)),
                    alignment_source_leak_score=float(getattr(config.qa, "alignment_source_leak_score", .75)),
                    **_calibration_kwargs(config, alignment_backend, performance_mode=performance.mode.value),
                )
                option["mounted_audit"] = regraded
                option["alignment"] = alignment_dict
                option["lid"] = lid
                option["lid_fusion"] = lid_fusion
                option["alignment_status"] = linguistic_status(regraded.result) or "ALIGNMENT_UNCERTAIN"
                if scene.topology != "EMBEDDED_FMV" and option.get("serialized_audit") is not None:
                    # For a line-separated asset the serialized artifact is
                    # the same speech file, but it has its own stage audit.
                    # Regrade that boundary with the same independent family;
                    # otherwise a stale pre-alignment ASR_UNCERTAIN result
                    # would veto a correctly confirmed candidate.
                    option["serialized_audit"] = audit_candidate_stage(
                        option["serialized_audit"].artifact_path or mounted_audit.artifact_path or "",
                        stage="SERIALIZED_QA",
                        expected_text=line.effective_target_text,
                        source_text=line.source_text,
                        target_sample_rate=int(mounted_rate),
                        target_frames=len(mounted_audio),
                        channels=int(mounted_audio.shape[1]),
                        reference_end=len(mounted_audio) / mounted_rate,
                        profile=profile,
                        hard_gates=_line_hard_gates(config, performance),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        require_asr=True,
                        linguistic_evidence=asr_evidence,
                        alignment_evidence=alignment_dict,
                        lid_evidence=lid,
                        alignment_min_target_score=float(getattr(config.qa, "alignment_min_target_score", .65)),
                        alignment_min_margin=float(getattr(config.qa, "alignment_min_margin", .20)),
                        alignment_source_leak_score=float(getattr(config.qa, "alignment_source_leak_score", .75)),
                        **_calibration_kwargs(config, alignment_backend, performance_mode=performance.mode.value),
                    )
                option["eligible"] = bool(regraded.passed and option["serialized_audit"].passed)
                if mfa_backend is not None and 4 in qa_route and not option["eligible"]:
                    try:
                        option["mfa"] = mfa_backend.align(mounted_audit.artifact_path or "", text=line.effective_target_text, language=config.target_language)
                        option["mfa_status"] = "EXECUTED_DIAGNOSTIC"
                        report["mfa"]["executed"] += 1
                    except Exception as exc:
                        option["mfa_status"] = "MFA_UNAVAILABLE"
                        option["mfa_error"] = str(exc)
                if option["eligible"]:
                    break
            except AlignmentUnavailable as exc:
                option["alignment_status"] = "ALIGNER_NOT_APPLICABLE"
                option["alignment_error"] = str(exc)
                break
            except Exception as exc:
                option["alignment_status"] = "ALIGNMENT_ERROR"
                option["alignment_error"] = str(exc)
        _line_linguistic_summary(row, options, expected_text=line.effective_target_text, source_text=line.source_text, target_language=config.target_language)
        row["candidate_stages"] = [_stage_bundle(option) for option in options]

    # Repairs are evidence records, not hidden retries.  A caller may provide
    # a concrete technical executor; without one the outcome remains an
    # explicit BLOCKED_NO_EXECUTOR/HOLD_NO_TTS record.
    for line in scene.lines:
        for option in options_by_line.get(line.id, []):
            if not option.get("eligible"):
                _record_repair(line, option)

    report["stage_evidence"]["LINGUISTIC_ALIGNMENT"] = {
        "status": "EXECUTED" if alignment_count else "NOT_RUN",
        "artifact_count": alignment_count,
        "independent_family": getattr(alignment_backend, "evidence_family", None).value if alignment_backend is not None and hasattr(getattr(alignment_backend, "evidence_family", None), "value") else (str(getattr(alignment_backend, "evidence_family", "")) if alignment_backend is not None else None),
    }

    generable_lines = [line for line in scene.lines if line.id in options_by_line]
    chosen_options: dict[str, dict[str, Any]] = {}
    final_scene_audit: StageAudit
    if scene.topology == "EMBEDDED_FMV":
        def _mount(working, line, option):
            window = _line_window(line, stem_rate, scene.id, scene.dialogue_channel)
            generated, generated_rate = read(option["processed_path"], always_2d=True)
            mounted, _ = mount_surgical(working, generated, generated_rate, window, stem_rate, empalme_b=bool(line.preserved_source_intervals or line.source_resume is not None))
            return mounted
        def _audit(working, index):
            attempt_path = out / "scene_candidates" / f"local_{index:03d}.mounted.wav"
            persist_audio_atomic(attempt_path, working, stem_rate)
            protected_ok, untouched_ok, integrity = _scene_integrity(source_array, working, scene.lines, scene, stem_rate)
            stage_counts["SCENE_QA"] += 1
            audit = audit_scene_stage(attempt_path, expected_sample_rate=stem_rate, expected_frames=len(source_array), expected_channels=source_array.shape[1], protected_intervals_ok=protected_ok, untouched_channels_ok=untouched_ok, line_windows=scene_line_windows)
            audit.diagnostics.update({"selection_strategy":"LOCAL_SCENE_REPAIR","local_attempt":index,"integrity":integrity})
            return audit.passed, audit
        local = select_local_scene(generable_lines, options_by_line, source_array, max_candidates_per_line=int(getattr(config,"scene_candidate_options",8)), max_iterations=max(1,int(getattr(config,"scene_selection_max_combinations",64))), mount_line=_mount, audit_scene=_audit, rank=lambda option: rank_candidate_v2(option["mounted_audit"].result))
        selected_working = local.working if local.passed else (source_array.copy() if not generable_lines else None)
        combo_audit = local.audit
        chosen_options = local.selected if local.passed else {}
        report["fmv_candidate_matrix"] = build_candidate_matrix(generable_lines, options_by_line)
        report["fmv_local_selection"] = {"passed":local.passed,"attempts":local.attempts,"diagnostics":local.matrix}
        if selected_working is not None:
            mounted_output = out / f"{scene.id}.mounted.wav"
            persist_audio_atomic(mounted_output, selected_working, stem_rate)
            protected_ok, untouched_ok, integrity = _scene_integrity(source_array, selected_working, scene.lines, scene, stem_rate)
            final_scene_audit = audit_scene_stage(mounted_output, expected_sample_rate=stem_rate, expected_frames=len(source_array), expected_channels=source_array.shape[1], protected_intervals_ok=protected_ok, untouched_channels_ok=untouched_ok, line_windows=scene_line_windows)
            final_scene_audit.diagnostics.update({"integrity": integrity, "selected": True})
            report["mounted_output"] = str(mounted_output)
            report["scene_qa"] = final_scene_audit.to_dict()
            stage_counts["SCENE_QA"] += 1
            for line in generable_lines:
                option = chosen_options[line.id]
                row = row_by_id[line.id]
                row.update({"status": "FINAL_PASS", "candidate_id": option["candidate"].candidate_id, "processing_hash": option["processing_hash"], "mount": option["mount_metrics"].to_dict() if option["mount_metrics"] is not None else None, "qa": option["mounted_audit"].result.to_dict() if option["mounted_audit"].result is not None else None, "selected_candidate_linguistic_decision": option["mounted_audit"].diagnostics.get("linguistic_decision")})
        else:
            final_scene_audit = combo_audit or StageAudit(stage="SCENE_QA", passed=False, diagnostics={"reason": "NO_ELIGIBLE_CANDIDATE_COMBINATION"}, failure_class=FailureClass.DETERMINISTIC_PROCESSING)
            report["scene_qa"] = final_scene_audit.to_dict()
            report["blockers"].append({"scene_id": scene.id, "reason": "SCENE_QA_FAILED"})
            for line in generable_lines:
                row_by_id[line.id].setdefault("status", "UNPROVEN_HOLD")
    else:
        for line in generable_lines:
            options = [option for option in options_by_line[line.id] if option["eligible"] and option["mounted_audit"].result is not None]
            chosen = select_passed_v2([(option["candidate"], option["mounted_audit"].result) for option in options]) if options else None
            row = row_by_id[line.id]
            if chosen is None:
                row.setdefault("status", "UNPROVEN_HOLD")
                report["blockers"].append({"line_id": line.id, "reason": "NO_POST_TRANSFORM_PASS"})
                continue
            candidate, _result = chosen
            option = next(item for item in options if item["candidate"].candidate_id == candidate.candidate_id)
            chosen_options[line.id] = option
            row.update({"status": "FINAL_PASS", "candidate_id": candidate.candidate_id, "output": option["mounted_path"], "processing_hash": option["processing_hash"], "qa": option["mounted_audit"].result.to_dict(), "selected_candidate_linguistic_decision": option["mounted_audit"].diagnostics.get("linguistic_decision")})
        final_scene_audit = _aggregate_line_scene_audit(report["lines"])
        report["scene_qa"] = final_scene_audit.to_dict()
        stage_counts["SCENE_QA"] = 1

    report["stage_evidence"]["SCENE_QA"] = {"status": "EXECUTED" if stage_counts["SCENE_QA"] else "NOT_RUN", "artifact_count": stage_counts["SCENE_QA"], "passed": final_scene_audit.passed}
    for stage in ("PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA", "LINGUISTIC_ALIGNMENT", "SCENE_QA"):
        if stage_counts[stage] and stage not in report["phases"]:
            report["phases"].append(stage)
    report["pass"] = bool(final_scene_audit.passed and not report["blockers"] and all(row_by_id[line.id].get("status") not in {"UNPROVEN_HOLD", BLOCKED} for line in scene.lines))
    report["model_pool"] = {
        "loaded": [identity.__dict__ for identity in pool.loaded()],
        "load_counts": {str(identity.__dict__): count for identity, count in pool.load_counts().items()},
    }
    run_state.commit(RunState(scene.id, "FINAL", {line_id: str(row.get("status", "UNKNOWN")) for line_id, row in row_by_id.items()}, list(report["blockers"]), cursor=scene.id), {"event": "scene_finished", "scene_id": scene.id, "passed": report["pass"]})
    report["contract_hash"] = contract_hash("scene-v2", {"scene": scene.to_dict(), "config": config.to_dict(), "scene_qa": report["scene_qa"]})
    atomic_json(out / "FINAL_REPORT_V2.json", report)
    repair_store.close()
    if owned_pool:
        pool.close()
    return report


__all__ = ["run_scene_v2"]
