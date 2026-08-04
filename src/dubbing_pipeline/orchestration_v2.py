"""V2 orchestration with post-transform and post-serialization QA.

The raw TTS candidate is only a probe.  A line becomes deliverable after its
processed file, mounted delivery, serialized artifact and (for FMV) complete
scene have all passed their own evidence-backed audits.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

from .audio import read
from .contracts import DeliveryWindow, FailureClass
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
from .qa_v2 import LanguageProfile, QAResultV2, evaluate_candidate_v2, rank_candidate_v2, select_passed_v2
from .reference import materialize_reference
from .scheduler import run_cohorts


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


def _transcribe_path(asr: Any, path: str | Path, cache: dict[str, tuple[str | None, str | None, float | None]]) -> tuple[str | None, str | None, float | None]:
    """Transcribe each distinct serialized artifact once."""
    if asr is None:
        return None, None, None
    key = sha256_file(path)
    if key in cache:
        return cache[key]
    value = asr.transcribe(str(path))
    if isinstance(value, dict):
        result = (value.get("text"), value.get("language"), value.get("probability"))
    else:
        result = (str(value), None, None)
    cache[key] = result
    return result


def _profile(config: Any, supplied: LanguageProfile | None) -> LanguageProfile:
    if supplied is not None:
        return supplied
    return LanguageProfile(
        source_language=config.source_language,
        target_language=config.target_language,
        source_markers=tuple(getattr(config.qa, "english_markers", ()) or ()),
        strong_source_words=tuple(getattr(config.qa, "strong_source_words", ()) or ()),
    )


def _result_or_failed(audit: StageAudit) -> QAResultV2:
    if audit.result is not None:
        return audit.result
    return QAResultV2(False, audit.gates, audit.diagnostics, audit.failure_class)


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
        "error": option.get("error"),
    }


def _aggregate_line_scene_audit(rows: list[dict[str, Any]]) -> StageAudit:
    passed = all(row.get("status") in {"FINAL_PASS", KEEP_ORIGINAL} for row in rows)
    qa_hash = contract_hash("scene-qa-v2", {"topology": "LINE_SEPARATED", "statuses": [row.get("status") for row in rows]})
    return StageAudit(stage="SCENE_QA", passed=passed, qa_hash=qa_hash, diagnostics={"topology": "LINE_SEPARATED", "line_count": len(rows)})


def run_scene_v2(scene: Scene, config: Any, *, runtime: GenerationRuntimeV2, asr: Any = None, stem_path: str | Path | None = None, output_dir: str | Path | None = None, language_profile: LanguageProfile | None = None) -> dict[str, Any]:
    """Run one scene and select only candidates that survive delivery QA.

    Raw QA is deliberately a filter, not the final decision.  Every surviving
    candidate is processed, mounted into a disposable scene, reopened and
    audited.  FMV winners are selected only from combinations that pass the
    complete scene audit.
    """
    validate_scene(scene)
    if config.lab_mode:
        if config.sandbox_root is None:
            raise ValueError("V2 lab mode requires sandbox_root")
        SandboxLayout.create(config.sandbox_root).ensure_safe()
    out = Path(output_dir or config.output_root) / scene.id
    out.mkdir(parents=True, exist_ok=True)
    profile = _profile(config, language_profile)
    refs: dict[str, Any] = {}
    lines = [line for line in scene.lines if classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment).policy not in {KEEP_ORIGINAL, BLOCKED}]
    for line in lines:
        refs[line.id] = materialize_reference(line, config.project_root, config.cache_root, language=config.source_language)

    transcript_cache: dict[str, tuple[str | None, str | None, float | None]] = {}
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
        transcript, language, probability = _transcribe_path(asr, candidate.raw_audio, transcript_cache)
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
            hard_gates=list(config.qa.hard_gates),
            final_word_min_tokens=config.qa.final_word_min_tokens,
            tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
            require_asr=True,
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
    }
    stage_counts = {name: 0 for name in ("RAW_TECHNICAL_QA", "PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA", "SCENE_QA")}
    stage_counts["RAW_TECHNICAL_QA"] = sum(len(value) for value in cohort.evaluations.values())
    report["stage_evidence"]["RAW_TECHNICAL_QA"] = {"status": "EXECUTED", "artifact_count": stage_counts["RAW_TECHNICAL_QA"]}

    source_array = None
    stem_rate = None
    if scene.topology == "EMBEDDED_FMV":
        source_path = Path(stem_path or scene.source_stem or "")
        if not source_path.is_file():
            raise ValueError("FMV scene requires a source stem")
        source_array, stem_rate = read(source_path, always_2d=True)
        source_array = source_array.copy()
        stem_rate = int(stem_rate)

    options_by_line: dict[str, list[dict[str, Any]]] = {}
    row_by_id: dict[str, dict[str, Any]] = {}
    for line in scene.lines:
        decision = classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment)
        row: dict[str, Any] = {"id": line.id, "policy": decision.policy, "policy_reason": decision.reason, "stages": {}}
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
            if raw_audit is None or not raw_audit.passed:
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
                processed_transcript, processed_language, processed_probability = _transcribe_path(asr, processed_path, transcript_cache)
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
                    hard_gates=list(config.qa.hard_gates),
                    final_word_min_tokens=config.qa.final_word_min_tokens,
                    tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                    require_asr=True,
                )
                if not processed_audit.passed:
                    stage_rows.append({"raw_audit": raw_audit, "processed_audit": processed_audit, "mounted_audit": None, "serialized_audit": None})
                    continue
                if scene.topology == "EMBEDDED_FMV":
                    window = _line_window(line, stem_rate, scene.id, scene.dialogue_channel)
                    mounted_array, mount_metrics = mount_surgical(source_array, read(processed_path, always_2d=True)[0], target_rate, window, stem_rate, empalme_b=bool(line.preserved_source_intervals or line.source_resume is not None))
                    mounted_path = persist_audio_atomic(candidate_root / "mounted.wav", mounted_array, stem_rate)
                    delivery_clip, clip_frames = _line_delivery_clip(mounted_array, line, scene, stem_rate)
                    clip_path = persist_audio_atomic(candidate_root / "mounted_line.wav", delivery_clip, stem_rate)
                    mounted_transcript, mounted_language, mounted_probability = _transcribe_path(asr, clip_path, transcript_cache)
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
                        hard_gates=list(config.qa.hard_gates),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        preserved_ok=mount_metrics.preserved_hash_before == mount_metrics.preserved_hash_after,
                        require_asr=True,
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
                    mounted_transcript, mounted_language, mounted_probability = _transcribe_path(asr, mounted_path, transcript_cache)
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
                        hard_gates=list(config.qa.hard_gates),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        require_asr=True,
                    )
                    stage_counts["SERIALIZED_QA"] += 1
                    serialized_transcript, serialized_language, serialized_probability = _transcribe_path(asr, mounted_path, transcript_cache)
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
                        hard_gates=list(config.qa.hard_gates),
                        final_word_min_tokens=config.qa.final_word_min_tokens,
                        tail_guard_seconds=config.qa.tail_guard_ms / 1000.0,
                        require_asr=True,
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
                    "eligible": bool(mounted_audit.passed and serialized_audit.passed),
                })
                stage_rows.append(options[-1])
            except Exception as exc:
                # Processing/mounting failures are evidence for this candidate,
                # not permission to select the earlier raw PASS.
                row.setdefault("transform_errors", []).append({"candidate_id": candidate.candidate_id, "error": str(exc)})
                stage_rows.append({"raw_audit": raw_audit, "processed_audit": None, "mounted_audit": None, "serialized_audit": None, "error": str(exc)})
        options_by_line[line.id] = options
        row["candidate_stages"] = [_stage_bundle(option) for option in stage_rows]
        report["lines"].append(row)

    for stage in ("PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA"):
        report["stage_evidence"][stage] = {"status": "EXECUTED" if stage_counts[stage] else "NOT_RUN", "artifact_count": stage_counts[stage]}

    generable_lines = [line for line in scene.lines if line.id in options_by_line]
    chosen_options: dict[str, dict[str, Any]] = {}
    final_scene_audit: StageAudit
    if scene.topology == "EMBEDDED_FMV":
        eligible_lists = []
        missing = False
        for line in generable_lines:
            eligible = [option for option in options_by_line[line.id] if option["eligible"]]
            eligible.sort(key=lambda option: rank_candidate_v2(option["mounted_audit"].result), reverse=True)
            if not eligible:
                missing = True
            eligible_lists.append(eligible[: max(1, int(getattr(config, "scene_candidate_options", 8)))])
        selected_working = source_array.copy() if not generable_lines else None
        combo_audit = None
        max_combinations = max(1, int(getattr(config, "scene_selection_max_combinations", 64)))
        if not missing and eligible_lists:
            for combo_index, combo in enumerate(itertools.product(*eligible_lists), start=1):
                if combo_index > max_combinations:
                    break
                working = source_array.copy()
                combo_failed = None
                for line, option in zip(generable_lines, combo):
                    try:
                        window = _line_window(line, stem_rate, scene.id, scene.dialogue_channel)
                        generated, generated_rate = read(option["processed_path"], always_2d=True)
                        working, _ = mount_surgical(working, generated, generated_rate, window, stem_rate, empalme_b=bool(line.preserved_source_intervals or line.source_resume is not None))
                    except Exception as exc:
                        combo_failed = str(exc)
                        break
                if combo_failed:
                    continue
                attempt_path = out / "scene_candidates" / f"combination_{combo_index:03d}.mounted.wav"
                persist_audio_atomic(attempt_path, working, stem_rate)
                protected_ok, untouched_ok, integrity = _scene_integrity(source_array, working, scene.lines, scene, stem_rate)
                stage_counts["SCENE_QA"] += 1
                combo_audit = audit_scene_stage(attempt_path, expected_sample_rate=stem_rate, expected_frames=len(source_array), expected_channels=source_array.shape[1], protected_intervals_ok=protected_ok, untouched_channels_ok=untouched_ok)
                combo_audit.diagnostics.update({"combination": combo_index, "integrity": integrity})
                if combo_audit.passed:
                    selected_working = working
                    chosen_options = {line.id: option for line, option in zip(generable_lines, combo)}
                    break
        if selected_working is not None:
            mounted_output = out / f"{scene.id}.mounted.wav"
            persist_audio_atomic(mounted_output, selected_working, stem_rate)
            protected_ok, untouched_ok, integrity = _scene_integrity(source_array, selected_working, scene.lines, scene, stem_rate)
            final_scene_audit = audit_scene_stage(mounted_output, expected_sample_rate=stem_rate, expected_frames=len(source_array), expected_channels=source_array.shape[1], protected_intervals_ok=protected_ok, untouched_channels_ok=untouched_ok)
            final_scene_audit.diagnostics.update({"integrity": integrity, "selected": True})
            report["mounted_output"] = str(mounted_output)
            report["scene_qa"] = final_scene_audit.to_dict()
            stage_counts["SCENE_QA"] += 1
            for line in generable_lines:
                option = chosen_options[line.id]
                row = row_by_id[line.id]
                row.update({"status": "FINAL_PASS", "candidate_id": option["candidate"].candidate_id, "processing_hash": option["processing_hash"], "mount": option["mount_metrics"].to_dict() if option["mount_metrics"] is not None else None, "qa": option["mounted_audit"].result.to_dict() if option["mounted_audit"].result is not None else None})
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
            row.update({"status": "FINAL_PASS", "candidate_id": candidate.candidate_id, "output": option["mounted_path"], "processing_hash": option["processing_hash"], "qa": option["mounted_audit"].result.to_dict()})
        final_scene_audit = _aggregate_line_scene_audit(report["lines"])
        report["scene_qa"] = final_scene_audit.to_dict()
        stage_counts["SCENE_QA"] = 1

    report["stage_evidence"]["SCENE_QA"] = {"status": "EXECUTED" if stage_counts["SCENE_QA"] else "NOT_RUN", "artifact_count": stage_counts["SCENE_QA"], "passed": final_scene_audit.passed}
    for stage in ("PROCESSED_QA", "MOUNTED_QA", "SERIALIZED_QA", "SCENE_QA"):
        if stage_counts[stage] and stage not in report["phases"]:
            report["phases"].append(stage)
    report["pass"] = bool(final_scene_audit.passed and not report["blockers"] and all(row_by_id[line.id].get("status") not in {"UNPROVEN_HOLD", BLOCKED} for line in scene.lines))
    report["contract_hash"] = contract_hash("scene-v2", {"scene": scene.to_dict(), "config": config.to_dict(), "scene_qa": report["scene_qa"]})
    atomic_json(out / "FINAL_REPORT_V2.json", report)
    return report


__all__ = ["run_scene_v2"]
