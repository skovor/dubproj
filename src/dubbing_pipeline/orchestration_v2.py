"""Reference implementation of the V2 generation→QA→mount order."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .audio import read, write
from .contracts import DeliveryWindow, FailureClass
from .contracts.manifest import validate_scene_value
from .generation_v2 import GenerationRuntimeV2, generate_cohort_v2
from .hashing import atomic_json, contract_hash
from .lab import SandboxLayout
from .mapping import validate_scene
from .models import Line, Scene
from .montage import mount_surgical
from .policy import BLOCKED, KEEP_ORIGINAL, classify_line
from .processing import process_candidate
from .qa_v2 import LanguageProfile, QAResultV2, evaluate_candidate_v2, select_passed_v2
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


def run_scene_v2(scene: Scene, config: Any, *, runtime: GenerationRuntimeV2, asr: Any = None, stem_path: str | Path | None = None, output_dir: str | Path | None = None, language_profile: LanguageProfile | None = None) -> dict[str, Any]:
    """Run one scene using globally sealed initial/retry cohorts.

    A real adapter should provide the scene object and reference mapping; this
    function never guesses a physical reference or writes outside config roots.
    """
    validate_scene(scene)
    if config.lab_mode:
        if config.sandbox_root is None:
            raise ValueError("V2 lab mode requires sandbox_root")
        SandboxLayout.create(config.sandbox_root).ensure_safe()
    out = Path(output_dir or config.output_root) / scene.id; out.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Any] = {}
    lines = [line for line in scene.lines if classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment).policy not in {KEEP_ORIGINAL, BLOCKED}]
    for line in lines:
        refs[line.id] = materialize_reference(line, config.project_root, config.cache_root, language=config.source_language)

    transcripts: dict[str, tuple[str | None, str | None, float | None]] = {}
    def generate(items: list[Line], round_index: int):
        takes = (config.fmv_initial_takes if scene.topology == "EMBEDDED_FMV" else config.initial_takes) if round_index == 1 else (config.fmv_retry_takes if scene.topology == "EMBEDDED_FMV" else config.retry_takes)
        take_count = int(takes)
        if take_count <= 0:
            return {line.id: [] for line in items}
        return generate_cohort_v2(runtime, [(line, refs[line.id], take_count) for line in items], config, round_index=round_index, cache_root=config.cache_root)
    def evaluate(candidate):
        line = next(item for item in lines if item.id == candidate.line_id)
        path = candidate.raw_audio
        if asr is None:
            transcript, language, probability = transcripts.get(candidate.line_id, (None, None, None))
        else:
            value = asr.transcribe(path)
            if isinstance(value, dict): transcript, language, probability = value.get("text"), value.get("language"), value.get("probability")
            else: transcript, language, probability = str(value), None, None
        target_rate = int(getattr(config, "native_sample_rate", 24000))
        # Raw generation is audited at native rate; window/frame contracts are
        # checked again after processing and montage, never used to reject a
        # valid raw candidate before it can be corrected.
        target_frames = None
        return evaluate_candidate_v2(path, expected_text=line.effective_target_text, source_text=line.source_text, target_sample_rate=target_rate, target_frames=target_frames, channels=1, reference_end=(line.end - line.start) if scene.topology != "LINE_SEPARATED" else None, transcript=transcript, language=language, language_probability=probability, profile=language_profile or LanguageProfile(config.source_language, config.target_language), hard_gates=list(config.qa.hard_gates), final_word_min_tokens=config.qa.final_word_min_tokens, tail_guard_seconds=config.qa.tail_guard_ms / 1000.0, require_asr=True, neutral_effort=False)
    cohort = run_cohorts(lines, item_id=lambda item: item.id, generate=generate, evaluate=evaluate, max_retry_rounds=1)
    report: dict[str, Any] = {"schema": "scene-report-v2", "scene": scene.id, "topology": scene.topology, "run_id": cohort.run_id, "lines": [], "blockers": cohort.blockers, "phases": cohort.phases}
    working_stem = None; stem_rate = None
    if scene.topology == "EMBEDDED_FMV":
        source_path = Path(stem_path or scene.source_stem or "")
        if not source_path.is_file(): raise ValueError("FMV scene requires a source stem")
        working_stem, stem_rate = read(source_path, always_2d=True)
    for line in scene.lines:
        decision = classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment)
        row: dict[str, Any] = {"id": line.id, "policy": decision.policy, "policy_reason": decision.reason}
        if decision.policy in {KEEP_ORIGINAL, BLOCKED}:
            row["status"] = decision.policy; report["lines"].append(row); continue
        evaluations = list(zip(cohort.candidates.get(line.id, []), cohort.evaluations.get(line.id, [])))
        chosen = select_passed_v2(evaluations)
        if chosen is None:
            row["status"] = "UNPROVEN_HOLD"; row["candidates"] = [{"id": candidate.candidate_id, "qa": result.to_dict()} for candidate, result in evaluations]; report["lines"].append(row); continue
        candidate, qa_result = chosen; row.update({"status": "AUTOMATIC_QA_PASS", "candidate_id": candidate.candidate_id, "qa": qa_result.to_dict()})
        processed = process_candidate(candidate.raw_audio, target_sample_rate=int(stem_rate or config.sample_rate), reference_end=(line.end - line.start) if scene.topology != "LINE_SEPARATED" else None, ffmpeg=config.ffmpeg, tmpdir=out / "processing")
        if scene.topology == "EMBEDDED_FMV":
            window = _line_window(line, int(stem_rate), scene.id, scene.dialogue_channel)
            working_stem, metrics = mount_surgical(working_stem, processed.audio, processed.sample_rate, window, int(stem_rate), empalme_b=bool(line.preserved_source_intervals or line.source_resume is not None))
            row["mount"] = metrics.to_dict()
        else:
            target = out / f"{line.id}.wav"; write(target, processed.audio, processed.sample_rate); row["output"] = str(target)
        row["processing_hash"] = processed.processing_hash; report["lines"].append(row)
    if working_stem is not None:
        mounted = out / f"{scene.id}.mounted.wav"; write(mounted, working_stem, int(stem_rate)); report["mounted_output"] = str(mounted)
    report["pass"] = not report["blockers"] and all(row.get("status") not in {"UNPROVEN_HOLD", "BLOCKED"} for row in report["lines"])
    report["contract_hash"] = contract_hash("scene-v2", {"scene": scene.to_dict(), "config": config.to_dict()})
    atomic_json(out / "FINAL_REPORT_V2.json", report)
    return report


__all__ = ["run_scene_v2"]
