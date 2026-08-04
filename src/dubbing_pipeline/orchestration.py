"""Config-driven orchestration for independent lines and embedded FMV."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .audio import align_onset_exact, read, resample_exact, write
from .config import PipelineConfig
from .generation import GenerationRuntime, SpeechBackend, generate_candidates, persist_candidates
from .hashing import atomic_json, contract_hash
from .mapping import validate_scene
from .models import Candidate, Scene
from .policy import BLOCKED, KEEP_ORIGINAL, classify_line
from .qa import GateResult, evaluate_candidate, select_passed
from .reference import materialize_reference


def _reference_for(line, project_root: Path) -> str | None:
    if line.reference_audio:
        path = Path(line.reference_audio)
        return str(path if path.is_absolute() else (project_root / path).resolve())
    if line.reference_segments:
        path = Path(line.reference_segments[0].path)
        return str(path if path.is_absolute() else (project_root / path).resolve())
    return None


def _transcribe(asr: Any, path: Path) -> tuple[str, str | None, float | None]:
    if asr is None:
        return "", None, None
    value = asr.transcribe(path)
    if isinstance(value, dict):
        return str(value.get("text", "")), value.get("language"), value.get("probability")
    return str(value), None, None


def _target_frames(line, sample_rate: int) -> int:
    return max(1, round((float(line.end) - float(line.start)) * sample_rate))


def _mount_line(stem, generated, line, sample_rate: int, dialogue_channel: int):
    """Replace only one authorized window in one dialogue channel."""
    import numpy as np
    source = np.asarray(stem, dtype="float32")
    if source.ndim != 2:
        raise ValueError("FMV stem must be a 2-D multi-channel array")
    if not 0 <= dialogue_channel < source.shape[1]:
        raise ValueError(f"dialogue channel {dialogue_channel} outside {source.shape[1]} channels")
    start, end = round(line.start * sample_rate), round(line.end * sample_rate)
    if end <= start or end > len(source):
        raise ValueError(f"line window outside stem: {line.id} {start}:{end} / {len(source)}")
    body = generated
    generated_rate = sample_rate
    if isinstance(generated, tuple):
        body, generated_rate = generated
    if generated_rate != sample_rate:
        body = resample_exact(body, generated_rate, sample_rate)
    replacement = align_onset_exact(source[start:end, dialogue_channel], body, sample_rate, end - start)
    result = source.copy()
    result[start:end, dialogue_channel] = replacement
    return result


def run_scene(scene: Scene, config: PipelineConfig, *, runtime: GenerationRuntime | None = None, backend: SpeechBackend | None = None, asr: Any = None, stem_path: str | Path | None = None, output_dir: str | Path | None = None, evaluator: Callable[[Path, Any], tuple[str, str | None, float | None]] | None = None) -> dict[str, Any]:
    """Run mapping/policy/generation/QA/mount for one scene.

    The function is intentionally dependency-light until a TTS candidate is
    requested. A production caller normally supplies one persistent
    `GenerationRuntime` to process all scenes in a round.
    """
    validate_scene(scene)
    out = Path(output_dir or config.output_root) / scene.id
    out.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        if backend is None:
            raise RuntimeError("a SpeechBackend or GenerationRuntime is required for generable lines")
        runtime = GenerationRuntime(backend)
    report: dict[str, Any] = {"schema": "generic-dubbing-scene-report-v1", "scene": scene.id, "topology": scene.topology, "lines": [], "review": []}
    stem = None; stem_rate = config.sample_rate
    if scene.topology == "EMBEDDED_FMV":
        if not stem_path:
            raise ValueError("EMBEDDED_FMV requires stem_path")
        stem, stem_rate = read(stem_path, always_2d=True)
        working_stem = stem.copy()
    else:
        working_stem = None
    for line in scene.lines:
        decision = classify_line(line, append_ellipsis_experiment=config.append_ellipsis_experiment)
        row: dict[str, Any] = {"id": line.id, "policy": decision.policy, "policy_reason": decision.reason, "target_text": line.effective_target_text}
        if decision.policy == KEEP_ORIGINAL:
            row["status"] = "KEEP_ORIGINAL"; report["lines"].append(row); continue
        if decision.policy == BLOCKED:
            row["status"] = "REVIEW"; report["review"].append({"id": line.id, "reason": decision.reason}); report["lines"].append(row); continue
        try:
            reference = materialize_reference(line, config.project_root, config.cache_root, language=config.source_language).audio_path
        except Exception:
            # Keep the legacy diagnostic path for callers that explicitly use
            # an external reference provider, but never silently mix a segment
            # transcript with the full stem when materialization is possible.
            reference = _reference_for(line, config.project_root)
        if not reference or not Path(reference).is_file():
            row.update({"status": "REVIEW", "failure": "MISSING_REFERENCE"}); report["review"].append({"id": line.id, "reason": "MISSING_REFERENCE"}); report["lines"].append(row); continue
        candidates = generate_candidates(runtime, line, reference, config, cache_root=config.cache_root)
        evaluations: list[tuple[Candidate, GateResult]] = []

        def evaluate_rows(rows: list[Candidate]) -> None:
            for candidate in rows:
                candidate_path = Path(candidate.path)
                if evaluator:
                    transcript, language, probability = evaluator(candidate_path, line)
                else:
                    transcript, language, probability = _transcribe(asr, candidate_path)
                # Raw QA is measured at the producer's native rate. Window
                # frames/tail are evaluated after processing and montage; a
                # raw candidate must not be rejected merely because it still
                # needs explicit resampling or duration correction.
                native_rate = int(getattr(config, "native_sample_rate", stem_rate))
                result = evaluate_candidate(candidate.path, line, target_sample_rate=native_rate, target_frames=None, reference_end=None, transcript=transcript, language=language, language_probability=probability, config=config)
                candidate.passed = result.passed; candidate.hard_gates = result.gates; candidate.diagnostics = result.diagnostics; candidate.failure_class = result.failure_class
                evaluations.append((candidate, result))

        evaluate_rows(candidates)
        if not any(result.passed for _, result in evaluations):
            retry_count = config.fmv_retry_takes if line.topology == "EMBEDDED_FMV" else config.retry_takes
            if retry_count:
                retry_candidates = generate_candidates(runtime, line, reference, config, round_index=2, takes=retry_count, cache_root=config.cache_root)
                candidates.extend(retry_candidates)
                evaluate_rows(retry_candidates)
        chosen = select_passed(evaluations)
        if chosen is None:
            row.update({"status": "REVIEW", "failure": "NO_HARD_GATE_PASS", "candidates": [item.to_dict() for item in candidates]}); report["review"].append({"id": line.id, "reason": "NO_HARD_GATE_PASS"})
        else:
            candidate, result = chosen; row.update({"status": "PASS", "candidate": candidate.to_dict(), "qa": result.diagnostics})
            if scene.topology == "EMBEDDED_FMV":
                working_stem = _mount_line(working_stem, read(candidate.path), line, stem_rate, scene.dialogue_channel)
            else:
                target = out / f"{line.id}.{Path(candidate.path).suffix.lstrip('.') or 'wav'}"; shutil.copy2(candidate.path, target); row["output"] = str(target)
        persist_candidates(out / f"{line.id}.candidates.json", candidates)
        report["lines"].append(row)
    if working_stem is not None:
        mounted = out / f"{scene.id}.mounted.wav"; write(mounted, working_stem, stem_rate); report["mounted_output"] = str(mounted)
    report["contract_hash"] = contract_hash("scene", {"scene": scene.to_dict(), "config": config.to_dict()})
    report["pass"] = not report["review"]
    atomic_json(out / "FINAL_REPORT.json", report)
    return report
