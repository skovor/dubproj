#!/usr/bin/env python3
"""Global FMV/anime scheduler for the active CODEX2 producer.

Generation is scene-global per round (one OmniVoice model, distinct-unit
batching); QA is scene-global per round (one Whisper/MMS runtime and caches).
The script resumes existing artifacts and never touches the KIRO tree.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

PIPELINE = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
PROJECT = Path(r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project")
SCRIPTS = PROJECT / "scripts"
MAP_ROOT = PIPELINE / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v3_codex2"
DEFAULT_OUT = PIPELINE / "P3R_ANIME_REPAIR_20260802_CODEX2"
_GLOBAL_LOCK_HANDLE = None


def acquire_global_lock(output_root: Path) -> None:
    """Prevent concurrent OmniVoice schedulers from duplicating RAM/VRAM use."""
    global _GLOBAL_LOCK_HANDLE
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".global_round.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    handle.seek(0)
    if handle.tell() == 0:
        handle.write("0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise SystemExit(
                "Another run_global_fmv_rounds.py is already active; "
                "refusing to load a second OmniVoice model."
            ) from exc
    _GLOBAL_LOCK_HANDLE = handle
    atexit.register(handle.close)
for path in (PIPELINE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import produce_anime_scene as producer


def run_continuous_language_audits(jobs: list[tuple], config: dict) -> list[dict]:
    """Decode every mounted dialogue stem once with a persistent Whisper model."""
    from audit_final_scene_language import audit_scene

    qa = config["qa"]
    model = producer.WhisperModel(
        qa.get("asr_model", "large-v3-turbo"),
        device=qa.get("asr_device", "cuda"),
        compute_type=qa.get("asr_compute_type", "float16"),
    )
    results = []
    try:
        for scene, _, _, out, _, _ in jobs:
            started = time.perf_counter()
            result = audit_scene(out, model, scene=scene)
            producer.stage_timing(
                out, "continuous_language_audit", started,
                global_runtime=True,
            )
            results.append(result)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def load_scene(map_path: Path, config: dict, output_root: Path) -> tuple[dict, object, int, Path, Path, Path]:
    scene = json.loads(map_path.read_text(encoding="utf-8"))
    scene["_codex2_contract_hash"] = producer.scene_contract_hash(
        scene, map_path, config, Path(producer.__file__).resolve(), PROJECT,
    )
    for line in scene.get("lines", []):
        line["_codex2_line_contract_hash"] = producer.line_contract_hash(
            scene, line, map_path, config, Path(producer.__file__).resolve(), PROJECT,
        )
        line["_codex2_generation_hash"] = producer.generation_contract_hash(
            scene, line, map_path, config, PROJECT,
        )
        line["_codex2_processing_hash"] = producer.processing_contract_hash(
            scene, line, map_path, config, Path(producer.__file__).resolve(), PROJECT,
        )
        line["_codex2_qa_hash"] = producer.qa_contract_hash(
            config["qa"], Path(producer.__file__).resolve(),
        )
    scene["_codex2_mount_hash"] = producer.mount_contract_hash(
        scene, config, Path(producer.__file__).resolve(),
    )
    stem_path = producer.resolve(scene["source_stem"], map_path.parent)
    full_path = stem_path.with_name(stem_path.name.replace("_dialog_ch5", "_6ch"))
    stem, sr = producer.read(stem_path)
    out = output_root / scene["scene"]
    out.mkdir(parents=True, exist_ok=True)
    producer.attach_accepted_contracts(scene, out)
    return scene, stem, sr, out, stem_path, full_path


def merge_rankings(path: Path, focused: dict[str, list[dict]]) -> dict[str, list[dict]]:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for line_id, rows in focused.items():
        by_file = {row.get("file"): row for row in existing.get(line_id, [])}
        by_file.update({row.get("file"): row for row in rows})
        existing[line_id] = list(by_file.values())
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return existing


def unload_generation(runtime: producer.GenerationRuntime | None) -> None:
    if runtime is not None:
        runtime.model = None
        runtime.prompt_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def current_rows_for_ids(rankings: dict[str, list[dict]], ids: set[str]) -> dict[str, list[dict]]:
    """Return only rows relevant to the current global QA decision."""
    return {line_id: rows for line_id, rows in rankings.items() if line_id in ids}


def retryable_ids(
    scene: dict,
    focused: dict[str, list[dict]],
    only_ids: set[str] | None,
) -> set[str]:
    """Retry only stochastic TTS failures; mapping/window failures are deterministic."""
    eligible = {
        line["id"] for line in scene.get("lines", [])
        if only_ids is None or line["id"] in only_ids
    }
    result = set()
    for line_id in eligible:
        rows = focused.get(line_id, [])
        if rows and not any(row.get("pass") for row in rows):
            classes = {row.get("failure_class") for row in rows}
            if "RANDOM_TTS" in classes:
                result.add(line_id)
    return result


def generation_work_exists(
    scene: dict,
    out: Path,
    profile: dict,
    round_index: int,
    only_ids: set[str] | None,
) -> bool:
    """Avoid loading CUDA/model state when a resumable round is already complete."""
    metadata_path = out / "candidates" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    accepted = {
        line["id"]: set(line.get("_codex2_accepted_line_contract_hashes", []))
        for line in scene.get("lines", [])
    }
    takes = int(profile["initial_takes"] if round_index == 0 else profile["retry_takes"])
    for line in scene.get("lines", []):
        if only_ids is not None and line["id"] not in only_ids:
            continue
        if producer.decide_line(line).action in {
            producer.KEEP_ORIGINAL, producer.BLOCKED,
        }:
            continue
        capacity_ok, _ = producer.window_capacity(line)
        if not capacity_ok:
            continue
        count = sum(
            row.get("line_id") == line["id"]
            and int(row.get("round", -1)) == round_index
            and row.get("contract_hash") in accepted.get(line["id"], set())
            for row in metadata
        )
        if count < takes:
            return True
    return False


def write_global_summary(
    output_root: Path,
    reports: list[dict],
    started: float,
    rounds: list[dict],
    *,
    run_id: str,
    global_run: bool,
) -> dict:
    """Persist release state and efficiency telemetry in one auditable artifact."""
    stage_seconds: dict[str, float] = {}
    candidate_count = 0
    for report in reports:
        out = output_root / report["scene"]
        timing_path = out / "STAGE_TIMINGS.json"
        if timing_path.exists():
            for item in json.loads(timing_path.read_text(encoding="utf-8")):
                stage = str(item.get("stage", "unknown"))
                stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(item.get("seconds", 0.0))
        metadata_path = out / "candidates" / "metadata.json"
        if metadata_path.exists():
            candidate_count += len(json.loads(metadata_path.read_text(encoding="utf-8")))
    required = sum(len(r["required_voice_ids"]) for r in reports)
    mounted = sum(len(r["mounted_voice_ids"]) for r in reports)
    summary = {
        "run_id": run_id,
        "run_scope": "global" if global_run else "targeted",
        "scenes": len(reports),
        "required": required,
        "mounted": mounted,
        "review": sorted(
            (r["scene"], line_id)
            for r in reports for line_id in r.get("missing_current_candidate_ids", [])
        ),
        "release_ready": all(bool(r.get("release_ready")) for r in reports),
        "rounds": rounds,
        "telemetry": {
            "stage_seconds": stage_seconds,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "candidate_rows_persisted": candidate_count,
            "mounted_lines_per_minute_wall": round(
                mounted / max((time.perf_counter() - started) / 60.0, 1e-6), 3,
            ),
        },
    }
    summary_name = "GLOBAL_ROUND_SUMMARY.json" if global_run else f"RUN_SUMMARY_{run_id}.json"
    (output_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def load_generation(profile: dict) -> producer.GenerationRuntime:
    model = producer.OmniVoice.from_pretrained(
        profile.get("model", "k2-fsa/OmniVoice"),
        device_map="cuda", dtype=torch.float16,
    )
    return producer.GenerationRuntime(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, default=MAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--regenerate-ids", nargs="*")
    parser.add_argument("--mount-only", action="store_true")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument(
        "--skip-continuous-audit", action="store_true",
        help="Skip the final full-scene language audit (debug only).",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Stable artifact namespace; defaults to UTC timestamp.",
    )
    args = parser.parse_args()
    config = json.loads(producer.CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    acquire_global_lock(output_root)
    wanted_scenes = set(args.scenes or [])
    wanted_ids = set(args.regenerate_ids or [])
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    global_run = not wanted_scenes and not wanted_ids
    (output_root / "RUN_METADATA.json").write_text(
        json.dumps({
            "run_id": run_id,
            "scope": "global" if global_run else "targeted",
            "scenes": sorted(wanted_scenes),
            "line_ids": sorted(wanted_ids),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    map_paths = sorted(args.maps.resolve().glob("*_map.json"))
    if wanted_scenes:
        map_paths = [p for p in map_paths if p.stem.removesuffix("_map") in wanted_scenes]
    jobs = [load_scene(path, config, output_root) for path in map_paths]
    if not jobs:
        raise SystemExit("No maps selected")
    profile = dict(config["anime"])
    profile["model"] = config["model"]
    max_rounds = args.max_rounds or int(profile.get("max_rounds", 2))

    if args.mount_only:
        global_started = time.perf_counter()
        reports = []
        for scene, stem, sr, out, stem_path, full_path in jobs:
            rankings_path = out / "QA_RANKING.json"
            rankings = json.loads(rankings_path.read_text(encoding="utf-8")) if rankings_path.exists() else {}
            mount_started = time.perf_counter()
            report = producer.select_and_mount(scene, stem_path, full_path, out, rankings)
            producer.stage_timing(out, "mount", mount_started, global_runtime=True, mount_only=True)
            producer.make_html(scene, stem_path, report, out)
            reports.append(report)
        if not args.skip_continuous_audit:
            run_continuous_language_audits(jobs, config)
            reports = []
            for scene, _, _, out, stem_path, full_path in jobs:
                rankings_path = out / "QA_RANKING.json"
                # Subtitle-free/KEEP_ORIGINAL maps have no QA file.  They
                # still need a final report and language audit, but an empty
                # ranking is the correct input for their mount pass.
                rankings = (
                    json.loads(rankings_path.read_text(encoding="utf-8"))
                    if rankings_path.exists() else {}
                )
                report = producer.select_and_mount(scene, stem_path, full_path, out, rankings)
                producer.make_html(scene, stem_path, report, out)
                reports.append(report)
        summary = write_global_summary(
            output_root, reports, global_started, [{"mode": "mount_only"}],
            run_id=run_id, global_run=global_run,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # A targeted repair starts after the latest persisted round for each ID;
    # the normal path starts with the canonical global round 0.
    round_by_scene: dict[str, int] = {}
    for scene, _, _, out, _, _ in jobs:
        metadata_path = out / "candidates" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
        if wanted_ids:
            ids = {line["id"] for line in scene["lines"]} & wanted_ids
            round_by_scene[scene["scene"]] = max(
                [int(row["round"]) for row in metadata if row.get("line_id") in ids] or [-1]
            ) + 1
        else:
            round_by_scene[scene["scene"]] = 0

    global_started = time.perf_counter()
    generation: producer.GenerationRuntime | None = None
    rounds_done: list[dict] = []
    pending_ids: dict[str, set[str] | None] = {
        scene["scene"]: (
            {line["id"] for line in scene["lines"]} & wanted_ids
            if wanted_ids else None
        )
        for scene, _, _, _, _, _ in jobs
    }
    try:
        # A focused repair intentionally performs one round.  The normal path
        # runs a global QA -> retry loop while preserving one model instance.
        total_rounds = 1 if wanted_ids else max(1, max_rounds)
        for local_round in range(total_rounds):
            round_info = {"round": local_round, "generated_scenes": 0, "retry_ids": {}}
            for scene, stem, sr, out, _, _ in jobs:
                target = pending_ids[scene["scene"]]
                round_index = round_by_scene[scene["scene"]] + local_round
                if target is not None and not target:
                    continue
                if not generation_work_exists(scene, out, profile, round_index, target):
                    continue
                if generation is None:
                    generation = load_generation(profile)
                started = time.perf_counter()
                producer.generate_round(scene, stem, sr, out, profile, round_index, target, runtime=generation)
                producer.stage_timing(out, "generation", started, round=round_index, global_runtime=True)
                round_info["generated_scenes"] += 1
            rounds_done.append(round_info)

            qa_runtime = producer.QARuntime()
            # Keep scenes with no synthesizable subtitle rows in the retry
            # table as an explicit empty set.  Otherwise a subtitle-free map
            # is skipped during QA and disappears from next_pending; the next
            # global round then indexes a missing scene key and the scheduler
            # aborts with KeyError after all prior work is done.
            next_pending: dict[str, set[str]] = {
                scene["scene"]: set() for scene, _, _, _, _, _ in jobs
            }
            try:
                for scene, stem, sr, out, _, _ in jobs:
                    target = pending_ids[scene["scene"]]
                    if target is not None and not target:
                        continue
                    round_index = round_by_scene[scene["scene"]] + local_round
                    if not (out / "candidates" / "metadata.json").exists():
                        continue
                    started = time.perf_counter()
                    focused, _ = producer.evaluate(
                        scene, stem, sr, out, config["qa"],
                        only_ids=target, only_rounds={round_index}, runtime=qa_runtime,
                    )
                    producer.stage_timing(out, "qa", started, round=round_index, global_runtime=True)
                    merge_rankings(out / "QA_RANKING.json", focused)
                    next_pending[scene["scene"]] = retryable_ids(scene, focused, target)
                    rounds_done[-1]["retry_ids"][scene["scene"]] = sorted(next_pending[scene["scene"]])
            finally:
                qa_runtime.asr = None
                qa_runtime.mms = None
                gc.collect()
                torch.cuda.empty_cache()
            if wanted_ids or not any(next_pending.values()):
                break
            pending_ids = next_pending
    finally:
        unload_generation(generation)

    reports = []
    for scene, stem, sr, out, stem_path, full_path in jobs:
        rankings_path = out / "QA_RANKING.json"
        rankings = json.loads(rankings_path.read_text(encoding="utf-8")) if rankings_path.exists() else {}
        started = time.perf_counter()
        report = producer.select_and_mount(scene, stem_path, full_path, out, rankings)
        producer.stage_timing(out, "mount", started, global_runtime=True)
        producer.make_html(scene, stem_path, report, out)
        reports.append(report)
    if not args.skip_continuous_audit:
        run_continuous_language_audits(jobs, config)
        reports = []
        for scene, _, _, out, stem_path, full_path in jobs:
            rankings_path = out / "QA_RANKING.json"
            rankings = (
                json.loads(rankings_path.read_text(encoding="utf-8"))
                if rankings_path.exists() else {}
            )
            report = producer.select_and_mount(scene, stem_path, full_path, out, rankings)
            producer.make_html(scene, stem_path, report, out)
            reports.append(report)
    summary = write_global_summary(
        output_root, reports, global_started, rounds_done,
        run_id=run_id, global_run=global_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
