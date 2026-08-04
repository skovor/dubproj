#!/usr/bin/env python3
"""Re-QA compatible historical FMV candidates under the current contract.

Generation and deterministic processing are separate stages.  A processing
or QA fix must not force a new OmniVoice call when the reference, synthesis
text and generation parameters are unchanged.  This helper promotes only
those candidates, preserving the old hash in ``revalidated_from_contract``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PIPELINE = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
PROJECT = Path(r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project")
MAP_ROOT = PIPELINE / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v3_codex2"
DEFAULT_OUT = PIPELINE / "P3R_ANIME_REPAIR_20260802_CODEX2"
for path in (PIPELINE, PROJECT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import produce_anime_scene as producer
from run_global_fmv_rounds import load_scene, merge_rankings


def _generation_compatible(
    row: dict, line: dict, decision: producer.Decision,
    reference_path: Path, profile: dict,
) -> bool:
    """Conservatively prove that a waveform can be reused for re-QA."""
    if not Path(row.get("processed") or "").exists() and not (
        reference_path.parent.parent / "candidates" / row.get("file", "")
    ).exists():
        return False
    spoken_target = line.get("delivery_text", line["target_text"])
    canonical = line.get("synthesis_text_override", decision.synthesis_text or spoken_target)
    if row.get("canonical_synthesis_text") != canonical:
        return False
    if row.get("action") != decision.action:
        return False
    if Path(row.get("reference_audio", "")).resolve() != reference_path.resolve():
        return False
    checks = {
        "num_step": profile.get("num_step"),
        "guidance_scale": profile.get("guidance_scale"),
        "position_temperature": profile.get("position_temperature"),
        "class_temperature": profile.get("class_temperature"),
        "postprocess_output": profile.get("postprocess_output"),
    }
    return all(row.get(key) == value for key, value in checks.items())


def promote_scene(scene: dict, out: Path, config: dict, wanted: set[str] | None) -> int:
    metadata_path = out / "candidates" / "metadata.json"
    if not metadata_path.exists():
        return 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Some older runs persisted QA rows before their generation metadata was
    # compacted. Recover those rows from the audit ranking when the waveform
    # is still present; the compatibility checks below remain mandatory.
    ranking_path = out / "QA_RANKING.json"
    if ranking_path.exists():
        rankings = json.loads(ranking_path.read_text(encoding="utf-8"))
        known_files = {row.get("file") for row in metadata}
        for line_id, rows in rankings.items():
            if wanted is not None and line_id not in wanted:
                continue
            for row in rows:
                name = row.get("file")
                if name and name not in known_files and (out / "candidates" / name).exists():
                    metadata.append(dict(row))
                    known_files.add(name)
    stem, sr = producer.read(
        producer.resolve(scene["source_stem"], MAP_ROOT),
    )
    refs = producer.prepare_references(scene, stem, sr, out)
    profile = dict(config["anime"])
    promoted = 0
    now = datetime.now(timezone.utc).isoformat()
    rankings = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else {}
    rank_by_file = {
        row.get("file"): row
        for rows in rankings.values()
        for row in rows
        if row.get("file")
    }
    for line in scene["lines"]:
        if wanted is not None and line["id"] not in wanted:
            continue
        decision = producer.decide_line(line)
        if decision.action in {producer.KEEP_ORIGINAL, producer.BLOCKED}:
            continue
        ref_entry = refs.get(line["id"]) or refs.get(line["speaker"])
        if not ref_entry:
            continue
        ref_path = Path(ref_entry[0])
        current_line_hash = line["_codex2_line_contract_hash"]
        current_generation_hash = line["_codex2_generation_hash"]
        current_processing_hash = line["_codex2_processing_hash"]
        canonical = line.get(
            "synthesis_text_override", decision.synthesis_text
            or line.get("delivery_text", line["target_text"]),
        )
        synthesis = canonical
        if profile.get("append_ellipsis_experiment", False):
            synthesis = producer.append_generation_suffix(
                canonical, profile.get("ellipsis_suffix", "..."),
            )
        line_rows = [row for row in metadata if row.get("line_id") == line["id"]]
        compatible = [
            row for row in line_rows
            if (out / "candidates" / row.get("file", "")).exists()
            and _generation_compatible(row, line, decision, ref_path, profile)
        ]
        # Four takes are enough for the release decision. Prefer an existing
        # PASS from a historical QA first, then the lowest score, then the
        # newest round. This prevents a line with twenty old rounds from
        # turning revalidation into another full global QA run.
        compatible.sort(
            key=lambda row: (
                not bool(rank_by_file.get(row.get("file"), {}).get("pass", row.get("pass", False))),
                float(rank_by_file.get(row.get("file"), {}).get("score", row.get("score", 1e9))),
                -int(row.get("round", -1)),
            )
        )
        selected_files = {row.get("file") for row in compatible[:4]}
        for row in line_rows:
            name = row.get("file")
            if name not in selected_files:
                if row.get("contract_hash") == current_line_hash:
                    row["archived_current_contract_hash"] = row["contract_hash"]
                    row["contract_hash"] = "ARCHIVED_REVALIDATION_EXCESS"
                continue
            if not _generation_compatible(row, line, decision, ref_path, profile):
                continue
            old_hash = row.get("contract_hash")
            if old_hash == current_line_hash:
                continue
            row["revalidated_from_contract_hash"] = old_hash
            row["revalidated_at_utc"] = now
            row["contract_hash"] = current_line_hash
            row["generation_contract_hash"] = current_generation_hash
            row["processing_contract_hash"] = current_processing_hash
            row["canonical_synthesis_text"] = canonical
            row["synthesis_text"] = synthesis
            row["ellipsis_experiment"] = synthesis != canonical
            promoted += 1
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--line-ids", nargs="*", default=[])
    parser.add_argument("--maps", type=Path, default=MAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = json.loads(producer.CONFIG_PATH.read_text(encoding="utf-8"))
    wanted_scenes = set(args.scenes)
    wanted_ids = set(args.line_ids)
    map_paths = sorted(args.maps.resolve().glob("*_map.json"))
    if wanted_scenes:
        map_paths = [p for p in map_paths if p.stem.removesuffix("_map") in wanted_scenes]
    total_promoted = 0
    reports = []
    qa_runtime = producer.QARuntime()
    for map_path in map_paths:
        scene, stem, sr, out, stem_path, full_path = load_scene(
            map_path, config, args.output_root.resolve(),
        )
        selected = ({line["id"] for line in scene["lines"]} & wanted_ids) if wanted_ids else None
        total_promoted += promote_scene(scene, out, config, selected)
        if not (out / "candidates" / "metadata.json").exists():
            # Subtitle-only/KEEP_ORIGINAL scenes have no generation metadata.
            # They are already resolved by the map policy and need no QA pass.
            continue
        if selected is not None and not selected:
            continue
        focused, _ = producer.evaluate(
            scene, stem, sr, out, config["qa"], only_ids=selected,
            runtime=qa_runtime,
        )
        merge_rankings(out / "QA_RANKING.json", focused)
        rankings = json.loads((out / "QA_RANKING.json").read_text(encoding="utf-8"))
        report = producer.select_and_mount(scene, stem_path, full_path, out, rankings)
        producer.make_html(scene, stem_path, report, out)
        reports.append(report)
    qa_runtime.asr = None
    qa_runtime.mms = None
    print(json.dumps({
        "promoted": total_promoted,
        "scenes": len(reports),
        "mounted": sum(len(r.get("mounted_voice_ids", [])) for r in reports),
        "review": [
            [r["scene"], line_id]
            for r in reports for line_id in r.get("missing_current_candidate_ids", [])
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
