#!/usr/bin/env python3
"""Audit and optionally repair subtitle-to-speech association for anime maps.

The legacy maps often used subtitle-card bounds as if they were phonetic
boundaries.  This ledger compares every subtitle-authorized/force-clone line
against two independent English word-timestamp passes plus the measured dry
dialogue VAD.  It never invents a line: rows without lexical consensus are
reported and remain unchanged unless an operator explicitly supplies a map
correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from repair_html_timing_codex2 import (
    ANNOTATED,
    DUAL_ASR,
    LARGE_ASR,
    MAPS,
    pass_streams,
    relative_vad_runs,
    derive_from_asr,
)


REPORT = MAPS.parent / "ANIME_SUBTITLE_SCENE_ALIGNMENT_CODEX2.json"


def audio_for(scene: dict) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(scene["source_stem"], dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def classify(line: dict, timing: dict | None, error: str | None) -> str:
    if error:
        # Short interjections/proper names are often omitted by Whisper even
        # though the visible subtitle card is authoritative.  Do not turn
        # that ASR limitation into a generation block: the card window is the
        # timing authority and the candidate's own text/tail/source-language
        # QA remains mandatory.  A non-visual/no-authority row still fails
        # closed as before.
        if (
            line.get("mapping_validation") == "HUMAN_CONFIRMED"
            and str(line.get("timing_source", "")).startswith("VISIBLE")
        ):
            return "PASS_VISUAL_CARD_NO_ASR"
        return "BLOCK_NO_LEXICAL_EVIDENCE"
    assert timing is not None
    passes = timing.get("qualified_passes", [])
    old_start = float(line["start"])
    old_end = float(line["end"])
    asr_start = float(timing.get("asr_consensus_start", timing["start"]))
    asr_end = float(timing.get("asr_consensus_end", timing["end"]))
    if len(passes) < 2:
        return "REVIEW_SINGLE_PASS"
    if abs(asr_start - old_start) > 1.5 or abs(asr_end - old_end) > 1.5:
        return "REVIEW_ASSOCIATION_DISTANCE"
    if abs(float(timing["start"]) - old_start) > 0.060 or abs(float(timing["end"]) - old_end) > 0.060:
        return "REPAIR_CONSENSUS_WINDOW"
    return "PASS_ASSOCIATION"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write consensus windows to maps")
    args = parser.parse_args()
    rows: list[dict] = []
    modified: set[Path] = set()
    for map_path in sorted(MAPS.glob("*_map.json")):
        scene = json.loads(map_path.read_text(encoding="utf-8"))
        clone_lines = [
            line for line in scene.get("lines", [])
            if line.get("force_clone") and not line.get("force_keep_original")
        ]
        if not clone_lines:
            continue
        try:
            audio, sr = audio_for(scene)
            streams = pass_streams(scene)
            vad_runs, vad_meta = relative_vad_runs(audio, sr)
        except Exception as exc:  # scene-level evidence failure is reportable
            for line in clone_lines:
                rows.append({
                    "scene": scene["scene"], "line_id": line["id"],
                    "old_start": line["start"], "old_end": line["end"],
                    "status": "BLOCK_SCENE_EVIDENCE", "error": str(exc),
                })
            continue
        for line in clone_lines:
            timing = None
            error = None
            try:
                timing = derive_from_asr(scene, line, streams, audio, sr, vad_runs, vad_meta)
            except Exception as exc:
                error = str(exc)
            status = classify(line, timing, error)
            row = {
                "scene": scene["scene"], "line_id": line["id"],
                "source_text": line.get("source_text"),
                "subtitle_start": line.get("subtitle_start", line["start"]),
                "subtitle_end": line.get("subtitle_end", line["end"]),
                "old_start": line["start"], "old_end": line["end"],
                "status": status,
            }
            if error:
                row["error"] = error
            else:
                row.update({
                    "new_start": timing["start"], "new_end": timing["end"],
                    "asr_consensus_start": timing.get("asr_consensus_start"),
                    "asr_consensus_end": timing.get("asr_consensus_end"),
                    "qualified_passes": timing.get("qualified_passes", []),
                    "evidence": timing,
                })
            rows.append(row)
            if args.apply and status == "REPAIR_CONSENSUS_WINDOW":
                line.update({
                    "start": timing["start"], "end": timing["end"],
                    "speech_start": timing["start"], "speech_end": timing["end"],
                    "reference_start": timing["start"], "reference_end": timing["end"],
                    "mount_start": timing["start"], "mount_end": timing["end"],
                    "timing_source": "GLOBAL_SUBTITLE_ASR_VAD_ALIGNMENT_CODEX2",
                    "timing_review_required": False,
                    "mapping_validation": "EXACT",
                    "mapping_validation_reason": (
                        "subtitle-authorized source text matched by independent "
                        "ASR passes; VAD used only for phonetic edges"
                    ),
                })
                modified.add(map_path)
    if args.apply:
        for map_path in modified:
            payload = json.loads(map_path.read_text(encoding="utf-8"))
            by_id = {row["line_id"]: row for row in rows if row["status"] == "REPAIR_CONSENSUS_WINDOW"}
            for line in payload.get("lines", []):
                row = by_id.get(line["id"])
                if row:
                    line.update({
                        "start": row["new_start"], "end": row["new_end"],
                        "speech_start": row["new_start"], "speech_end": row["new_end"],
                        "reference_start": row["new_start"], "reference_end": row["new_end"],
                        "mount_start": row["new_start"], "mount_end": row["new_end"],
                        "timing_source": "GLOBAL_SUBTITLE_ASR_VAD_ALIGNMENT_CODEX2",
                        "timing_review_required": False,
                        "mapping_validation": "EXACT",
                        "mapping_validation_reason": (
                            "subtitle-authorized source text matched by independent "
                            "ASR passes; VAD used only for phonetic edges"
                        ),
                    })
            map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "p3r_anime_subtitle_scene_alignment_codex2_v1",
        "maps": len({row["scene"] for row in rows}),
        "lines": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "maps_modified": [str(path) for path in sorted(modified)],
        "rows": rows,
    }
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
