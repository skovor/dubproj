#!/usr/bin/env python3
"""Build, validate, and deploy the CODEX2 P3R anime/FMV package.

The movie containers are not remuxed.  ``rebuild_anime_usm.py`` replaces only
the English ADX payload in each original CRI USM, preserving the video, the
container layout, and every non-target byte.  Deployment is deliberately
limited to the 20 audited Event_Main movies and their three Reloaded-II
language lookup paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT / "P3R_ANIME_REPAIR_20260802_CODEX2"
MAPS = ROOT / "P3R_ANIME_VISUAL_DUB_20260801" / "maps_delivery_aligned_v3_codex2"
ORIGINALS = ROOT / "scratch" / "all_usm_scan"
REBUILDER = ROOT / "rebuild_anime_usm.py"
TOOLS_PYTHON = ROOT / "tools_cricodecs_env" / "Scripts" / "python.exe"
VGAUDIO = ROOT.parent / "packaging_tools" / "VGAudioCli.exe"
VGMSTREAM = ROOT.parent / "vgmstream" / "vgmstream-cli.exe"
MOD = Path(r"C:\Users\juand\Desktop\Reloaded-II\Mods\p3r.doblaje.aleman")
STREAM = MOD / "UnrealEssentials" / "P3R" / "Content" / "Xrd777" / "CriData" / "Stream"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scene_ids() -> list[str]:
    return sorted(path.stem.removesuffix("_map") for path in MAPS.glob("*_map.json"))


def original_for(scene: str) -> Path:
    matches = [
        path for path in ORIGINALS.glob("*.usm")
        if path.name.lower() == f"ms_event_main_{scene}_movi_vp9.usm".lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one original USM for {scene}, found {matches}")
    return matches[0]


def report_for(scene: str) -> tuple[Path, dict]:
    path = DELIVERY / scene / "FINAL_REPORT.json"
    if not path.is_file():
        raise RuntimeError(f"missing FINAL_REPORT for {scene}: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "contract_pass",
        "other_channels_equal",
        "semantic_source_language_pass",
        "all_generated_voice_hard_gates_pass",
        "all_required_dubs_mounted",
        "continuous_language_pass",
    )
    failed = [key for key in required if report.get(key) is not True]
    if failed:
        raise RuntimeError(f"{scene} report is not packageable; failed={failed}")
    if report.get("missing_current_candidate_ids"):
        raise RuntimeError(f"{scene} has missing current candidates: {report['missing_current_candidate_ids']}")
    wav = Path(report["output"])
    if not wav.is_file():
        raise RuntimeError(f"{scene} German 6ch output missing: {wav}")
    return path, report


def validate_usm(path: Path) -> dict:
    if not path.is_file() or path.read_bytes()[:4] != b"CRID":
        raise RuntimeError(f"invalid CRID signature: {path}")
    probe = subprocess.run(
        [str(VGMSTREAM), "-s", "2", "-m", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    diagnostic = (probe.stdout or "") + (probe.stderr or "")
    required = ("stream count: 2", "channels: 6", "encoding: CRI ADX")
    if probe.returncode or not all(item in diagnostic for item in required):
        raise RuntimeError(f"vgmstream validation failed for {path}:\n{diagnostic[-1600:]}")
    return {
        "stream_count": 2,
        "channels": 6,
        "encoding": "CRI ADX",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def runtime_relative_paths(scene: str, filename: str) -> list[Path]:
    base = Path("UnrealEssentials") / "P3R" / "Content" / "Xrd777" / "CriData" / "Stream"
    if scene.startswith("3"):
        base /= "Astrea"
    return [
        base / "Movie_VP9" / "Anim" / filename,
        base / "de" / "Movie_VP9" / "Anim" / filename,
        base / "en" / "Movie_VP9" / "Anim" / filename,
    ]


def build(stage: Path, scenes: list[str]) -> tuple[list[dict], Path]:
    built_dir = stage / "_built_usm"
    rows: list[dict] = []
    for index, scene in enumerate(scenes, 1):
        report_path, report = report_for(scene)
        original = original_for(scene)
        filename = original.name
        built = built_dir / filename
        command = [
            str(TOOLS_PYTHON), str(REBUILDER),
            "--original-usm", str(original),
            "--german-wav", str(Path(report["output"])),
            "--output-usm", str(built),
            "--vgaudio", str(VGAUDIO),
        ]
        print(f"[{index}/{len(scenes)}] rebuilding {scene}", flush=True)
        result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"USM rebuild failed for {scene}")
        if built.stat().st_size != original.stat().st_size:
            raise RuntimeError(f"container size changed for {scene}")
        validation = validate_usm(built)
        rows.append({
            "scene": scene,
            "original_usm": str(original),
            "german_wav": str(report["output"]),
            "final_report": str(report_path),
            "built_usm": str(built),
            "original_bytes": original.stat().st_size,
            "built_bytes": built.stat().st_size,
            "original_sha256": sha256(original),
            "built_sha256": validation["sha256"],
            "validation": validation,
            "runtime_relative_paths": [str(path) for path in runtime_relative_paths(scene, filename)],
        })
    return rows, built_dir


def backup_existing(paths: list[Path], backup: Path) -> list[dict]:
    rows: list[dict] = []
    for destination in paths:
        if not destination.is_file():
            continue
        relative = destination.relative_to(MOD)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, target)
        if sha256(destination) != sha256(target):
            raise RuntimeError(f"backup hash mismatch: {destination}")
        rows.append({
            "destination": str(destination),
            "backup": str(target),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })
    return rows


def deploy(rows: list[dict], backup: Path) -> list[dict]:
    destination_paths = [
        MOD / relative
        for row in rows
        for relative in runtime_relative_paths(row["scene"], Path(row["built_usm"]).name)
    ]
    backup_rows = backup_existing(destination_paths, backup)
    deployed: list[dict] = []
    for row in rows:
        source = Path(row["built_usm"])
        for relative in runtime_relative_paths(row["scene"], source.name):
            destination = MOD / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".codex2.deploying")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            got = sha256(destination)
            if got != row["built_sha256"]:
                raise RuntimeError(f"deployment hash mismatch: {destination}")
            deployed.append({
                "scene": row["scene"],
                "source": str(source),
                "destination": str(destination),
                "sha256": got,
                "bytes": destination.stat().st_size,
            })
    return backup_rows, deployed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="copy the validated stage into Reloaded-II")
    parser.add_argument("--stage", type=Path, help="reuse a prebuilt stage (not normally needed)")
    args = parser.parse_args()
    if not TOOLS_PYTHON.is_file() or not VGAUDIO.is_file() or not VGMSTREAM.is_file():
        raise RuntimeError("packaging dependencies are missing")
    scenes = scene_ids()
    if len(scenes) != 20:
        raise RuntimeError(f"expected 20 audited scenes, found {len(scenes)}: {scenes}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stage = args.stage or (DELIVERY / f"FMV_RELOADEDII_STAGE_20260804_{stamp}")
    stage.mkdir(parents=True, exist_ok=False)
    rows, built_dir = build(stage, scenes)
    runtime_root = stage / "runtime"
    for row in rows:
        source = Path(row["built_usm"])
        for relative in runtime_relative_paths(row["scene"], source.name):
            target = runtime_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(target) != row["built_sha256"]:
                raise RuntimeError(f"stage runtime hash mismatch: {target}")
    manifest = {
        "format": "CODEX2_FMV_RELOADEDII",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "scene_count": len(rows),
        "audited_lines_final": 172,
        "synthesized_mounted_lines": 170,
        "policy_preserved_lines": 2,
        "runtime_copy_count": len(rows) * 3,
        "runtime_smoke_pending": True,
        "scenes": rows,
        "backup": None,
        "deployed": [],
    }
    if args.deploy:
        backup = MOD / f"_backup_codex2_fmv_{stamp}"
        backup_rows, deployed = deploy(rows, backup)
        manifest["backup"] = {"path": str(backup), "files": backup_rows}
        manifest["deployed"] = deployed
        manifest["runtime_smoke_pending"] = True
    manifest_path = stage / "FMV_RELOADEDII_MANIFEST.json"
    write_json(manifest_path, manifest)
    # A copy at the package root makes the deployment status discoverable
    # without walking the stage directory.
    write_json(DELIVERY / "FMV_RELOADEDII_MANIFEST.json", manifest)
    print(json.dumps({
        "stage": str(stage),
        "manifest": str(manifest_path),
        "deployed": args.deploy,
        "scene_count": len(rows),
        "runtime_copy_count": len(manifest["deployed"]),
        "backup": manifest["backup"]["path"] if manifest["backup"] else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
