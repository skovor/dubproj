#!/usr/bin/env python3
"""Create a hash-only baseline without copying or modifying production data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def build(repo: Path, output: Path, p3r_root: Path | None) -> dict[str, Any]:
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines()
    files = [{"path": item, "sha256": sha256(repo / item), "bytes": (repo / item).stat().st_size} for item in tracked if (repo / item).is_file()]
    p3r: dict[str, Any] = {"status": "NOT_PROVIDED"}
    if p3r_root is not None:
        p3r_root = p3r_root.resolve(); summary = p3r_root / "GLOBAL_ROUND_SUMMARY.json"
        scene_rows = []
        for report in sorted(p3r_root.glob("*/FINAL_REPORT.json")):
            try:
                value = json.loads(report.read_text(encoding="utf-8"))
                scene_rows.append({"scene": report.parent.name, "path": str(report), "sha256": sha256(report), "line_count": len(value.get("lines", [])), "review_count": len(value.get("review", [])), "release_blockers": value.get("release_blockers", [])})
            except Exception as exc:
                scene_rows.append({"scene": report.parent.name, "path": str(report), "sha256": sha256(report), "error": str(exc)})
        summary_value = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
        timing_totals: dict[str, float] = {}; timing_rows = 0
        for timing_path in sorted(p3r_root.glob("*/STAGE_TIMINGS.json")):
            try: timing_rows_value = json.loads(timing_path.read_text(encoding="utf-8"))
            except Exception: continue
            for timing in timing_rows_value if isinstance(timing_rows_value, list) else []:
                if isinstance(timing, dict) and isinstance(timing.get("seconds"), (int, float)):
                    timing_rows += 1; stage = str(timing.get("stage", "unknown")); timing_totals[stage] = timing_totals.get(stage, 0.0) + float(timing["seconds"])
        timing_total = sum(timing_totals.values())
        p3r = {"status": "PRESENT", "root": str(p3r_root), "summary": {"path": str(summary), "sha256": sha256(summary) if summary.is_file() else None, "required": summary_value.get("required"), "mounted": summary_value.get("mounted"), "release_ready": summary_value.get("release_ready"), "review": summary_value.get("review", [])}, "legacy_stage_timing": {"rows": timing_rows, "seconds_by_stage": timing_totals, "total_seconds": timing_total, "minutes": timing_total / 60.0, "is_run_scoped": False}, "scene_count": len(scene_rows), "line_count": sum(int(row.get("line_count", 0)) for row in scene_rows), "scenes": scene_rows}
    value = {"schema": "baseline-manifest-v2", "created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(repo), "classification_policy": ["EXPECTED_MATCH", "KNOWN_LEGACY_BUG", "UNVERIFIED", "INTENTIONAL_CHANGE_REQUIRED"], "repository": {"root": str(repo.resolve()), "tracked_file_count": len(files), "files": files}, "p3r_reference": p3r, "production_inputs_copied": False, "golden_evidence": [], "known_audit_findings": ["rollback_absent_files", "fmv_bed_erasure", "fake_hard_gates", "unsafe_atomic_json", "serial_fmv_scheduler", "incomplete_generation_hash"]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", required=True); parser.add_argument("--output", required=True); parser.add_argument("--p3r-root")
    args = parser.parse_args(); value = build(Path(args.repo), Path(args.output), Path(args.p3r_root) if args.p3r_root else None); print(json.dumps({"output": str(Path(args.output).resolve()), "git_revision": value["git_revision"], "tracked_file_count": value["repository"]["tracked_file_count"], "p3r": value["p3r_reference"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
