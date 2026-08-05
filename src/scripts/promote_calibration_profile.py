"""Recompute sealed calibration evidence and promote a profile safely.

The command intentionally has no identity defaults.  Locks and the current
Git checkout are the only accepted provenance sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_pipeline.calibration import FeatureRow, load_draft
from dubbing_pipeline.calibration.promote import promote_profile
from dubbing_pipeline.calibration.validate import ValidationReport, evaluate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[FeatureRow]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            result.append(FeatureRow(str(value["clip_id"]), str(value["split"]), str(value["split_group"]), int(value["label"]), dict(value["features"]), str(value.get("performance_mode", "NEUTRAL")), value.get("metadata")))
    return result


def _report(path: Path) -> ValidationReport:
    value = json.loads(path.read_text(encoding="utf-8"))
    return ValidationReport(str(value["split"]), int(value["count"]), float(value["brier_score"]), float(value["expected_calibration_error"]), int(value["false_pass_count"]), int(value["false_fail_count"]), tuple(value.get("predictions", [])), str(value.get("run_id", "")), bool(value.get("recomputed", False)))


def _git_sha() -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40:
        raise RuntimeError("git rev-parse did not return a full commit SHA")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_artifact")
    parser.add_argument("validation_report")
    parser.add_argument("hidden_report")
    parser.add_argument("--final-artifact", required=True)
    parser.add_argument("--features", required=True, help="sealed FeatureRow JSONL; validation and hidden rows are recomputed")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--models-lock", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target_path, final_path = Path(args.target_artifact), Path(args.final_artifact)
    target = load_draft(target_path); final = load_draft(final_path)
    feature_rows = _rows(Path(args.features)); validation_rows = [row for row in feature_rows if row.split == "validation"]; hidden_rows = [row for row in feature_rows if row.split == "hidden_test"]
    validation = _report(Path(args.validation_report)); hidden = _report(Path(args.hidden_report))
    models_path, runtime_path = Path(args.models_lock), Path(args.runtime_lock)
    models = json.loads(models_path.read_text(encoding="utf-8")); model = (models.get("models") or [None])[0]
    if not isinstance(model, dict):
        raise RuntimeError("models lock has no model identity")
    backend_id = str(model.get("backend_id") or model.get("backend") or "")
    identity = {"backend_id": backend_id, "model_id": str(model.get("model_id", "")), "model_revision": str(model.get("revision", "")), "feature_schema_version": str(target.get("feature_schema_version", "")), "target_language": str(model.get("language", "de")), "source_language": "en", "performance_modes": sorted({row.performance_mode for row in feature_rows})}
    thresholds = {"target_pass_probability": .8, "target_failure_probability": .2, "final_anchor_pass_probability": .8, "source_lid_probability": .8}
    provenance = {"code_commit": _git_sha(), "runtime_lock_sha256": _sha(runtime_path), "models_lock_sha256": _sha(models_path)}
    profile = promote_profile(profile_id=f"{identity['model_id']}-{uuid.uuid4().hex[:8]}", target_artifact={**target, "artifact_path": str(target_path), "artifact_sha256": _sha(target_path)}, final_anchor_artifact={**final, "artifact_path": str(final_path), "artifact_sha256": _sha(final_path)}, validation=validation, hidden=hidden, validation_rows=validation_rows, hidden_rows=hidden_rows, dataset_files={"manifest_sha256": args.manifest, "labels_sha256": args.labels, "split_manifest_sha256": args.splits}, identity=identity, thresholds=thresholds, provenance=provenance, output=args.output)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
