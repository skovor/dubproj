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

from dubbing_pipeline.calibration import FeatureRow, LIDFeatureRow, load_draft
from dubbing_pipeline.calibration.promote import promote_profile
from dubbing_pipeline.calibration.validate import ValidationReport, evaluate
from dubbing_pipeline.goldset import GoldsetStore


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path, *, lid: bool = False) -> list[FeatureRow | LIDFeatureRow]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            klass = LIDFeatureRow if lid else FeatureRow
            result.append(klass(str(value["clip_id"]), str(value["split"]), str(value["split_group"]), int(value["label"]), dict(value["features"]), str(value.get("performance_mode", "NEUTRAL")), value.get("metadata")))
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
    parser.add_argument("--lid-artifact", required=True)
    parser.add_argument("--features", required=True, help="sealed FeatureRow JSONL; validation and hidden rows are recomputed")
    parser.add_argument("--final-features", required=True, help="independent final-anchor FeatureRow JSONL")
    parser.add_argument("--lid-features", required=True, help="independent LIDFeatureRow JSONL")
    parser.add_argument("--final-validation-report", required=True)
    parser.add_argument("--final-hidden-report", required=True)
    parser.add_argument("--lid-validation-report", required=True)
    parser.add_argument("--lid-hidden-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--models-lock", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--goldset-db", required=True, help="authoritative SQLite GoldsetStore")
    parser.add_argument("--hidden-receipt-id", required=True)
    parser.add_argument("--hidden-run-id", required=True)
    args = parser.parse_args(argv)
    target_path, final_path, lid_path = Path(args.target_artifact), Path(args.final_artifact), Path(args.lid_artifact)
    target = load_draft(target_path); final = load_draft(final_path); lid = load_draft(lid_path)
    feature_rows = _rows(Path(args.features)); anchor_rows = _rows(Path(args.final_features)); lid_rows = _rows(Path(args.lid_features), lid=True)
    validation_rows = [row for row in feature_rows if row.split == "validation"]; hidden_rows = [row for row in feature_rows if row.split == "hidden_test"]
    anchor_validation_rows = [row for row in anchor_rows if row.split == "validation"]; anchor_hidden_rows = [row for row in anchor_rows if row.split == "hidden_test"]
    lid_validation_rows = [row for row in lid_rows if row.split == "validation"]; lid_hidden_rows = [row for row in lid_rows if row.split == "hidden_test"]
    validation = _report(Path(args.validation_report)); hidden = _report(Path(args.hidden_report)); anchor_validation = _report(Path(args.final_validation_report)); anchor_hidden = _report(Path(args.final_hidden_report)); lid_validation = _report(Path(args.lid_validation_report)); lid_hidden = _report(Path(args.lid_hidden_report))
    models_path, runtime_path = Path(args.models_lock), Path(args.runtime_lock)
    models = json.loads(models_path.read_text(encoding="utf-8")); model = (models.get("models") or [None])[0]
    if not isinstance(model, dict):
        raise RuntimeError("models lock has no model identity")
    backend_id = str(model.get("backend_id") or model.get("backend") or "")
    identity = {"backend_id": backend_id, "model_id": str(model.get("model_id", "")), "model_revision": str(model.get("revision", "")), "feature_schema_version": str(target.get("feature_schema_version", "")), "target_language": str(model.get("language", "de")), "source_language": "en", "performance_modes": sorted({row.performance_mode for row in feature_rows})}
    thresholds = {"target_pass_probability": .8, "target_failure_probability": .2, "final_anchor_pass_probability": .8, "source_lid_probability": .8}
    provenance = {"code_commit": _git_sha(), "runtime_lock_sha256": _sha(runtime_path), "models_lock_sha256": _sha(models_path)}
    profile_id = f"{identity['model_id']}-{uuid.uuid4().hex[:8]}"
    store = GoldsetStore(args.goldset_db)
    try:
        hidden_receipt = store.get_hidden_evaluation_receipt(args.hidden_receipt_id, args.hidden_run_id)
        finalization = store.finalize_hidden_evaluation(
            receipt_id=args.hidden_receipt_id, run_id=args.hidden_run_id, profile_id=profile_id,
            code_commit=provenance["code_commit"],
            role_hidden_rows={"target": hidden_rows, "final_anchor": anchor_hidden_rows, "lid": lid_hidden_rows},
            role_hidden_reports={"target": hidden.to_dict(), "final_anchor": anchor_hidden.to_dict(), "lid": lid_hidden.to_dict()},
            hidden_jsonl_hashes={"target": _sha(Path(args.features)), "final_anchor": _sha(Path(args.final_features)), "lid": _sha(Path(args.lid_features))},
            hidden_report_hashes={"target": _sha(Path(args.hidden_report)), "final_anchor": _sha(Path(args.final_hidden_report)), "lid": _sha(Path(args.lid_hidden_report))},
        )
        profile = promote_profile(profile_id=profile_id, target_artifact={**target, "artifact_path": str(target_path), "artifact_sha256": _sha(target_path)}, final_anchor_artifact={**final, "artifact_path": str(final_path), "artifact_sha256": _sha(final_path)}, lid_artifact={**lid, "artifact_path": str(lid_path), "artifact_sha256": _sha(lid_path)}, validation=validation, hidden=hidden, validation_rows=validation_rows, hidden_rows=hidden_rows, role_validation_rows={"target": validation_rows, "final_anchor": anchor_validation_rows, "lid": lid_validation_rows}, role_hidden_rows={"target": hidden_rows, "final_anchor": anchor_hidden_rows, "lid": lid_hidden_rows}, role_validation_reports={"target": validation, "final_anchor": anchor_validation, "lid": lid_validation}, role_hidden_reports={"target": hidden, "final_anchor": anchor_hidden, "lid": lid_hidden}, dataset_files={"manifest_sha256": args.manifest, "labels_sha256": args.labels, "split_manifest_sha256": args.splits}, identity=identity, thresholds=thresholds, provenance=provenance, output=args.output, hidden_evaluation_receipt=hidden_receipt, require_hidden_evaluation_receipt=True, hidden_evaluation_finalization=finalization, goldset_store=store, require_hidden_evaluation_finalization=True)
    finally:
        store.close()
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
