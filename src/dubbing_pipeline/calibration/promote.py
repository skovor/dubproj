"""Fail-closed profile promotion with recomputed dataset hashes."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from .train import CalibrationArtifact
from .validate import ValidationReport, evaluate
from .receipt import promotion_profile_payload, promotion_profile_payload_sha256

class PromotionError(ValueError): pass

def _prediction_digest(report: ValidationReport) -> str:
    return hashlib.sha256(json.dumps(list(report.predictions), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _content_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact_spec(value: Mapping[str, Any], role: str, identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path = Path(str(value.get("artifact_path", "")))
    artifact_sha = str(value.get("artifact_sha256", "")).casefold()
    if not artifact_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
        raise PromotionError(f"{role} artifact must be an existing hashed file")
    actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha != artifact_sha:
        raise PromotionError(f"{role} artifact hash does not match its bytes")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PromotionError(f"{role} artifact is not valid JSON") from exc
    expected_schema = "final-anchor-v1" if role == "final_anchor" else ("lid-fusion-v2" if role == "lid" else str(identity.get("feature_schema_version", "char-alignment-v2")))
    if payload.get("schema") != "platt-calibrator-v1" or payload.get("feature_schema_version") != expected_schema or payload.get("status", "DRAFT") not in {"DRAFT", "VALIDATED"}:
        raise PromotionError(f"{role} artifact schema/status mismatch")
    features = list(payload.get("features") or [])
    coefficients = list(payload.get("coefficients") or [])
    normalization = payload.get("normalization")
    if not features or len(features) != len(coefficients) or not isinstance(normalization, list) or len(normalization) != len(features):
        raise PromotionError(f"{role} artifact is incomplete")
    spec = {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": expected_schema, "normalization_version": str(payload.get("normalization_version", "alignment-text-normalization-v2")), "feature_names": features, "artifact_path": str(artifact_path), "artifact_sha256": artifact_sha}
    return spec, payload


def _require_class_composition(rows: list[Any], split: str, *, minimum: int = 2) -> None:
    labels = [int(row.label) for row in rows if getattr(row, "split", None) == split]
    if len(labels) < minimum * 2 or labels.count(0) < minimum or labels.count(1) < minimum:
        raise PromotionError(f"{split} requires at least {minimum} positive and {minimum} negative sealed rows")


def _validate_report(report: ValidationReport, *, role: str, split: str) -> None:
    """Reject claims that cannot be a metric report for this role/split."""
    if report.split != split:
        raise PromotionError(f"{role} report is mislabelled as {report.split!r}")
    if report.count < 0 or report.false_pass_count < 0 or report.false_fail_count < 0:
        raise PromotionError(f"{role} report contains negative counts")
    if report.count != len(report.predictions):
        raise PromotionError(f"{role} report count does not match predictions")
    for value in (report.brier_score, report.expected_calibration_error):
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise PromotionError(f"{role} report metric is outside [0,1]")
    for prediction in report.predictions:
        try:
            probability = float(prediction["probability"])
            label = int(prediction["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionError(f"{role} report has malformed prediction") from exc
        if label not in (0, 1) or not 0.0 <= probability <= 1.0:
            raise PromotionError(f"{role} report has an invalid prediction")


def promote_profile(*, profile_id: str, target_artifact: CalibrationArtifact | Mapping[str, Any], final_anchor_artifact: CalibrationArtifact | Mapping[str, Any] | None = None, lid_artifact: CalibrationArtifact | Mapping[str, Any] | None = None, validation: ValidationReport, hidden: ValidationReport, dataset_files: Mapping[str, str | Path], identity: Mapping[str, Any], thresholds: Mapping[str, float], provenance: Mapping[str, Any], output: str | Path, validation_rows: Iterable[Any] | None = None, hidden_rows: Iterable[Any] | None = None, role_validation_rows: Mapping[str, Iterable[Any]] | None = None, role_hidden_rows: Mapping[str, Iterable[Any]] | None = None, role_validation_reports: Mapping[str, ValidationReport] | None = None, role_hidden_reports: Mapping[str, ValidationReport] | None = None, minimum_class_count: int = 2) -> dict[str, Any]:
    """Create VALIDATED only from sealed rows and recomputed predictions.

    Precomputed JSON reports are treated as claims, not evidence.  Callers
    must provide the immutable validation/hidden rows so this function can
    recompute every probability and compare the supplied reports to that
    recomputation before promotion.
    """
    _validate_report(validation, role="target", split="validation")
    _validate_report(hidden, role="target", split="hidden_test")
    if validation_rows is None or hidden_rows is None:
        raise PromotionError("sealed validation_rows and hidden_rows are required; reports alone are not evidence")
    # The legacy arguments remain aliases for target evidence only.  Anchor and
    # LID rows must be supplied independently; reusing target rows would make
    # an apparently calibrated profile impossible to audit.
    role_validation_rows = dict(role_validation_rows or {})
    role_hidden_rows = dict(role_hidden_rows or {})
    role_validation_rows.setdefault("target", validation_rows)
    role_hidden_rows.setdefault("target", hidden_rows)
    if not {"target", "final_anchor", "lid"}.issubset(role_validation_rows) or not {"target", "final_anchor", "lid"}.issubset(role_hidden_rows):
        raise PromotionError("independent validation and hidden rows are required for target, final_anchor, and lid")
    role_validation_rows = {role: list(rows) for role, rows in role_validation_rows.items()}
    role_hidden_rows = {role: list(rows) for role, rows in role_hidden_rows.items()}
    role_validation_reports = dict(role_validation_reports or {})
    role_hidden_reports = dict(role_hidden_reports or {})
    role_validation_reports.setdefault("target", validation)
    role_hidden_reports.setdefault("target", hidden)
    artifacts_by_role = {"target": target_artifact, "final_anchor": final_anchor_artifact, "lid": lid_artifact}
    pass_keys = {"target": "target_pass_probability", "final_anchor": "final_anchor_pass_probability", "lid": "source_lid_probability"}
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for role, artifact_value in artifacts_by_role.items():
        if artifact_value is None:
            raise PromotionError(f"{role} artifact is required")
        if role not in role_validation_reports or role not in role_hidden_reports:
            raise PromotionError(f"independent {role} validation and hidden reports are required")
        supplied_validation = role_validation_reports[role]
        supplied_hidden = role_hidden_reports[role]
        _validate_report(supplied_validation, role=role, split="validation")
        _validate_report(supplied_hidden, role=role, split="hidden_test")
        validation_role_rows = role_validation_rows[role]
        hidden_role_rows = role_hidden_rows[role]
        _require_class_composition(validation_role_rows, "validation", minimum=minimum_class_count)
        _require_class_composition(hidden_role_rows, "hidden_test", minimum=minimum_class_count)
        pass_probability = float(thresholds.get(pass_keys[role], .8))
        fail_probability = float(thresholds.get("target_failure_probability", .2))
        recomputed_validation = evaluate(artifact_value, validation_role_rows, split="validation", pass_probability=pass_probability, fail_probability=fail_probability, run_id=supplied_validation.run_id)
        recomputed_hidden = evaluate(artifact_value, hidden_role_rows, split="hidden_test", pass_probability=pass_probability, fail_probability=fail_probability, run_id=supplied_hidden.run_id)
        if recomputed_validation.to_dict() != supplied_validation.to_dict() or recomputed_hidden.to_dict() != supplied_hidden.to_dict():
            raise PromotionError(f"supplied {role} report differs from recomputed predictions")
        if recomputed_hidden.false_pass_count > 0:
            raise PromotionError(f"{role} hidden false PASS blocks promotion")
        if not supplied_hidden.run_id.strip():
            raise PromotionError(f"{role} hidden test must have a one-shot run_id")
        reports[role] = {"validation": supplied_validation.to_dict(), "hidden_test": supplied_hidden.to_dict()}
    if not profile_id.strip() or not all(str(identity.get(key, "")).strip() for key in ("backend_id", "model_id", "model_revision", "feature_schema_version", "target_language", "source_language")): raise PromotionError("incomplete identity")
    expected = {"target_pass_probability", "target_failure_probability", "final_anchor_pass_probability", "source_lid_probability"}
    if set(thresholds) < expected or not (0 <= float(thresholds["target_failure_probability"]) < float(thresholds["target_pass_probability"]) <= 1): raise PromotionError("invalid thresholds")
    hashes = {}
    for name, path in dataset_files.items():
        file_path = Path(path)
        if not file_path.is_file(): raise PromotionError(f"missing dataset artifact: {name}")
        hashes[name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if not all(str(provenance.get(key, "")).strip() for key in ("code_commit", "runtime_lock_sha256", "models_lock_sha256")): raise PromotionError("incomplete provenance")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(provenance.get("code_commit", ""))): raise PromotionError("code_commit must be a real Git SHA")
    if any(not re.fullmatch(r"[0-9a-fA-F]{64}", str(provenance.get(key, ""))) for key in ("runtime_lock_sha256", "models_lock_sha256")): raise PromotionError("lock provenance hashes are invalid")
    artifact = target_artifact.to_dict() if isinstance(target_artifact, CalibrationArtifact) else dict(target_artifact)
    anchor = final_anchor_artifact.to_dict() if isinstance(final_anchor_artifact, CalibrationArtifact) else dict(final_anchor_artifact)
    lid = lid_artifact.to_dict() if isinstance(lid_artifact, CalibrationArtifact) else dict(lid_artifact)
    target_spec, target_payload = _artifact_spec(artifact, "target", identity); anchor_spec, anchor_payload = _artifact_spec(anchor, "final_anchor", identity); lid_spec, lid_payload = _artifact_spec(lid, "lid", identity)
    report_digests = {role: {split: _prediction_digest(ValidationReport(**report)) for split, report in role_reports.items()} for role, role_reports in reports.items()}
    # The receipt is deliberately a separate, content-addressed object.  It
    # contains every input identity which can affect a promotion decision but
    # excludes timestamps and the receipt hash itself, so it can be rehashed
    # independently by runtime QA.
    receipt_payload = {
        "schema": "dubproj-promotion-receipt-v1", "profile_id": profile_id,
        "code_commit": str(provenance["code_commit"]).lower(),
        "artifact_sha256": {"target": target_spec["artifact_sha256"], "final_anchor": anchor_spec["artifact_sha256"], "lid": lid_spec["artifact_sha256"]},
        "dataset_sha256": dict(hashes),
        "lock_sha256": {"runtime": str(provenance["runtime_lock_sha256"]).lower(), "models": str(provenance["models_lock_sha256"]).lower()},
        "report_prediction_sha256": report_digests,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = destination.with_suffix(destination.suffix + ".promotion_receipt.json")
    receipt_bytes = (json.dumps(receipt_payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    profile = {
        "schema": "generic-dubbing-alignment-calibration-profile-v2", "status": "VALIDATED", "authority": True, "profile_id": profile_id,
        "identity": dict(identity), "thresholds": {key: float(value) for key, value in thresholds.items()},
        "calibrators": {"target": target_spec, "final_anchor": anchor_spec, "lid": lid_spec},
        "dataset": {**hashes, "calibration_count": int(artifact.get("sample_count", 0)), "validation_count": validation.count, "hidden_test_count": hidden.count},
        "metrics": {"hidden_false_pass_count": hidden.false_pass_count, "hidden_false_fail_count": hidden.false_fail_count, "brier_score": hidden.brier_score, "expected_calibration_error": hidden.expected_calibration_error, "validation": validation.to_dict(), "validation_predictions_sha256": _prediction_digest(validation), "hidden_predictions_sha256": _prediction_digest(hidden), "reports": reports, "recomputed": True},
        "provenance": {**dict(provenance), "created_at": datetime.now(timezone.utc).isoformat(), "hidden_test_run_id": hidden.run_id, "promotion_receipt_path": str(receipt_path), "promotion_receipt_sha256": receipt_sha},
    }
    profile_payload = promotion_profile_payload(profile)
    profile_payload_sha = promotion_profile_payload_sha256(profile)
    receipt_payload["profile_payload"] = profile_payload
    receipt_payload["promoted_profile_payload_sha256"] = profile_payload_sha
    receipt_bytes = (json.dumps(receipt_payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    profile["provenance"]["promotion_receipt_sha256"] = receipt_sha
    profile["provenance"]["promoted_profile_payload_sha256"] = profile_payload_sha
    destination.write_text(json.dumps(profile, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return profile

__all__ = ["PromotionError", "promote_profile"]
