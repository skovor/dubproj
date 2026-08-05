"""Fail-closed profile promotion with recomputed dataset hashes."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from .train import CalibrationArtifact
from .validate import ValidationReport, evaluate, predict_artifact

class PromotionError(ValueError): pass

def _prediction_digest(report: ValidationReport) -> str:
    return hashlib.sha256(json.dumps(list(report.predictions), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    expected_schema = "final-anchor-v1" if role == "final_anchor" else ("lid-fusion-v1" if role == "lid" else str(identity.get("feature_schema_version", "char-alignment-v2")))
    if payload.get("schema") != "platt-calibrator-v1" or payload.get("feature_schema_version") != expected_schema or payload.get("status", "DRAFT") not in {"DRAFT", "VALIDATED"}:
        raise PromotionError(f"{role} artifact schema/status mismatch")
    features = list(payload.get("features") or [])
    coefficients = list(payload.get("coefficients") or [])
    normalization = payload.get("normalization")
    if not features or len(features) != len(coefficients) or not isinstance(normalization, list) or len(normalization) != len(features):
        raise PromotionError(f"{role} artifact is incomplete")
    spec = {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": expected_schema, "normalization_version": str(payload.get("normalization_version", "alignment-text-normalization-v1")), "feature_names": features, "artifact_path": str(artifact_path), "artifact_sha256": artifact_sha}
    return spec, payload


def _require_class_composition(rows: list[Any], split: str, *, minimum: int = 2) -> None:
    labels = [int(row.label) for row in rows if getattr(row, "split", None) == split]
    if len(labels) < minimum * 2 or labels.count(0) < minimum or labels.count(1) < minimum:
        raise PromotionError(f"{split} requires at least {minimum} positive and {minimum} negative sealed rows")


def promote_profile(*, profile_id: str, target_artifact: CalibrationArtifact | Mapping[str, Any], final_anchor_artifact: CalibrationArtifact | Mapping[str, Any] | None = None, lid_artifact: CalibrationArtifact | Mapping[str, Any] | None = None, validation: ValidationReport, hidden: ValidationReport, dataset_files: Mapping[str, str | Path], identity: Mapping[str, Any], thresholds: Mapping[str, float], provenance: Mapping[str, Any], output: str | Path, validation_rows: Iterable[Any] | None = None, hidden_rows: Iterable[Any] | None = None, minimum_class_count: int = 2) -> dict[str, Any]:
    """Create VALIDATED only from sealed rows and recomputed predictions.

    Precomputed JSON reports are treated as claims, not evidence.  Callers
    must provide the immutable validation/hidden rows so this function can
    recompute every probability and compare the supplied reports to that
    recomputation before promotion.
    """
    if validation.split != "validation" or hidden.split != "hidden_test": raise PromotionError("validation/hidden reports are mislabelled")
    if validation_rows is None or hidden_rows is None:
        raise PromotionError("sealed validation_rows and hidden_rows are required; reports alone are not evidence")
    validation_rows = list(validation_rows); hidden_rows = list(hidden_rows)
    _require_class_composition(validation_rows, "validation", minimum=minimum_class_count)
    _require_class_composition(hidden_rows, "hidden_test", minimum=minimum_class_count)
    target_validation = evaluate(target_artifact, validation_rows, split="validation", pass_probability=float(thresholds.get("target_pass_probability", .8)), fail_probability=float(thresholds.get("target_failure_probability", .2)), run_id=validation.run_id)
    target_hidden = evaluate(target_artifact, hidden_rows, split="hidden_test", pass_probability=float(thresholds.get("target_pass_probability", .8)), fail_probability=float(thresholds.get("target_failure_probability", .2)), run_id=hidden.run_id)
    if target_validation.to_dict() != validation.to_dict() or target_hidden.to_dict() != hidden.to_dict():
        raise PromotionError("supplied validation/hidden report differs from recomputed predictions")
    if target_hidden.false_pass_count > 0: raise PromotionError("hidden false PASS blocks promotion")
    if not hidden.run_id.strip(): raise PromotionError("hidden test must have a one-shot run_id")
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
    if final_anchor_artifact is None or lid_artifact is None: raise PromotionError("target, final-anchor, and LID artifacts are all required")
    artifact = target_artifact.to_dict() if isinstance(target_artifact, CalibrationArtifact) else dict(target_artifact)
    anchor = final_anchor_artifact.to_dict() if isinstance(final_anchor_artifact, CalibrationArtifact) else dict(final_anchor_artifact)
    lid = lid_artifact.to_dict() if isinstance(lid_artifact, CalibrationArtifact) else dict(lid_artifact)
    target_spec, target_payload = _artifact_spec(artifact, "target", identity); anchor_spec, anchor_payload = _artifact_spec(anchor, "final_anchor", identity); lid_spec, lid_payload = _artifact_spec(lid, "lid", identity)
    profile = {
        "schema": "generic-dubbing-alignment-calibration-profile-v2", "status": "VALIDATED", "authority": True, "profile_id": profile_id,
        "identity": dict(identity), "thresholds": {key: float(value) for key, value in thresholds.items()},
        "calibrators": {"target": target_spec, "final_anchor": anchor_spec, "lid": lid_spec},
        "dataset": {**hashes, "calibration_count": int(artifact.get("sample_count", 0)), "validation_count": validation.count, "hidden_test_count": hidden.count},
        "metrics": {"hidden_false_pass_count": hidden.false_pass_count, "hidden_false_fail_count": hidden.false_fail_count, "brier_score": hidden.brier_score, "expected_calibration_error": hidden.expected_calibration_error, "validation": validation.to_dict(), "validation_predictions_sha256": _prediction_digest(validation), "hidden_predictions_sha256": _prediction_digest(hidden), "recomputed": True},
        "provenance": {**dict(provenance), "created_at": datetime.now(timezone.utc).isoformat(), "hidden_test_run_id": hidden.run_id},
    }
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(profile, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return profile

__all__ = ["PromotionError", "promote_profile"]
