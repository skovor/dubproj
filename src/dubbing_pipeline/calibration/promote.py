"""Fail-closed profile promotion with recomputed dataset hashes."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from .train import CalibrationArtifact
from .validate import ValidationReport

class PromotionError(ValueError): pass

def promote_profile(*, profile_id: str, target_artifact: CalibrationArtifact | Mapping[str, Any], validation: ValidationReport, hidden: ValidationReport, dataset_files: Mapping[str, str | Path], identity: Mapping[str, Any], thresholds: Mapping[str, float], provenance: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Create VALIDATED only when hidden evaluation is sealed and has no critical false PASS."""
    if validation.split != "validation" or hidden.split != "hidden_test": raise PromotionError("validation/hidden reports are mislabelled")
    if hidden.false_pass_count > 0: raise PromotionError("hidden false PASS blocks promotion")
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
    artifact = target_artifact.to_dict() if isinstance(target_artifact, CalibrationArtifact) else dict(target_artifact)
    artifact_path = str(artifact.get("artifact_path", "")); artifact_sha = str(artifact.get("artifact_sha256", ""))
    if not artifact_path or len(artifact_sha) != 64: raise PromotionError("target artifact must be hashed")
    profile = {
        "schema": "generic-dubbing-alignment-calibration-profile-v1", "status": "VALIDATED", "authority": True, "profile_id": profile_id,
        "identity": dict(identity), "thresholds": {key: float(value) for key, value in thresholds.items()},
        "calibrator": {"type": "platt", "engine": "builtin", "format": "json", "feature_schema_version": identity["feature_schema_version"], "normalization_version": "alignment-text-normalization-v1", "artifact_path": artifact_path, "artifact_sha256": artifact_sha},
        "dataset": {**hashes, "calibration_count": int(artifact.get("sample_count", 0)), "validation_count": validation.count, "hidden_test_count": hidden.count},
        "metrics": {"hidden_false_pass_count": hidden.false_pass_count, "hidden_false_fail_count": hidden.false_fail_count, "brier_score": hidden.brier_score, "expected_calibration_error": hidden.expected_calibration_error, "validation": validation.to_dict()},
        "provenance": {**dict(provenance), "created_at": datetime.now(timezone.utc).isoformat(), "hidden_test_run_id": hidden.run_id},
    }
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(profile, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return profile

__all__ = ["PromotionError", "promote_profile"]
