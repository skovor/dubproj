"""Fail-closed profile promotion with recomputed dataset hashes."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from .train import CalibrationArtifact
from .validate import ValidationReport

class PromotionError(ValueError): pass

def promote_profile(*, profile_id: str, target_artifact: CalibrationArtifact | Mapping[str, Any], final_anchor_artifact: CalibrationArtifact | Mapping[str, Any] | None = None, validation: ValidationReport, hidden: ValidationReport, dataset_files: Mapping[str, str | Path], identity: Mapping[str, Any], thresholds: Mapping[str, float], provenance: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
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
    if final_anchor_artifact is None: raise PromotionError("target and final-anchor artifacts are both required")
    artifact = target_artifact.to_dict() if isinstance(target_artifact, CalibrationArtifact) else dict(target_artifact)
    anchor = final_anchor_artifact.to_dict() if isinstance(final_anchor_artifact, CalibrationArtifact) else dict(final_anchor_artifact)
    def artifact_spec(value: Mapping[str, Any], role: str) -> dict[str, Any]:
        artifact_path = str(value.get("artifact_path", "")); artifact_sha = str(value.get("artifact_sha256", "")).casefold()
        if not artifact_path or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha): raise PromotionError(f"{role} artifact must be hashed")
        expected_schema = "final-anchor-v1" if role == "final_anchor" else str(identity["feature_schema_version"])
        if str(value.get("feature_schema_version", expected_schema)) != expected_schema: raise PromotionError(f"{role} feature schema mismatch")
        return {"type":"platt","engine":"builtin","format":"json","feature_schema_version":expected_schema,"normalization_version":"alignment-text-normalization-v1","feature_names":list(value.get("features") or value.get("feature_names") or []),"artifact_path":artifact_path,"artifact_sha256":artifact_sha}
    target_spec = artifact_spec(artifact, "target"); anchor_spec = artifact_spec(anchor, "final_anchor")
    profile = {
        "schema": "generic-dubbing-alignment-calibration-profile-v2", "status": "VALIDATED", "authority": True, "profile_id": profile_id,
        "identity": dict(identity), "thresholds": {key: float(value) for key, value in thresholds.items()},
        "calibrators": {"target": target_spec, "final_anchor": anchor_spec},
        "dataset": {**hashes, "calibration_count": int(artifact.get("sample_count", 0)), "validation_count": validation.count, "hidden_test_count": hidden.count},
        "metrics": {"hidden_false_pass_count": hidden.false_pass_count, "hidden_false_fail_count": hidden.false_fail_count, "brier_score": hidden.brier_score, "expected_calibration_error": hidden.expected_calibration_error, "validation": validation.to_dict()},
        "provenance": {**dict(provenance), "created_at": datetime.now(timezone.utc).isoformat(), "hidden_test_run_id": hidden.run_id},
    }
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(profile, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return profile

__all__ = ["PromotionError", "promote_profile"]
