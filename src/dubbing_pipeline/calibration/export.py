"""Safe JSON export/import for draft calibration artifacts."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Mapping
from .features import FEATURE_SCHEMA_VERSION, NORMALIZATION_VERSION
from .train import CalibrationArtifact

def export_draft(artifact: CalibrationArtifact, path: str | Path) -> str:
    if artifact.status != "DRAFT": raise ValueError("only DRAFT artifacts can be created by commit 3")
    payload = artifact.to_dict(); destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    import hashlib
    return hashlib.sha256(destination.read_bytes()).hexdigest()

def load_draft(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "platt-calibrator-v1" or value.get("status") != "DRAFT": raise ValueError("artifact is not a draft platt calibrator")
    if value.get("feature_schema_version") not in {FEATURE_SCHEMA_VERSION, "final-anchor-v1", "lid-fusion-v1"} or value.get("normalization_version") != NORMALIZATION_VERSION: raise ValueError("calibration schema mismatch")
    features = list(value.get("features") or []); coefficients = list(value.get("coefficients") or [])
    normalization = value.get("normalization")
    if len(features) != len(coefficients) or not isinstance(normalization, list) or len(normalization) != len(features) or not all(math.isfinite(float(x)) for x in coefficients + [value.get("intercept", 0.0)]): raise ValueError("invalid coefficients")
    if not all(isinstance(item, dict) and math.isfinite(float(item.get("mean"))) and math.isfinite(float(item.get("scale"))) and float(item.get("scale")) > 0.0 for item in normalization): raise ValueError("invalid normalization")
    return value

__all__ = ["export_draft", "load_draft"]
