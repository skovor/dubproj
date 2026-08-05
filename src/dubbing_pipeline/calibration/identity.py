"""Resolve one explicit alignment model identity for calibration promotion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ModelIdentityError(ValueError):
    pass


def _models(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    result: list[Mapping[str, Any]] = []
    for key in ("models", "model_roles", "components", "backends"):
        items = value.get(key)
        if isinstance(items, list):
            result.extend(item for item in items if isinstance(item, Mapping))
        elif isinstance(items, Mapping):
            result.extend(dict(item, role=name) for name, item in items.items() if isinstance(item, Mapping))
    for key in ("alignment", "alignment_model"):
        item = value.get(key)
        if isinstance(item, Mapping):
            result.append(dict(item, role="alignment"))
    if value.get("role") == "alignment":
        result.append(value)
    return result


def _candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    def required(*keys: str) -> str:
        for key in keys:
            text = str(value.get(key, "") or "").strip()
            if text:
                return text
        raise ModelIdentityError(f"alignment identity is missing {keys[0]}")

    identity = {
        "role": "alignment",
        "backend_id": required("backend_id", "backend"),
        "model_id": required("model_id", "model"),
        "model_revision": required("model_revision", "revision"),
        "source_language": required("source_language", "source_lang"),
        "target_language": required("target_language", "target_lang"),
        "feature_schema_version": required("feature_schema_version", "schema"),
    }
    modes = value.get("performance_modes")
    if not isinstance(modes, list) or not modes or any(not str(item).strip() for item in modes):
        raise ModelIdentityError("alignment identity must declare performance_modes")
    identity["performance_modes"] = sorted({str(item).strip() for item in modes})
    return identity


def resolve_alignment_identity(config_path: str | Path, models_lock_path: str | Path | None = None) -> dict[str, Any]:
    """Resolve exactly one role, optionally cross-checking the model lock."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config_candidates = [item for item in _models(config) if str(item.get("role", "")).strip().lower() == "alignment"]
    if len(config_candidates) != 1:
        raise ModelIdentityError(f"config must contain exactly one alignment role; found {len(config_candidates)}")
    identity = _candidate(config_candidates[0])
    if models_lock_path is not None:
        lock = json.loads(Path(models_lock_path).read_text(encoding="utf-8"))
        lock_candidates = [item for item in _models(lock) if str(item.get("role", "")).strip().lower() == "alignment"]
        if len(lock_candidates) != 1:
            raise ModelIdentityError(f"models lock must contain exactly one alignment role; found {len(lock_candidates)}")
        locked = _candidate(lock_candidates[0])
        if any(identity[key] != locked[key] for key in ("backend_id", "model_id", "model_revision", "source_language", "target_language", "feature_schema_version", "performance_modes")):
            raise ModelIdentityError("config alignment identity does not match models lock")
    return identity


__all__ = ["ModelIdentityError", "resolve_alignment_identity"]
