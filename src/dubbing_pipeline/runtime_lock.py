"""Reproducibility locks and production preflight checks.

The generic source cannot guess a user's CUDA driver, model checkout or model
file hashes.  It therefore ships an explicit *unprovisioned* lock template and
fails closed for real runs until ``freeze_runtime.py`` has produced a complete
lock.  Dependency-light lab/preflight runs may inspect the template, but they
must not be mistaken for a reproducible production run.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json, sha256_file


RUNTIME_LOCK_SCHEMA = "generic-dubbing-runtime-lock-v1"
MODELS_LOCK_SCHEMA = "generic-dubbing-model-lock-v1"
UNKNOWN_VALUES = frozenset({"", "unknown", "latest", "main", "master", "null", "none", "unprovisioned", "replace_me"})

# These names are deliberately explicit.  A missing optional package is a
# valid state for a dependency-light lab, but it cannot silently become a
# production lock with an unrecorded version.
REQUIRED_RUNTIME_COMPONENTS = (
    "python", "windows", "cuda", "nvidia_driver", "pytorch", "torchaudio",
    "faster_whisper", "ctranslate2", "whisperx", "speechbrain", "mfa",
    "ffmpeg", "omnivoice",
)


class RuntimeLockError(ValueError):
    """Raised when a lock is malformed or not sufficiently pinned."""


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in UNKNOWN_VALUES
    return False


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeLockError(f"{label} must be an object")
    return value


def load_lock(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    lock_path = Path(path)
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeLockError(f"cannot read lock {lock_path}: {exc}") from exc
    value = dict(_require_mapping(value, str(lock_path)))
    if value.get("schema") != expected_schema:
        raise RuntimeLockError(f"{lock_path}: expected schema {expected_schema!r}")
    value.setdefault("lock_path", str(lock_path.resolve()))
    return value


def validate_runtime_lock(value: Mapping[str, Any], *, strict: bool = True) -> list[str]:
    """Return errors for a runtime lock; strict mode is required in production."""
    errors: list[str] = []
    if value.get("schema") != RUNTIME_LOCK_SCHEMA:
        errors.append("runtime lock schema is missing or unsupported")
    status = str(value.get("status", "")).upper()
    if strict and status != "COMPLETE":
        errors.append(f"runtime lock status must be COMPLETE, got {status or 'missing'}")
    environment = value.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("runtime lock environment is missing")
    else:
        for component in ("python", "windows", "cuda", "nvidia_driver"):
            if component not in environment or _is_unknown(environment.get(component)):
                if strict:
                    errors.append(f"runtime environment {component} is not pinned")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping):
        errors.append("runtime lock dependencies are missing")
    else:
        for component in REQUIRED_RUNTIME_COMPONENTS:
            if component not in dependencies or _is_unknown(dependencies.get(component)):
                if strict:
                    errors.append(f"runtime dependency {component} is not pinned")
    return errors


def validate_models_lock(value: Mapping[str, Any], *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    if value.get("schema") != MODELS_LOCK_SCHEMA:
        errors.append("models lock schema is missing or unsupported")
    status = str(value.get("status", "")).upper()
    if strict and status != "COMPLETE":
        errors.append(f"models lock status must be COMPLETE, got {status or 'missing'}")
    models = value.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models lock must contain a non-empty models array")
        return errors
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            errors.append(f"models[{index}] must be an object")
            continue
        for key in ("model_id", "revision", "sha256", "language", "sample_rate", "backend", "backend_version"):
            if key not in model or _is_unknown(model.get(key)):
                if strict:
                    errors.append(f"models[{index}].{key} is not pinned")
        digest = model.get("sha256")
        if strict and (not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest)):
            errors.append(f"models[{index}].sha256 must be a 64-character SHA-256")
        files = model.get("files", [])
        if not isinstance(files, list):
            errors.append(f"models[{index}].files must be an array")
    return errors


def find_model(value: Mapping[str, Any], model_id: str) -> Mapping[str, Any] | None:
    for item in value.get("models", []) if isinstance(value.get("models"), list) else []:
        if isinstance(item, Mapping) and str(item.get("model_id")) == str(model_id):
            return item
    return None


def reproducibility_report(config: Any, *, strict: bool | None = None) -> dict[str, Any]:
    """Validate config + lockfiles without importing optional ML packages."""
    if strict is None:
        strict = not bool(getattr(config, "lab_mode", True))
    errors: list[str] = []
    warnings: list[str] = []
    runtime_path = getattr(config, "runtime_lock", None)
    models_path = getattr(config, "models_lock", None)
    runtime: dict[str, Any] | None = None
    models: dict[str, Any] | None = None
    if runtime_path is None:
        (errors if strict else warnings).append("runtime_lock path is not configured")
    else:
        try:
            runtime = load_lock(runtime_path, expected_schema=RUNTIME_LOCK_SCHEMA)
            errors.extend(validate_runtime_lock(runtime, strict=bool(strict)))
        except RuntimeLockError as exc:
            errors.append(str(exc))
    if models_path is None:
        (errors if strict else warnings).append("models_lock path is not configured")
    else:
        try:
            models = load_lock(models_path, expected_schema=MODELS_LOCK_SCHEMA)
            errors.extend(validate_models_lock(models, strict=bool(strict)))
        except RuntimeLockError as exc:
            errors.append(str(exc))
    for field in ("model_revision", "backend_version"):
        value = getattr(config, field, None)
        if _is_unknown(value):
            (errors if strict else warnings).append(f"config.{field} is not pinned")
    if models is not None:
        model = find_model(models, str(getattr(config, "model_id", "")))
        if model is None:
            (errors if strict else warnings).append(f"model {getattr(config, 'model_id', '')!r} is absent from models.lock.json")
        elif not _is_unknown(getattr(config, "model_revision", None)) and str(model.get("revision")) != str(getattr(config, "model_revision")):
            errors.append("config.model_revision does not match models.lock.json")
        elif not _is_unknown(getattr(config, "backend_version", None)) and str(model.get("backend_version")) != str(getattr(config, "backend_version")):
            errors.append("config.backend_version does not match models.lock.json")
    if not strict and errors:
        warnings.extend(errors)
        errors = []
    status = "PASS" if not errors and not warnings else ("LAB_UNPINNED" if not strict else "BLOCKED")
    return {
        "schema": "generic-dubbing-reproducibility-report-v1",
        "status": status,
        "strict": bool(strict),
        "runtime_lock": str(runtime_path) if runtime_path is not None else None,
        "models_lock": str(models_path) if models_path is not None else None,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def assert_reproducible(config: Any, *, strict: bool | None = None) -> dict[str, Any]:
    report = reproducibility_report(config, strict=strict)
    if report["status"] != "PASS":
        raise RuntimeLockError("production preflight blocked: " + "; ".join(report["errors"]))
    return report


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def executable_version(executable: str) -> str:
    path = shutil.which(executable)
    if path is None:
        return "not-installed"
    try:
        completed = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "not-readable"
    first = (completed.stdout or completed.stderr or "").splitlines()
    return first[0].strip() if first else "not-readable"


def collect_runtime_lock(*, omnivoice_version: str = "not-installed") -> dict[str, Any]:
    """Collect exact host/package versions without importing ML runtimes."""
    try:
        import torch  # type: ignore
        cuda = str(torch.version.cuda or "not-installed")
    except Exception:
        cuda = "not-installed"
    dependencies = {
        "python": platform.python_version(),
        "windows": platform.release(),
        "cuda": cuda,
        "nvidia_driver": "not-installed",
        "pytorch": package_version("torch"),
        "torchaudio": package_version("torchaudio"),
        "faster_whisper": package_version("faster-whisper"),
        "ctranslate2": package_version("ctranslate2"),
        "whisperx": package_version("whisperx"),
        "speechbrain": package_version("speechbrain"),
        "mfa": package_version("montreal-forced-aligner"),
        "ffmpeg": executable_version("ffmpeg"),
        "omnivoice": omnivoice_version,
    }
    return {
        "schema": RUNTIME_LOCK_SCHEMA,
        "lock_version": 1,
        "status": "COMPLETE",
        "generated_by": "scripts/freeze_runtime.py",
        "environment": {
            "python": platform.python_version(),
            "windows": platform.release(),
            "architecture": platform.machine(),
            "cuda": cuda,
            "nvidia_driver": "not-installed",
        },
        "dependencies": dependencies,
    }


def model_file_entry(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.is_file():
        raise RuntimeLockError(f"model file does not exist: {item}")
    return {"path": str(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)}


def lock_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


__all__ = [
    "MODELS_LOCK_SCHEMA", "RUNTIME_LOCK_SCHEMA", "RuntimeLockError",
    "assert_reproducible", "collect_runtime_lock", "find_model", "load_lock",
    "lock_digest", "model_file_entry", "reproducibility_report",
    "validate_models_lock", "validate_runtime_lock",
]
