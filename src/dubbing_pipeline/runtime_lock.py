"""Reproducibility locks, live-environment checks and backend identity gates.

The generic source cannot guess a user's CUDA driver, model checkout or model
file hashes. It therefore ships an explicit *unprovisioned* lock template and
fails closed for real runs until ``freeze_runtime.py`` has produced a complete
lock and the current machine still matches it.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json, sha256_file


RUNTIME_LOCK_SCHEMA = "generic-dubbing-runtime-lock-v2"
MODELS_LOCK_SCHEMA = "generic-dubbing-model-lock-v2"
UNKNOWN_VALUES = frozenset({
    "", "unknown", "latest", "main", "master", "null", "none",
    "unprovisioned", "replace_me", "not-installed", "not-readable",
})
COMPONENT_STATES = frozenset({"INSTALLED", "DISABLED_EXPLICITLY", "REQUIRED_BUT_MISSING", "UNKNOWN"})
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

DEFAULT_CAPABILITIES: dict[str, dict[str, Any]] = {
    "generation": {"enabled": True, "requires": ["omnivoice", "pytorch", "torchaudio"]},
    "asr_screening": {"enabled": True, "requires": ["faster_whisper", "ctranslate2"]},
    "ctc_alignment": {"enabled": True, "requires": ["whisperx"]},
    "independent_lid": {"enabled": False, "requires": ["speechbrain"]},
    "mfa_fallback": {"enabled": False, "requires": ["mfa"]},
}
REQUIRED_RUNTIME_COMPONENTS = (
    "python", "windows", "cuda", "nvidia_driver", "pytorch", "torchaudio",
    "faster_whisper", "ctranslate2", "whisperx", "speechbrain", "mfa",
    "ffmpeg", "omnivoice",
)


class RuntimeLockError(ValueError):
    """Raised when a lock, live snapshot or backend identity is unsafe."""


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().casefold() in UNKNOWN_VALUES


def _pinned(value: Any) -> bool:
    return not _is_unknown(value)


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


def _capabilities(value: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    source = value if isinstance(value, Mapping) else DEFAULT_CAPABILITIES
    result: dict[str, dict[str, Any]] = {}
    for name, raw in source.items():
        if not isinstance(raw, Mapping):
            result[str(name)] = {"enabled": False, "requires": []}
            continue
        result[str(name)] = {
            "enabled": bool(raw.get("enabled", False)),
            "requires": [str(item) for item in raw.get("requires", [])],
        }
    return result


def _component(value: Mapping[str, Any], name: str) -> tuple[str, Any]:
    components = value.get("components")
    if isinstance(components, Mapping) and isinstance(components.get(name), Mapping):
        item = components[name]
        return str(item.get("status", "UNKNOWN")).upper(), item.get("version")
    # v1 compatibility is read-only; a strict v2 lock must use explicit state.
    dependencies = value.get("dependencies")
    raw = dependencies.get(name) if isinstance(dependencies, Mapping) else None
    if _is_unknown(raw):
        return "UNKNOWN", raw
    return "INSTALLED", raw


def runtime_completion_errors(value: Mapping[str, Any], *, required_capabilities: Mapping[str, Any] | None = None) -> list[str]:
    """Check whether a collected snapshot is complete without trusting status."""
    errors: list[str] = []
    environment = value.get("environment")
    if not isinstance(environment, Mapping):
        return ["runtime environment is missing"]
    for field in ("python", "windows", "architecture", "device"):
        if not _pinned(environment.get(field)):
            errors.append(f"runtime environment {field} is not pinned")
    device = str(environment.get("device", "")).casefold()
    if device == "cuda":
        for field in ("torch_cuda_build", "cuda_runtime", "nvidia_driver", "gpu_name", "compute_capability"):
            if not _pinned(environment.get(field)):
                errors.append(f"CUDA environment {field} is not captured")
        if not isinstance(environment.get("vram_bytes"), int) or int(environment.get("vram_bytes", 0)) <= 0:
            errors.append("CUDA environment vram_bytes is not captured")
    capabilities = _capabilities(required_capabilities or value.get("capabilities"))
    if not capabilities:
        errors.append("runtime capabilities are missing")
    for capability, policy in capabilities.items():
        if not policy.get("enabled"):
            continue
        for name in policy.get("requires", []):
            status, version = _component(value, name)
            if status != "INSTALLED":
                errors.append(f"capability {capability} requires {name}, status is {status}")
            elif not _pinned(version):
                errors.append(f"capability {capability} requires pinned version for {name}")
    components = value.get("components")
    if not isinstance(components, Mapping):
        errors.append("runtime components with explicit states are missing")
    else:
        for name, item in components.items():
            if not isinstance(item, Mapping):
                errors.append(f"runtime component {name} is malformed")
                continue
            status = str(item.get("status", "UNKNOWN")).upper()
            if status not in COMPONENT_STATES:
                errors.append(f"runtime component {name} has invalid status {status}")
            elif status in {"UNKNOWN", "REQUIRED_BUT_MISSING"}:
                errors.append(f"runtime component {name} is {status}")
            elif status == "INSTALLED" and not _pinned(item.get("version")):
                errors.append(f"runtime component {name} has no pinned installed version")
    return sorted(set(errors))


def validate_runtime_lock(value: Mapping[str, Any], *, strict: bool = True, required_capabilities: Mapping[str, Any] | None = None) -> list[str]:
    """Validate lock shape and, in strict mode, every enabled capability."""
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
        for field in ("python", "windows", "architecture", "device"):
            if field not in environment or _is_unknown(environment.get(field)):
                if strict:
                    errors.append(f"runtime environment {field} is not pinned")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping):
        errors.append("runtime lock dependencies are missing")
    elif strict:
        for name, version in dependencies.items():
            if isinstance(version, str) and _is_unknown(version):
                errors.append(f"runtime dependency {name} uses ambiguous value {version!r}")
    components = value.get("components")
    if not isinstance(components, Mapping):
        errors.append("runtime lock components are missing")
    else:
        for name, item in components.items():
            if not isinstance(item, Mapping) or str(item.get("status", "UNKNOWN")).upper() not in COMPONENT_STATES:
                errors.append(f"runtime component {name} has no valid explicit state")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, Mapping):
        errors.append("runtime lock capabilities are missing")
    if strict:
        errors.extend(runtime_completion_errors(value, required_capabilities=required_capabilities))
    return sorted(set(errors))


def _resolve_model_path(lock: Mapping[str, Any], file_row: Mapping[str, Any], *, base_dir: Path) -> Path | None:
    resolved = file_row.get("resolved_path_at_freeze")
    if isinstance(resolved, str) and resolved and Path(resolved).is_file():
        return Path(resolved)
    logical = file_row.get("logical_path", file_row.get("path"))
    if not isinstance(logical, str) or not logical:
        return None
    candidate = Path(logical)
    if not candidate.is_absolute():
        root = lock.get("models_root", ".")
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = base_dir / root_path
        candidate = root_path / candidate
    return candidate if candidate.is_file() else None


def aggregate_model_sha256(files: list[Mapping[str, Any]]) -> str:
    semantic = [{
        "logical_path": str(item.get("logical_path", item.get("path", ""))),
        "bytes": int(item.get("bytes", 0)),
        "sha256": str(item.get("sha256", "")),
    } for item in files]
    return hashlib.sha256(canonical_json(sorted(semantic, key=lambda item: item["logical_path"]))).hexdigest()


def verify_model_files(value: Mapping[str, Any], *, base_dir: str | Path = ".", strict: bool = True) -> list[str]:
    """Verify current model bytes against every locked file and aggregate hash."""
    errors: list[str] = []
    base = Path(base_dir).resolve()
    models = value.get("models", [])
    if not isinstance(models, list):
        return ["models lock must contain a models array"]
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            continue
        files = model.get("files", [])
        if not isinstance(files, list) or not files:
            if strict:
                errors.append(f"models[{index}] has no locally verifiable files")
            continue
        verified_rows: list[dict[str, Any]] = []
        for file_index, row in enumerate(files):
            if not isinstance(row, Mapping):
                errors.append(f"models[{index}].files[{file_index}] is malformed")
                continue
            path = _resolve_model_path(value, row, base_dir=base)
            if path is None:
                errors.append(f"models[{index}].files[{file_index}] is not present at its logical path")
                continue
            expected_bytes = row.get("bytes")
            expected_sha = row.get("sha256")
            actual_bytes = path.stat().st_size
            actual_sha = sha256_file(path)
            if expected_bytes != actual_bytes:
                errors.append(f"models[{index}].files[{file_index}] byte count changed")
            if expected_sha != actual_sha:
                errors.append(f"models[{index}].files[{file_index}] SHA-256 changed")
            verified_rows.append({
                "logical_path": row.get("logical_path", row.get("path", "")),
                "bytes": expected_bytes,
                "sha256": expected_sha,
            })
        if verified_rows and model.get("sha256") != aggregate_model_sha256(verified_rows):
            errors.append(f"models[{index}] aggregate SHA-256 does not match its files")
    return sorted(set(errors))


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
        for key in ("model_id", "revision", "sha256", "language", "sample_rate", "backend", "backend_id", "backend_version"):
            if key not in model or _is_unknown(model.get(key)):
                if strict:
                    errors.append(f"models[{index}].{key} is not pinned")
        digest = model.get("sha256")
        if strict and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
            errors.append(f"models[{index}].sha256 must be a 64-character SHA-256")
        files = model.get("files", [])
        if not isinstance(files, list):
            errors.append(f"models[{index}].files must be an array")
        elif strict and not files:
            errors.append(f"models[{index}] has no verifiable model files")
        for file_index, row in enumerate(files if isinstance(files, list) else []):
            if not isinstance(row, Mapping):
                errors.append(f"models[{index}].files[{file_index}] must be an object")
                continue
            for key in ("logical_path", "bytes", "sha256"):
                if key not in row or _is_unknown(row.get(key)):
                    if strict:
                        errors.append(f"models[{index}].files[{file_index}].{key} is not pinned")
            if strict and (not isinstance(row.get("bytes"), int) or int(row.get("bytes", -1)) < 0):
                errors.append(f"models[{index}].files[{file_index}].bytes is invalid")
            if strict and (not isinstance(row.get("sha256"), str) or not SHA256_RE.fullmatch(row.get("sha256", ""))):
                errors.append(f"models[{index}].files[{file_index}].sha256 is invalid")
    return sorted(set(errors))


def find_model(value: Mapping[str, Any], model_id: str) -> Mapping[str, Any] | None:
    for item in value.get("models", []) if isinstance(value.get("models"), list) else []:
        if isinstance(item, Mapping) and str(item.get("model_id")) == str(model_id):
            return item
    return None


def compare_runtime_snapshot(lock: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Compare the live host against a lock; GPU UUID is portable warning only."""
    errors: list[str] = []
    warnings: list[str] = []
    locked_env = lock.get("environment", {})
    live_env = snapshot.get("environment", {})
    for field in ("python", "windows", "architecture", "device", "torch_cuda_build", "cuda_runtime", "nvidia_driver", "gpu_name", "compute_capability", "vram_bytes"):
        if field in locked_env and locked_env.get(field) != live_env.get(field):
            errors.append(f"live environment {field} differs from runtime.lock.json")
    if locked_env.get("gpu_uuid") not in (None, "", "disabled") and locked_env.get("gpu_uuid") != live_env.get("gpu_uuid"):
        warnings.append("GPU UUID differs from runtime.lock.json; portability warning")
    locked_components = lock.get("components", {})
    live_components = snapshot.get("components", {})
    if isinstance(locked_components, Mapping) and isinstance(live_components, Mapping):
        for name, locked in locked_components.items():
            live = live_components.get(name, {})
            if not isinstance(locked, Mapping) or not isinstance(live, Mapping):
                errors.append(f"live component {name} is missing")
                continue
            if locked.get("status") != live.get("status"):
                errors.append(f"live component {name} status differs from runtime.lock.json")
            if locked.get("status") == "INSTALLED" and locked.get("version") != live.get("version"):
                errors.append(f"live component {name} version differs from runtime.lock.json")
    return sorted(set(errors)), sorted(set(warnings))


def assert_backend_matches_lock(backend: Any, models_lock: str | Path | Mapping[str, Any], *, role: str, expected_model_id: str | None = None, expected_backend_version: str | None = None) -> dict[str, Any]:
    """Fail before evidence/generation if a loaded backend is not the locked one."""
    lock = load_lock(models_lock, expected_schema=MODELS_LOCK_SCHEMA) if not isinstance(models_lock, Mapping) else dict(models_lock)
    model_id = str(expected_model_id or getattr(backend, "model_id", "unknown"))
    model = find_model(lock, model_id)
    if model is None:
        raise RuntimeLockError(f"{role} backend model {model_id!r} is absent from models.lock.json")
    backend_id = str(getattr(backend, "backend_id", "unknown"))
    revision = str(getattr(backend, "model_revision", "unknown"))
    version = str(expected_backend_version or getattr(backend, "backend_version", "unknown"))
    if _is_unknown(backend_id) or _is_unknown(revision) or _is_unknown(version):
        raise RuntimeLockError(f"{role} backend identity is incomplete")
    if str(model.get("revision")) != revision:
        raise RuntimeLockError(f"{role} model revision differs from models.lock.json")
    locked_backend_id = model.get("backend_id")
    if locked_backend_id is not None and str(locked_backend_id) != backend_id:
        raise RuntimeLockError(f"{role} backend_id differs from models.lock.json")
    if model.get("backend_version") is not None and str(model.get("backend_version")) != version:
        raise RuntimeLockError(f"{role} backend version differs from models.lock.json")
    return {"role": role, "model_id": model_id, "backend_id": backend_id, "model_revision": revision, "backend_version": version}


def reproducibility_report(config: Any, *, strict: bool | None = None) -> dict[str, Any]:
    """Validate lockfiles, live host and model bytes without optional ML imports."""
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
            errors.extend(validate_runtime_lock(runtime, strict=bool(strict), required_capabilities=getattr(config, "capabilities", None)))
        except RuntimeLockError as exc:
            errors.append(str(exc))
    if models_path is None:
        (errors if strict else warnings).append("models_lock path is not configured")
    else:
        try:
            models = load_lock(models_path, expected_schema=MODELS_LOCK_SCHEMA)
            errors.extend(validate_models_lock(models, strict=bool(strict)))
            errors.extend(verify_model_files(models, base_dir=Path(models_path).parent, strict=bool(strict)))
        except RuntimeLockError as exc:
            errors.append(str(exc))
    for field in ("model_revision", "backend_version"):
        value = getattr(config, field, None)
        if _is_unknown(value):
            (errors if strict else warnings).append(f"config.{field} is not pinned")
    if runtime is not None and strict:
        snapshot = collect_runtime_lock(device=str(getattr(config, "device", "cuda")), capabilities=getattr(config, "capabilities", None), omnivoice_version=str(getattr(config, "backend_version", "unknown")))
        live_errors, live_warnings = compare_runtime_snapshot(runtime, snapshot)
        errors.extend(live_errors)
        warnings.extend(live_warnings)
    if models is not None:
        model = find_model(models, str(getattr(config, "model_id", "")))
        if model is None:
            (errors if strict else warnings).append(f"model {getattr(config, 'model_id', '')!r} is absent from models.lock.json")
        else:
            if not _is_unknown(getattr(config, "model_revision", None)) and str(model.get("revision")) != str(getattr(config, "model_revision")):
                errors.append("config.model_revision does not match models.lock.json")
            if not _is_unknown(getattr(config, "backend_version", None)) and str(model.get("backend_version")) != str(getattr(config, "backend_version")):
                errors.append("config.backend_version does not match models.lock.json")
    if not strict and errors:
        warnings.extend(errors)
        errors = []
    status = "PASS" if not errors and not warnings else ("LAB_UNPINNED" if not strict else "BLOCKED")
    return {
        "schema": "generic-dubbing-reproducibility-report-v2",
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
        raise RuntimeLockError("production preflight blocked: " + "; ".join(report["errors"] or report["warnings"]))
    return report


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def executable_version(executable: str) -> str | None:
    path = shutil.which(executable)
    if path is None:
        return None
    try:
        completed = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (completed.stdout or completed.stderr or "").splitlines()
    return first[0].strip() if first else None


def _gpu_snapshot() -> dict[str, Any]:
    result = {"nvidia_driver": None, "gpu_name": None, "gpu_uuid": None, "vram_bytes": None, "compute_capability": None}
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion().decode(errors="replace")
        result["gpu_name"] = pynvml.nvmlDeviceGetName(handle).decode(errors="replace")
        result["gpu_uuid"] = pynvml.nvmlDeviceGetUUID(handle).decode(errors="replace")
        result["vram_bytes"] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
        pynvml.nvmlShutdown()
    except Exception:
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                completed = subprocess.run([executable, "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
                row = (completed.stdout or "").splitlines()[0].split(",") if completed.returncode == 0 and completed.stdout else []
                if len(row) >= 4:
                    result["nvidia_driver"] = row[0].strip()
                    result["gpu_name"] = row[1].strip()
                    result["gpu_uuid"] = row[2].strip()
                    result["vram_bytes"] = int(float(row[3].strip()) * 1024 * 1024)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            result["compute_capability"] = ".".join(str(item) for item in torch.cuda.get_device_capability(0))
            if result["vram_bytes"] is None:
                result["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
            if result["gpu_name"] is None:
                result["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return result


def _state_for(value: Any, *, required: bool) -> dict[str, Any]:
    if value is not None and _pinned(value):
        return {"status": "INSTALLED", "version": str(value)}
    return {"status": "REQUIRED_BUT_MISSING" if required else "DISABLED_EXPLICITLY", "version": None}


def collect_runtime_lock(*, device: str = "cuda", capabilities: Mapping[str, Any] | None = None, omnivoice_version: str | None = None) -> dict[str, Any]:
    """Collect a live snapshot and mark it COMPLETE only if it is usable."""
    caps = _capabilities(capabilities)
    required = {component for policy in caps.values() if policy.get("enabled") for component in policy.get("requires", [])}
    try:
        import torch  # type: ignore
        torch_cuda_build = str(torch.version.cuda) if torch.version.cuda else None
    except Exception:
        torch_cuda_build = None
    gpu = _gpu_snapshot()
    cuda_enabled = str(device).casefold() == "cuda"
    components: dict[str, dict[str, Any]] = {}
    raw_values = {
        "python": platform.python_version(), "windows": platform.release(),
        "cuda": torch_cuda_build, "nvidia_driver": gpu.get("nvidia_driver"),
        "pytorch": package_version("torch"), "torchaudio": package_version("torchaudio"),
        "faster_whisper": package_version("faster-whisper"), "ctranslate2": package_version("ctranslate2"),
        "whisperx": package_version("whisperx"), "speechbrain": package_version("speechbrain"),
        "mfa": package_version("montreal-forced-aligner"), "ffmpeg": executable_version("ffmpeg"),
        "omnivoice": omnivoice_version,
    }
    for name, value in raw_values.items():
        needs = name in required or (name in {"cuda", "nvidia_driver"} and cuda_enabled)
        components[name] = _state_for(value, required=needs)
    environment = {
        "python": platform.python_version(), "windows": platform.release(), "architecture": platform.machine(),
        "device": str(device), "torch_cuda_build": torch_cuda_build if cuda_enabled else "disabled",
        "cuda_runtime": torch_cuda_build if cuda_enabled else "disabled",
        "nvidia_driver": gpu.get("nvidia_driver") if cuda_enabled else "disabled",
        "gpu_name": gpu.get("gpu_name") if cuda_enabled else "disabled",
        "gpu_uuid": gpu.get("gpu_uuid") if cuda_enabled else "disabled",
        "compute_capability": gpu.get("compute_capability") if cuda_enabled else "disabled",
        "vram_bytes": gpu.get("vram_bytes") if cuda_enabled else 0,
    }
    value: dict[str, Any] = {
        "schema": RUNTIME_LOCK_SCHEMA, "lock_version": 2, "status": "UNPROVISIONED",
        "generated_by": "scripts/freeze_runtime.py", "environment": environment,
        "components": components, "dependencies": {name: item.get("version") for name, item in components.items()},
        "capabilities": caps,
    }
    if not runtime_completion_errors(value, required_capabilities=caps):
        value["status"] = "COMPLETE"
    return value


def model_file_entry(path: str | Path, *, logical_path: str | None = None) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.is_file():
        raise RuntimeLockError(f"model file does not exist: {item}")
    return {
        "logical_path": logical_path or item.name,
        "resolved_path_at_freeze": str(item),
        "bytes": item.stat().st_size,
        "sha256": sha256_file(item),
    }


def lock_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


__all__ = [
    "COMPONENT_STATES", "DEFAULT_CAPABILITIES", "MODELS_LOCK_SCHEMA", "RUNTIME_LOCK_SCHEMA", "RuntimeLockError",
    "aggregate_model_sha256", "assert_backend_matches_lock", "assert_reproducible", "collect_runtime_lock",
    "compare_runtime_snapshot", "find_model", "load_lock", "lock_digest", "model_file_entry",
    "reproducibility_report", "runtime_completion_errors", "validate_models_lock", "validate_runtime_lock",
    "verify_model_files",
]
