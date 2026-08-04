"""Staging, hash manifests, reversible backup and atomic runtime deployment."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import atomic_json, sha256_file
from .deploy_v2 import DeploymentError, PackageEntry, deploy_atomic_v2, stage_files_v2, validate_relative_destination


@dataclass(frozen=True)
class PackageFile:
    source: Path
    relative_destination: Path


def stage_files(files: Iterable[PackageFile], stage_root: str | Path) -> list[dict]:
    return stage_files_v2([PackageEntry(Path(item.source), validate_relative_destination(item.relative_destination)) for item in files], stage_root)


def backup_destinations(files: Iterable[PackageFile], runtime_root: str | Path, backup_root: str | Path) -> list[dict]:
    runtime, backup = Path(runtime_root).resolve(), Path(backup_root).resolve(); rows = []
    for item in files:
        relative = validate_relative_destination(item.relative_destination)
        destination = (runtime / relative).resolve()
        try:
            destination.relative_to(runtime)
        except ValueError as exc:
            raise DeploymentError(f"destination escapes runtime root: {relative}") from exc
        if not destination.is_file():
            continue
        target = (backup / relative).resolve(); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(destination, target)
        if sha256_file(destination) != sha256_file(target):
            raise IOError(f"backup hash mismatch: {destination}")
        rows.append({"destination": str(destination), "backup": str(target), "sha256": sha256_file(destination), "bytes": destination.stat().st_size})
    return rows


def deploy_atomic(files: Iterable[PackageFile], stage_root: str | Path, runtime_root: str | Path, backup_root: str | Path) -> dict:
    """Deploy a whole package with full EXISTS/ABSENT rollback and safe paths."""
    items = list(files)
    return deploy_atomic_v2([PackageEntry(Path(item.source), validate_relative_destination(item.relative_destination)) for item in items], stage_root, runtime_root, backup_root, lab_mode=False)


def write_package_manifest(path: str | Path, *, stage: str | Path, files: list[dict], deployment: dict | None = None, contract: dict | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    atomic_json(path, {"schema": "generic-dubbing-package-v1", "created_utc": now, "stage": str(stage), "files": files, "deployment": deployment, "contract": contract or {}, "runtime_smoke_pending": True if deployment is None else bool(deployment.get("runtime_smoke_pending", True))})
