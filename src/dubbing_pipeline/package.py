"""Staging, hash manifests, reversible backup and atomic runtime deployment."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import atomic_json, sha256_file


@dataclass(frozen=True)
class PackageFile:
    source: Path
    relative_destination: Path


def stage_files(files: Iterable[PackageFile], stage_root: str | Path) -> list[dict]:
    stage = Path(stage_root); stage.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in files:
        target = stage / item.relative_destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, target)
        rows.append({"source": str(item.source), "stage": str(target), "relative_destination": str(item.relative_destination), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return rows


def backup_destinations(files: Iterable[PackageFile], runtime_root: str | Path, backup_root: str | Path) -> list[dict]:
    runtime, backup = Path(runtime_root), Path(backup_root); rows = []
    for item in files:
        destination = runtime / item.relative_destination
        if not destination.is_file():
            continue
        target = backup / item.relative_destination; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(destination, target)
        if sha256_file(destination) != sha256_file(target):
            raise IOError(f"backup hash mismatch: {destination}")
        rows.append({"destination": str(destination), "backup": str(target), "sha256": sha256_file(destination), "bytes": destination.stat().st_size})
    return rows


def deploy_atomic(files: Iterable[PackageFile], stage_root: str | Path, runtime_root: str | Path, backup_root: str | Path) -> dict:
    """Deploy a whole package with per-file atomic replacement and rollback."""
    items = list(files); stage, runtime = Path(stage_root), Path(runtime_root)
    backup_rows = backup_destinations(items, runtime, backup_root)
    deployed: list[dict] = []
    try:
        for item in items:
            source = stage / item.relative_destination; destination = runtime / item.relative_destination
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".deploying")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            expected = sha256_file(source); actual = sha256_file(destination)
            if expected != actual:
                raise IOError(f"deployment hash mismatch: {destination}")
            deployed.append({"destination": str(destination), "sha256": actual, "bytes": destination.stat().st_size})
    except Exception:
        for row in backup_rows:
            source, destination = Path(row["backup"]), Path(row["destination"])
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".rollback")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
        raise
    return {"backup": backup_rows, "deployed": deployed, "runtime_smoke_pending": True}


def write_package_manifest(path: str | Path, *, stage: str | Path, files: list[dict], deployment: dict | None = None, contract: dict | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    atomic_json(path, {"schema": "generic-dubbing-package-v1", "created_utc": now, "stage": str(stage), "files": files, "deployment": deployment, "contract": contract or {}, "runtime_smoke_pending": True if deployment is None else bool(deployment.get("runtime_smoke_pending", True))})
