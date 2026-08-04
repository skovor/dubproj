"""Containment-checked, crash-recoverable package deployment."""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .hashing import atomic_json, sha256_file


class DeploymentError(ValueError):
    pass


def _contained(root: Path, candidate: Path) -> Path:
    root = root.resolve(); resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DeploymentError(f"path escapes root: {candidate}") from exc
    return resolved


def validate_relative_destination(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts) or ":" in str(path):
        raise DeploymentError(f"unsafe relative destination: {value}")
    return path


@dataclass(frozen=True)
class PackageEntry:
    source: Path
    relative_destination: Path


def stage_files_v2(files: Iterable[PackageEntry], stage_root: str | Path) -> list[dict]:
    stage = Path(stage_root).resolve()
    if stage.exists():
        raise DeploymentError(f"stage already exists: {stage}")
    stage.mkdir(parents=True)
    rows: list[dict] = []
    seen: set[str] = set()
    for item in files:
        relative = validate_relative_destination(item.relative_destination)
        key = str(relative).casefold()
        if key in seen:
            raise DeploymentError(f"duplicate package destination: {relative}")
        seen.add(key)
        source = Path(item.source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = _contained(stage, stage / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"source": str(source), "stage": str(target), "relative_destination": str(relative), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return rows


def deploy_atomic_v2(files: Iterable[PackageEntry], stage_root: str | Path, runtime_root: str | Path, backup_root: str | Path, *, failure_injector: Callable[[int, Path], None] | None = None, lab_mode: bool = True, sandbox_root: str | Path | None = None) -> dict:
    """Apply all files or restore the exact EXISTS/ABSENT pre-state."""
    stage = Path(stage_root).resolve(); runtime = Path(runtime_root).resolve(); backups = Path(backup_root).resolve()
    if lab_mode and sandbox_root is not None:
        _contained(Path(sandbox_root), runtime)
        _contained(Path(sandbox_root), stage)
        _contained(Path(sandbox_root), backups)
    items = list(files)
    seen: set[str] = set(); rows: list[dict] = []
    for item in items:
        relative = validate_relative_destination(item.relative_destination)
        key = str(relative).casefold()
        if key in seen:
            raise DeploymentError(f"duplicate destination: {relative}")
        seen.add(key)
        source = _contained(stage, stage / relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _contained(runtime, runtime / relative)
        previous_exists = destination.is_file()
        rows.append({"relative_destination": str(relative), "destination": str(destination), "source": str(source), "previous_state": "EXISTS" if previous_exists else "ABSENT", "previous_sha256": sha256_file(destination) if previous_exists else None})
    transaction_id = f"tx-{uuid.uuid4().hex}"
    transaction_dir = backups / transaction_id
    transaction_dir.mkdir(parents=True, exist_ok=False)
    journal = transaction_dir / "journal.json"
    atomic_json(journal, {"transaction_id": transaction_id, "state": "PLANNED", "rows": rows})
    deployed: list[dict] = []
    try:
        for index, row in enumerate(rows):
            destination = Path(row["destination"]); source = Path(row["source"])
            if row["previous_state"] == "EXISTS":
                backup = transaction_dir / Path(row["relative_destination"])
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
                if sha256_file(backup) != row["previous_sha256"]:
                    raise DeploymentError(f"backup hash mismatch: {destination}")
                row["backup"] = str(backup)
            atomic_json(journal, {"transaction_id": transaction_id, "state": "APPLYING", "index": index, "rows": rows})
            if failure_injector:
                failure_injector(index, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(prefix=f".{destination.name}.{transaction_id}.", suffix=".deploying", dir=str(destination.parent))
            os.close(handle)
            temporary = Path(name)
            try:
                shutil.copy2(source, temporary)
                if sha256_file(temporary) != sha256_file(source):
                    raise DeploymentError(f"staging hash mismatch: {source}")
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            actual = sha256_file(destination)
            if actual != sha256_file(source):
                raise DeploymentError(f"deployment hash mismatch: {destination}")
            deployed.append({"destination": str(destination), "sha256": actual, "bytes": destination.stat().st_size})
        atomic_json(journal, {"transaction_id": transaction_id, "state": "COMMITTED", "rows": rows, "deployed": deployed})
    except Exception:
        # Restore existing files and explicitly remove every newly-created file.
        for row in reversed(rows):
            destination = Path(row["destination"])
            backup = Path(row["backup"]) if row.get("backup") else None
            if row["previous_state"] == "EXISTS" and backup and backup.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, destination)
            elif row["previous_state"] == "ABSENT" and destination.exists():
                destination.unlink()
        atomic_json(journal, {"transaction_id": transaction_id, "state": "ROLLED_BACK", "rows": rows})
        raise
    return {"transaction_id": transaction_id, "journal": str(journal), "backup": [row for row in rows if row.get("backup")], "deployed": deployed, "runtime_smoke_pending": True}


__all__ = ["DeploymentError", "PackageEntry", "deploy_atomic_v2", "stage_files_v2", "validate_relative_destination"]
