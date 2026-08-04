"""Stable, portable hashes and crash-safe manifest writes."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def contract_hash(kind: str, payload: dict[str, Any], files: list[str | Path] = (), *, include_paths: bool = False) -> str:
    """Hash semantic file bytes, not machine-specific absolute paths by default."""
    entries = []
    for item in files:
        path = Path(item).resolve()
        entry: dict[str, Any] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        if include_paths:
            entry["path"] = str(path)
        entries.append(entry)
    return sha256_bytes(canonical_json({"schema": "generic-dubbing-contract-v2", "kind": kind, "payload": payload, "files": entries}))


def atomic_bytes(path: str | Path, data: bytes) -> None:
    """Write, fsync, read back and replace using a per-writer temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise IOError(f"atomic write readback mismatch: {temporary}")
        os.replace(temporary, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Windows can reject fsync on a directory handle; replace itself
            # is still atomic and the file data was fsynced above.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value) + b"\n")
