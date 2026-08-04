"""Independent-file/bank route for VN and in-engine dialogue."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .audio import AudioSpec, spec
from .hashing import atomic_json, sha256_file


def inventory_line_assets(root: str | Path, *, extensions: tuple[str, ...] = (".wav", ".ogg", ".wem", ".hca")) -> list[dict]:
    base = Path(root)
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.suffix.lower() in extensions):
        row = {"path": str(path), "relative": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix.lower() == ".wav":
            audio = spec(path); row["audio"] = {"frames": audio.frames, "sample_rate": audio.sample_rate, "channels": audio.channels}
        rows.append(row)
    return rows


def write_line_manifest(root: str | Path, output: str | Path) -> list[dict]:
    rows = inventory_line_assets(root)
    atomic_json(output, {"schema": "generic-line-asset-inventory-v1", "root": str(Path(root).resolve()), "assets": rows})
    return rows


def verify_replacement(original: str | Path, replacement: str | Path, *, expected: AudioSpec | None = None) -> None:
    """Check the target contract before a bank/container adapter consumes a WAV."""
    original_spec = spec(original)
    replacement_spec = spec(replacement)
    expected = expected or original_spec
    if replacement_spec != expected:
        raise ValueError(f"replacement contract mismatch: expected={expected} got={replacement_spec}")
