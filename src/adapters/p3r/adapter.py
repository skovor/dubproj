"""Evidence-backed P3R adapter without production-path defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubbing_pipeline.mapping import validate_manifest


@dataclass(frozen=True)
class P3RAdapterConfig:
    project_root: Path
    manifest: Path
    runtime_root: Path | None = None
    source_language: str = "en"
    target_language: str = "de"

    @classmethod
    def load(cls, path: str | Path) -> "P3RAdapterConfig":
        config_path = Path(path).resolve(); value = json.loads(config_path.read_text(encoding="utf-8")); base = config_path.parent
        def resolve(item: str | None) -> Path | None:
            if item in (None, ""): return None
            candidate = Path(item); return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        return cls(project_root=resolve(value.get("project_root")) or base, manifest=resolve(value.get("manifest")) or (base / "manifest.json"), runtime_root=resolve(value.get("runtime_root")), source_language=str(value.get("source_language", "en")), target_language=str(value.get("target_language", "de")))


class P3RAdapter:
    name = "p3r"

    def __init__(self, config: P3RAdapterConfig):
        self.config = config

    def inventory(self) -> dict[str, Any]:
        summary = validate_manifest(self.config.manifest)
        return {"adapter": self.name, "project_root": str(self.config.project_root), **summary}

    def runtime_destinations(self, asset_name: str) -> list[Path]:
        if self.config.runtime_root is None:
            raise RuntimeError("P3R runtime_root is not configured")
        return [self.config.runtime_root / asset_name]

    def runtime_smoke(self) -> dict[str, Any]:
        return {"status": "NOT_RUN", "reason": "game runtime smoke requires explicit external invocation", "runtime_root": str(self.config.runtime_root) if self.config.runtime_root else None}


__all__ = ["P3RAdapter", "P3RAdapterConfig"]
