"""Minimal adapter protocol for a new game or middleware."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from dubbing_pipeline.fmv_container import ContainerAdapter
from dubbing_pipeline.models import Line


class GameAdapter(Protocol):
    name: str

    def inventory(self, project_root: Path) -> dict: ...
    def resolve_reference(self, line: Line, project_root: Path) -> Path: ...
    def runtime_destinations(self, asset_name: str, project_root: Path) -> Iterable[Path]: ...
    def container(self) -> ContainerAdapter | None: ...
    def runtime_smoke(self, project_root: Path) -> dict: ...
