"""Text -> event -> audio mapping and subtitle authority checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import Line, Scene
from .contracts import ContractError
from .contracts.manifest import normalize_manifest, validate_scene_value

TOPOLOGIES = {"LINE_SEPARATED", "IN_ENGINE_TIMELINE", "EMBEDDED_FMV"}


class MappingError(ValueError):
    pass


def load_scene(path: str | Path) -> Scene:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = Scene.from_dict(value)
    validate_scene(scene)
    return scene


def validate_scene(scene: Scene) -> None:
    if scene.topology not in TOPOLOGIES:
        raise MappingError(f"unsupported topology: {scene.topology}")
    seen: set[str] = set()
    windows: list[tuple[float, float, str]] = []
    for line in scene.lines:
        if line.id in seen:
            raise MappingError(f"duplicate line id: {line.id}")
        seen.add(line.id)
        if line.end < line.start:
            raise MappingError(f"negative window: {line.id}")
        if line.start < 0:
            raise MappingError(f"negative start: {line.id}")
        if scene.topology == "EMBEDDED_FMV":
            missing = [key for key, value in {
                "movie_identity_verified": scene.movie_identity_verified or line.movie_identity_verified,
                "card_identity_verified": line.card_identity_verified,
                "card_timebase_verified": line.card_timebase_verified,
            }.items() if not value and line.subtitle_authorized]
            if missing:
                raise MappingError(f"{line.id}: missing FMV evidence {missing}")
            windows.append((float(line.start), float(line.end), line.id))
    if scene.topology == "EMBEDDED_FMV":
        for previous, current in zip(sorted(windows), sorted(windows)[1:]):
            if current[0] < previous[1] - 1e-9 and not bool(scene.metadata.get("extensions", {}).get("allow_overlap")):
                raise MappingError(f"undeclared overlapping FMV windows: {previous[2]} and {current[2]}")


def validate_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        scenes = normalize_manifest(value)
    except ContractError as exc:
        raise MappingError(str(exc)) from exc
    return {"scene_count": len(scenes), "line_count": sum(len(raw.get("lines", [])) for raw in scenes), "topologies": sorted({raw.get("topology", "LINE_SEPARATED") for raw in scenes})}


def authorized_lines(scene: Scene) -> Iterable[Line]:
    return (line for line in scene.lines if line.subtitle_authorized)
