"""Spoiler-safe review bundle for GPT or human QA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_json


def build_review_bundle(manifest: str | Path, *, output: str | Path, include_text: bool = False) -> dict[str, Any]:
    source = json.loads(Path(manifest).read_text(encoding="utf-8"))
    bundle: dict[str, Any] = {"schema": "generic-dubbing-review-v1", "source_manifest": str(Path(manifest).resolve()), "scenes": []}
    scenes = source.get("scenes", source if isinstance(source, list) else [])
    for scene in scenes:
        item = {"id": scene.get("id"), "topology": scene.get("topology"), "line_count": len(scene.get("lines", [])), "lines": []}
        for line in scene.get("lines", []):
            row = {key: line.get(key) for key in ("id", "start", "end", "subtitle_authorized", "force_keep_original", "preserve_reason")}
            if include_text:
                row.update({key: line.get(key) for key in ("source_text", "target_text")})
            item["lines"].append(row)
        bundle["scenes"].append(item)
    atomic_json(output, bundle)
    return bundle
