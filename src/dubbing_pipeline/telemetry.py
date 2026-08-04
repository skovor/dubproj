"""Run-scoped stage telemetry; no cumulative cross-run timing rows."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .hashing import atomic_json


@dataclass
class TelemetryEvent:
    run_id: str
    stage_id: str
    scene_id: str | None = None
    line_id: str | None = None
    candidate_hash: str | None = None
    started_at: str = ""
    ended_at: str = ""
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    gpu_seconds: float | None = None
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None
    cache_hit: bool = False
    status: str = "UNKNOWN"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class TelemetryCollector:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex}"
        self.events: list[TelemetryEvent] = []

    def stage(self, stage_id: str, **details: Any):
        collector = self
        class _Stage:
            def __enter__(self):
                self.start = time.perf_counter(); self.cpu = time.process_time(); self.started_at = datetime.now(timezone.utc).isoformat(); return self
            def __exit__(self, exc_type, exc, tb):
                ended = datetime.now(timezone.utc).isoformat()
                event = TelemetryEvent(collector.run_id, stage_id, started_at=self.started_at, ended_at=ended,
                                       wall_seconds=time.perf_counter() - self.start, cpu_seconds=time.process_time() - self.cpu,
                                       status="ERROR" if exc else "PASS", details=dict(details))
                collector.events.append(event)
                return False
        return _Stage()

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, Any]] = {}
        for event in self.events:
            row = by_stage.setdefault(event.stage_id, {"wall_seconds": 0.0, "events": 0, "errors": 0})
            row["wall_seconds"] += event.wall_seconds; row["events"] += 1; row["errors"] += int(event.status == "ERROR")
        return {"run_id": self.run_id, "events": [item.to_dict() for item in self.events], "by_stage": by_stage}

    def write(self, path: str | os.PathLike[str]) -> None:
        atomic_json(path, self.summary())


__all__ = ["TelemetryCollector", "TelemetryEvent"]
