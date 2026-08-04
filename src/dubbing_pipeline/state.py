"""Append-only resumable state journal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import RunState
from .hashing import atomic_json


class StateStore:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.journal_path = self.root / "journal.jsonl"

    def load(self) -> RunState | None:
        if not self.state_path.is_file():
            return None
        return RunState(**json.loads(self.state_path.read_text(encoding="utf-8")))

    def commit(self, state: RunState, event: dict[str, Any] | None = None) -> None:
        state.updated_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        atomic_json(self.state_path, state.to_dict())
        if event is not None:
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()


__all__ = ["StateStore"]
