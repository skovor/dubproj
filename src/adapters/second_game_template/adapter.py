from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SecondGameAdapter:
    game_id: str
    independent_adapter: bool = True
    def validate(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if not self.game_id.strip(): raise ValueError("game_id required")
        return {"valid": bool(manifest.get("scenes")),"game_id":self.game_id,"independent_adapter":self.independent_adapter,"reason":"template_requires_project_specific_container_and_timing_checks"}

