"""Filesystem guardrails for the V2 laboratory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .deploy_v2 import DeploymentError


@dataclass(frozen=True)
class SandboxLayout:
    root: Path
    repo: Path
    output_root: Path
    cache_root: Path
    staging_root: Path
    runtime_root: Path
    backups_root: Path
    quarantine_root: Path

    @classmethod
    def create(cls, root: str | Path) -> "SandboxLayout":
        base = Path(root).resolve()
        values = {"repo": base / "repo", "output_root": base / "runs", "cache_root": base / "cache_v2", "staging_root": base / "staging", "runtime_root": base / "runtime_clone", "backups_root": base / "backups", "quarantine_root": base / "quarantine"}
        for value in values.values():
            value.mkdir(parents=True, exist_ok=True)
        return cls(base, **values)

    def ensure_safe(self, *, lab_mode: bool = True) -> None:
        if not lab_mode:
            raise DeploymentError("V2 requires lab_mode=true until an explicit runtime release gate")
        for path in (self.output_root, self.cache_root, self.staging_root, self.runtime_root, self.backups_root, self.quarantine_root):
            resolved = path.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise DeploymentError(f"sandbox root escaped: {path}") from exc

    def contains(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False


__all__ = ["SandboxLayout"]
