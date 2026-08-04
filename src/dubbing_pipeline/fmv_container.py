"""Generic container adapter boundary.

The PCM montage is game-independent. USM/ADX, BIK, WebM, Wwise and custom
containers are deliberately adapters: they must prove stream counts, byte
layout and round-trip behaviour for the target game before deployment.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .hashing import sha256_file


@dataclass(frozen=True)
class ContainerValidation:
    bytes: int
    sha256: str
    signature: str
    diagnostic: str = ""


class ContainerAdapter(Protocol):
    def rebuild(self, original: Path, dialogue_wav: Path, output: Path, *, track: int = 1) -> None: ...
    def validate(self, path: Path) -> ContainerValidation: ...


class ExternalCommandAdapter:
    """Adapter for a project-specific rebuild command.

    The command receives `{original}`, `{dialogue}`, `{output}` and `{track}`.
    It is used for formats whose parser is not part of this generic package.
    """

    def __init__(self, rebuild_command: list[str], validate_command: list[str] | None = None):
        self.rebuild_command = rebuild_command; self.validate_command = validate_command

    def rebuild(self, original: Path, dialogue_wav: Path, output: Path, *, track: int = 1) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [item.format(original=str(original), dialogue=str(dialogue_wav), output=str(output), track=track) for item in self.rebuild_command]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode or not output.is_file():
            raise RuntimeError(f"container rebuild failed: {completed.stderr[-1000:]}")

    def validate(self, path: Path) -> ContainerValidation:
        diagnostic = ""
        if self.validate_command:
            command = [item.format(path=str(path)) for item in self.validate_command]
            completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
            diagnostic = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode:
                raise RuntimeError(f"container validation failed: {diagnostic[-1000:]}")
        return ContainerValidation(path.stat().st_size, sha256_file(path), path.read_bytes()[:4].decode("latin1", errors="replace"), diagnostic)


def validate_same_size(original: Path, rebuilt: Path) -> None:
    if original.stat().st_size != rebuilt.stat().st_size:
        raise ValueError(f"container size changed: {original.stat().st_size} -> {rebuilt.stat().st_size}")
