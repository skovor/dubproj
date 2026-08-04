"""Runtime isolation for the fresh OmniVoice 0.2.1 Persona project."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare() -> None:
    project = Path(__file__).resolve().parents[1]
    clean_root = project.parent
    candidates = [
        clean_root / "env" / "Lib" / "site-packages" / "torch" / "lib",
        Path(r"C:\Users\juand\Desktop\moddeutsch\ffmpeg7\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin"),
    ]
    current = os.environ.get("PATH", "")
    prefixes = [str(path) for path in candidates if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join(prefixes + [current])
    os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(os, "add_dll_directory"):
        for path in candidates:
            if path.is_dir():
                try:
                    os.add_dll_directory(str(path))
                except OSError:
                    pass
    workspace = project / "workspace"
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
