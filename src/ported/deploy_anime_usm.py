#!/usr/bin/env python3
"""Atomically deploy a validated P3R anime USM to Reloaded-II."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_MOD_ROOT = Path(
    r"C:\Users\juand\Desktop\Reloaded-II\Mods\p3r.doblaje.aleman"
    r"\UnrealEssentials\P3R\Content\Xrd777\CriData\Stream"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usm", required=True, type=Path)
    parser.add_argument("--game-name", required=True)
    parser.add_argument("--vgmstream", required=True, type=Path)
    parser.add_argument("--mod-root", type=Path, default=DEFAULT_MOD_ROOT)
    args = parser.parse_args()

    if not args.usm.is_file() or args.usm.stat().st_size < 1_000_000:
        raise ValueError(f"refusing invalid/small USM: {args.usm}")
    with args.usm.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"CRID":
        raise ValueError(f"refusing non-CRI container: {args.usm}")
    if not args.game_name.endswith(".usm") or Path(args.game_name).name != args.game_name:
        raise ValueError(f"invalid game filename: {args.game_name}")
    if not args.vgmstream.is_file():
        raise FileNotFoundError(args.vgmstream)

    probe = subprocess.run(
        [str(args.vgmstream), "-s", "2", "-m", str(args.usm)],
        capture_output=True,
    )
    metadata = (probe.stdout + probe.stderr).decode("utf-8", errors="replace")
    required = ("stream count: 2", "channels: 6", "encoding: CRI ADX")
    if probe.returncode or not all(token in metadata for token in required):
        raise ValueError(f"USM validation failed:\n{metadata[-2000:]}")

    source_hash = digest(args.usm)
    destinations = [
        args.mod_root / "Movie_VP9" / "Anim" / args.game_name,
        args.mod_root / "de" / "Movie_VP9" / "Anim" / args.game_name,
        args.mod_root / "en" / "Movie_VP9" / "Anim" / args.game_name,
    ]
    backup_root = (
        args.mod_root.parent.parent.parent
        / "_backup_before_anime_recovery"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size:
            relative = destination.relative_to(args.mod_root)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)

        staging = destination.with_suffix(destination.suffix + ".deploying")
        shutil.copy2(args.usm, staging)
        if staging.stat().st_size != args.usm.stat().st_size or digest(staging) != source_hash:
            raise ValueError(f"staging verification failed: {staging}")
        os.replace(staging, destination)
        print(f"DEPLOYED {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
