#!/usr/bin/env python3
"""Dependency-light integrity check for the GitHub source bundle."""
from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    if not compileall.compile_dir(str(ROOT / "dubbing_pipeline"), quiet=1):
        return 1
    manifest = json.loads((ROOT / "PORT_MANIFEST.json").read_text(encoding="utf-8"))
    ported = ROOT / "ported"
    listed = {name for group in manifest["ported_modules"].values() for name in group}
    actual = {path.name for path in ported.glob("*.py")}
    if listed != actual:
        print(f"ported manifest mismatch: listed-only={sorted(listed - actual)} actual-only={sorted(actual - listed)}", file=sys.stderr)
        return 2
    # The generic runtime must not have a project-name/path dependency.
    forbidden = ("P3R", "p3r", "Reloaded-II", "Fukuro", "Xrd777")
    offenders = []
    for path in (ROOT / "dubbing_pipeline").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path))
    if offenders:
        print("project-specific token in generic modules:", offenders, file=sys.stderr)
        return 3
    print(f"generic source check: PASS ({len(actual)} ported modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
