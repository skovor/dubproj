#!/usr/bin/env python3
"""Run the dependency-light release gates and emit machine-readable results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str], cwd: Path) -> dict:
    started = time.perf_counter(); completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": completed.returncode, "wall_seconds": time.perf_counter() - started, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-pytest", action="store_true", help="Do not duplicate a pytest run already executed by the caller")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = [run([sys.executable, "-m", "compileall", "-q", "."], root)]
    if not args.skip_pytest:
        checks.append(run([sys.executable, "-m", "pytest", "-q"], root))
    checks.extend([
        run([sys.executable, "tests/run_smoke.py"], root),
        run([sys.executable, "tests/run_v2.py"], root),
        run([sys.executable, "scripts/check_port.py"], root),
        run([sys.executable, "scripts/validate_instructions.py"], root),
    ])
    report = {"schema": "v2-release-check-v1", "status": "PASS" if all(item["returncode"] == 0 for item in checks) else "FAIL", "checks": checks}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
