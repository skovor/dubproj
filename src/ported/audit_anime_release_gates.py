#!/usr/bin/env python3
"""Fail-closed audit for already-mounted P3R anime movie takes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    findings = []
    failures = Counter()
    reports = sorted((args.run / "anime").glob("*/FINAL_REPORT.json"))
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for line in report.get("lines", []):
            if line.get("action") == "KEEP_ORIGINAL" or line.get("pass"):
                continue
            failed = sorted(
                key for key, value in (line.get("hard_gates") or {}).items()
                if not value
            )
            failures.update(failed)
            findings.append({
                "scene": report.get("scene"),
                "line_id": line.get("id"),
                "winner": line.get("winner"),
                "round": line.get("round"),
                "failed_hard_gates": failed,
            })
    payload = {
        "reports": len(reports),
        "generated_lines_not_release_safe": len(findings),
        "failure_counts": dict(sorted(failures.items())),
        "findings": findings,
        "release_safe": not findings,
    }
    args.write.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "findings"}, ensure_ascii=False))
    return 0 if payload["release_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
