#!/usr/bin/env python3
"""Read-only P3R shadow inventory; never invokes TTS or writes game outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_pipeline.hashing import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--baseline", required=True); parser.add_argument("--out", required=True); args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")); p3r = baseline.get("p3r_reference", {})
    scenes = []
    for row in p3r.get("scenes", []):
        scenes.append({"scene": row.get("scene"), "classification": "INCONCLUSIVE", "reason": "V2 shadow not run; baseline is read-only", "legacy_report_sha256": row.get("sha256"), "legacy_release_blockers": row.get("release_blockers", [])})
    report = {"schema": "p3r-v2-shadow-v1", "status": "NOT_RUN", "production_mutations": 0, "scene_count": len(scenes), "line_count": p3r.get("summary", {}).get("required"), "classifications": {"INCONCLUSIVE": len(scenes)}, "scenes": scenes, "next_gate": "run adapter microset in disposable runtime_clone with OmniVoice/ASR explicitly released"}
    atomic_json(args.out, report); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
