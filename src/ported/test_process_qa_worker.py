#!/usr/bin/env python3
"""Non-destructive smoke test for the process-based cinematic QA worker."""
from __future__ import annotations

import re
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
PERSONA = ROOT.parent / "OmniVoice-clean-0.2.1" / "persona_project"
pytestmark = pytest.mark.historical
if os.environ.get("DUBPROJ_RUN_HISTORICAL") != "1" or not (PERSONA / "scripts" / "generate_p3r_cinematic_assets_v021.py").is_file():
    pytest.skip("historical OmniVoice/P3R assets are opt-in and unavailable", allow_module_level=True)

import generate_p3r_cinematic_assets_v021 as base
import generate_p3r_cinematic_assets_v021_rounds as rounds


def main() -> int:
    raw_root = base.RUN_ROOT / "raw_round_candidates"
    smoke_root = base.RUN_ROOT / "_qa_process_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    by_token = {
        rounds.candidate_token(row["physical_key"]): row
        for row in base.normalize_rows()
    }
    payloads = []
    for path in sorted(raw_root.glob("*.npy")):
        match = re.fullmatch(r"([0-9a-f]{20})_a(\d{2})\.npy", path.name)
        if not match or match.group(1) not in by_token:
            continue
        row = by_token[match.group(1)]
        payloads.append(
            {
                "row": row,
                "attempt_number": int(match.group(2)),
                "raw_path": str(path),
                "fitted_path": str(smoke_root / path.name),
                "qa_tmp_root": str(smoke_root / "tmp"),
                "source_context": None,
            }
        )
        if len(payloads) == 2:
            break
    if len(payloads) != 2:
        raise RuntimeError("need two recoverable raw candidates")
    with ProcessPoolExecutor(
        max_workers=2, mp_context=get_context("spawn")
    ) as executor:
        results = list(executor.map(rounds.evaluate_candidate_process, payloads))
    for result in results:
        fitted = Path(result["fitted_path"])
        if not fitted.is_file() or "qa" not in result:
            raise AssertionError(result)
        fitted.unlink()
    print("process_qa_worker_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
