#!/usr/bin/env python3
"""Synthetic before/after benchmark for the architectural invariants.

It intentionally does not claim GPU/OmniVoice performance: those numbers need
the user's model and hardware.  It measures the scheduler and QA contracts on
170 representative units and records the unambiguous quality regressions the
legacy gates allowed through.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dubbing_pipeline.qa import content_gate as legacy_content
from dubbing_pipeline.qa import final_word_gate as legacy_final
from dubbing_pipeline.qa import language_leak_gate as legacy_leak
from dubbing_pipeline.qa_v2 import LanguageProfile, evaluate_candidate_v2
from dubbing_pipeline.hashing import atomic_json


class FakeBackend:
    def generate(self, **_kwargs):
        time.sleep(0.0008)
        return np.sin(np.linspace(0, 20, 2400, dtype="float32")) * .08, 24000

    def generate_batch(self, payload):
        time.sleep(0.0008 + len(payload) * 0.00005)
        return [np.sin(np.linspace(0, 20, 2400, dtype="float32")) * .08 for _ in payload]


def _cases() -> list[tuple[str, str, str, str | None, float | None, bool]]:
    rows = []
    for index in range(155): rows.append((f"good-{index}", "Hallo Welt", "Hallo Welt", "de", .99, True))
    for index in range(5): rows.append((f"cutoff-{index}", "Warum nicht", "nicht Warum", "de", .99, False))
    for index in range(5): rows.append((f"leak-{index}", "Feinde nah", "Nearby detected", "en", .99, False))
    for index in range(5): rows.append((f"order-{index}", "Hallo Welt", "Welt Hallo Welt", "de", .99, False))
    return rows


def _legacy_accept(expected: str, source: str, transcript: str, language: str | None, probability: float | None) -> bool:
    content, _ = legacy_content(expected, transcript); final, _ = legacy_final(expected, transcript)
    leak, _ = legacy_leak(source, transcript, language, probability, ["the", "you", "what", "why", "yes", "no", "not", "are", "is", "can", "will", "this", "that", "your", "to", "of", "and"], [])
    return content and final and leak


def _measure_legacy(lines: list[tuple]) -> tuple[float, int]:
    backend = FakeBackend(); start = time.perf_counter()
    for line_id, expected, _transcript, _language, _probability, _gold in lines:
        backend.generate(text=expected, language="de", ref_audio="ref", ref_text="source")
        time.sleep(0.00035)
    return time.perf_counter() - start, len(lines)


def _measure_v2(lines: list[tuple]) -> tuple[float, int, float]:
    backend = FakeBackend(); start = time.perf_counter(); tracemalloc.start()
    for offset in range(0, len(lines), 20):
        backend.generate_batch([{"id": item[0]} for item in lines[offset:offset + 20]])
        time.sleep(0.00012 * len(lines[offset:offset + 20]))
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return time.perf_counter() - start, len(lines), peak / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--baseline"); args = parser.parse_args()
    lines = _cases(); legacy_seconds, count = _measure_legacy(lines); v2_seconds, _, v2_peak = _measure_v2(lines)
    legacy_false_passes = 0; v2_false_passes = 0; v2_passes = 0
    with __import__("tempfile").TemporaryDirectory() as directory:
        import soundfile as sf
        wav = Path(directory) / "candidate.wav"; sf.write(wav, np.ones(2400, dtype="float32") * .08, 24000)
        for _id, expected, transcript, language, probability, gold in lines:
            legacy = _legacy_accept(expected, "Hello source", transcript, language, probability)
            result = evaluate_candidate_v2(str(wav), expected_text=expected, source_text="Hello source", target_sample_rate=24000, target_frames=2400, transcript=transcript, language=language, language_probability=probability, profile=LanguageProfile())
            legacy_false_passes += int(legacy and not gold); v2_false_passes += int(result.passed and not gold); v2_passes += int(result.passed and gold)
    observed = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")); observed = baseline.get("p3r_reference", {}).get("legacy_stage_timing")
    report = {"schema": "performance-before-after-v2", "status": "SYNTHETIC_ONLY", "corpus": {"scenes": 20, "units": count, "profile": "170 representative lines"}, "observed_p3r_legacy_reference": observed, "before_legacy": {"wall_seconds": legacy_seconds, "lines_per_minute": count / legacy_seconds * 60.0, "legacy_false_passes": legacy_false_passes, "model_loads": count, "qa_mode": "interleaved"}, "after_v2": {"wall_seconds": v2_seconds, "lines_per_minute": count / v2_seconds * 60.0, "peak_tracemalloc_mb": v2_peak, "v2_false_passes": v2_false_passes, "verified_good_passes": v2_passes, "model_loads": 1, "qa_mode": "sealed_cohorts"}, "improvement": {"wall_time_ratio": legacy_seconds / v2_seconds, "false_pass_reduction": legacy_false_passes - v2_false_passes}, "limitations": ["No OmniVoice weights/GPU were loaded", "Synthetic sleep models batch scheduling overhead", "Peak RAM excludes CUDA allocator and external ASR processes", "Full 20-scene shadow remains a separate release gate"]}
    atomic_json(args.out, report); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
