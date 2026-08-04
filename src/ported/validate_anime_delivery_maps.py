#!/usr/bin/env python3
"""Hard validation for derived P3R anime delivery maps."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
SOURCE = ROOT / "maps_delivery_aligned_v3_codex2"
DERIVED = SOURCE
LEGACY = ROOT.parent / "anime_all_20260725"


def lexical(text: str) -> bool:
    return bool(re.search(r"[A-Za-zÄÖÜäöüß0-9]", str(text or "")))
OUT = ROOT / "ANIME_DELIVERY_MAP_VALIDATION.json"


def main() -> int:
    errors: list[dict] = []
    warnings: list[dict] = []
    source_ids: set[str] = set()
    derived_ids: set[str] = set()
    legacy_ids: set[str] = set()
    synth_count = keep_count = 0
    for legacy_path in sorted(LEGACY.glob("*_map.json")):
        legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_ids.update(
            line["id"] for line in legacy_payload.get("lines", [])
            if lexical(line.get("source_text", ""))
            and lexical(line.get("target_text", ""))
            and float(line.get("end", 0.0)) > float(line.get("start", 0.0))
        )
    for source_path in sorted(SOURCE.glob("*_map.json")):
        original = json.loads(source_path.read_text(encoding="utf-8"))
        derived_path = DERIVED / source_path.name
        if not derived_path.exists():
            errors.append({"scene": original["scene"], "error": "missing_derived_map"})
            continue
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        source_ids.update(line["id"] for line in original["lines"])
        for line in derived["lines"]:
            derived_ids.update(line.get("source_card_ids", [line["id"]]))
        # Reconciliation may deliberately collapse a legacy ASR duplicate
        # into an already mounted visual-card line. Count that alias as
        # covered without creating a second generated delivery.
        derived_ids.update(derived.get("deduplicated_ids", {}).keys())
        info = sf.info(derived["source_stem"])
        duration = info.frames / info.samplerate
        synth = []
        for line in derived["lines"]:
            start, end = float(line["start"]), float(line["end"])
            if not (0.0 <= start < end <= duration + 1e-6):
                errors.append({"line_id": line["id"], "error": "invalid_window", "start": start, "end": end, "duration": duration})
            # Delivery/mount edges and the English prompt edges are different
            # contracts.  The prompt may be wider (phoneme-safe tail) or have
            # a later onset to exclude an unscripted preceding effort, but it
            # must always be a valid source interval and contain the verified
            # speech tail.  The producer uses these fields explicitly when
            # preparing OmniVoice references.
            if "reference_start" in line or "reference_end" in line:
                ref_start = float(line.get("reference_start", start))
                ref_end = float(line.get("reference_end", end))
                speech_end = float(line.get("speech_end", end))
                if not (0.0 <= ref_start < ref_end <= duration + 1e-6):
                    errors.append({
                        "line_id": line["id"], "error": "invalid_reference_window",
                        "reference_start": ref_start, "reference_end": ref_end,
                        "duration": duration,
                    })
                if speech_end > ref_end + 1e-6:
                    errors.append({
                        "line_id": line["id"], "error": "reference_cuts_speech_tail",
                        "speech_end": speech_end, "reference_end": ref_end,
                    })
            if not str(line.get("target_text", "")).strip():
                errors.append({"line_id": line["id"], "error": "empty_target_text"})
            if line.get("timing_review_required"):
                errors.append({"line_id": line["id"], "error": "timing_review_still_required"})
            if line.get("force_keep_original"):
                keep_count += 1
                if line.get("force_clone"):
                    errors.append({"line_id": line["id"], "error": "keep_and_clone_conflict"})
            else:
                synth_count += 1
                synth.append(line)
                if (
                    not line.get("force_clone")
                    or line.get("mapping_validation")
                    not in {
                        "EXACT", "CONTEXTUAL", "HUMAN_CONFIRMED",
                        "LEGACY_ASR_RECOVERED",
                    }
                ):
                    errors.append({"line_id": line["id"], "error": "synth_without_exact_evidence"})
        synth.sort(key=lambda item: float(item["start"]))
        for left, right in zip(synth, synth[1:]):
            overlap = float(left["end"]) - float(right["start"])
            if overlap > 0.02:
                errors.append({
                    "scene": derived["scene"], "error": "synth_delivery_overlap",
                    "left": left["id"], "right": right["id"], "overlap_seconds": overlap,
                })
    missing = legacy_ids - derived_ids
    extra = derived_ids - legacy_ids
    if missing:
        errors.append({"error": "legacy_lexical_coverage_missing", "missing": sorted(missing), "extra": sorted(extra)})
    if extra:
        # Visual-card maps can contain valid cards absent from the legacy ASR
        # inventory (for example duplicated/cinematic cards). They are not a
        # coverage failure; retain them for identity and report them for audit.
        warnings.append({"error": "visual_card_ids_not_in_legacy_inventory", "ids": sorted(extra)})
    protected = {
        "100_090_M_L007": "synth",
        "300_010_M_L003": "synth",
        "320_140_M_L007": "synth",
        "300_060_M_L019": "keep",
        "210_210_M_L018": "keep",
    }
    derived_lookup = {}
    for path in DERIVED.glob("*_map.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        derived_lookup.update({line["id"]: line for line in payload["lines"]})
    for line_id, expected in protected.items():
        line = derived_lookup.get(line_id)
        actual = "keep" if line and line.get("force_keep_original") else "synth"
        if line is None or actual != expected:
            errors.append({"line_id": line_id, "error": "protected_decision_changed", "expected": expected, "actual": actual})
    payload = {
        "schema": "p3r_anime_delivery_map_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_safe_for_generation": not errors,
        "source_cards": len(source_ids), "derived_deliveries": len(derived_ids),
        "synthesize": synth_count, "keep_original": keep_count,
        "intentionally_removed_duplicate_cards": [],
        "errors": errors, "warnings": warnings,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
