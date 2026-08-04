#!/usr/bin/env python3
"""Compile release-safe derived anime delivery maps without touching sources."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from rebuild_anime_delivery_windows_mms import refine_edges, words


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
OUT_MAPS = ROOT / "maps_delivery_aligned_v2"
REPORT = ROOT / "ANIME_DELIVERY_MAP_FINALIZATION.json"
OVERRIDES = ROOT / "ANIME_HUMAN_CONTEXT_OVERRIDES.json"

NONVERBAL = {
    "ah", "aha", "ahahaha", "eh", "hehe", "heheh", "hm", "hmm", "huh",
    "oh", "ooh", "ugh", "uh", "umm", "wha", "whoa", "wow",
}


def by_id(path: Path, accepted_status: str | None = None) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if accepted_status:
        rows = [row for row in rows if row.get("status") == accepted_status]
    return {row["line_id"]: row for row in rows}


def best_asr_pair(matches: dict[str, dict], token_count: int) -> tuple[list[dict], list[str]] | None:
    required = 0.50
    qualified = [name for name, match in matches.items() if match["coverage"] >= required]
    best = None
    for index, first_name in enumerate(qualified):
        for second_name in qualified[index + 1:]:
            first = {p["official_index"]: p for p in matches[first_name]["pairs"]}
            second = {p["official_index"]: p for p in matches[second_name]["pairs"]}
            shared = sorted(set(first) & set(second))
            if not shared:
                continue
            ds = abs(first[shared[0]]["start"] - second[shared[0]]["start"])
            de = abs(first[shared[-1]]["end"] - second[shared[-1]]["end"])
            score = max(ds, de)
            if score <= 0.75 and (best is None or score < best[0]):
                best = (score, first_name, second_name)
    if best is None:
        return None
    _, first_name, second_name = best
    return matches[first_name]["pairs"] + matches[second_name]["pairs"], [first_name, second_name]


def main() -> int:
    consensus = by_id(ROOT / "ANIME_TIMING_CONSENSUS.json")
    block = by_id(ROOT / "ANIME_SCENE_BLOCK_ALIGNMENT_PROPOSALS.json")
    first_rescue = by_id(ROOT / "ANIME_ASR_SEEDED_MMS_RESCUE.json", "RESCUED")
    card_rescue = by_id(ROOT / "ANIME_CARD_SEEDED_MMS_RESCUE.json", "RESCUED")
    card_all = by_id(ROOT / "ANIME_CARD_SEEDED_MMS_RESCUE.json")
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))["lines"]
    if OUT_MAPS.exists():
        shutil.rmtree(OUT_MAPS)
    OUT_MAPS.mkdir(parents=True)
    decisions: list[dict] = []
    source_cache: dict[str, tuple[np.ndarray, int]] = {}

    for map_path in sorted((ROOT / "maps").glob("*_map.json")):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        scene = mapping["scene"]
        derived_lines: list[dict] = []
        source_line_lookup = {line["id"]: line for line in mapping["lines"]}
        absorbed_ids: set[str] = set()
        for line in mapping["lines"]:
            line_id = line["id"]
            if line_id in absorbed_ids:
                continue
            tokens = words(line.get("source_text", ""))
            lexical = [token for token in tokens if token not in NONVERBAL]
            decision = ""
            evidence = ""
            timing = None

            override = overrides.get(line_id)
            if override and override["action"] == "GROUP_WITH":
                members = [source_line_lookup[item] for item in override["members"]]
                output = dict(members[0])
                output["source_card_ids"] = [item["id"] for item in members]
                output["source_text"] = " ".join(item["source_text"].strip() for item in members)
                output["target_text"] = " ".join(item["target_text"].strip() for item in members)
                output["start"] = float(override["start"])
                output["end"] = float(override["end"])
                output["force_clone"] = True
                output.pop("force_keep_original", None)
                output.pop("force_keep_reason", None)
                output["mapping_validation"] = "EXACT"
                output["mapping_validation_reason"] = f"human_context_group:{override['reason']}"
                output["timing_source"] = "MULTI_CARD_CONTINUOUS_DELIVERY_V2"
                output["timing_review_required"] = False
                derived_lines.append(output)
                absorbed_ids.update(override["members"][1:])
                decisions.append({
                    "scene": scene, "line_id": line_id, "decision": "SYNTHESIZE_GROUP",
                    "absorbs": override["members"], "evidence": override["reason"],
                    "new_start": output["start"], "new_end": output["end"],
                })
                for absorbed in override["members"][1:]:
                    decisions.append({
                        "scene": scene, "line_id": absorbed,
                        "decision": "ABSORBED_IN_CONTINUOUS_DELIVERY",
                        "absorbed_by": line_id, "evidence": override["reason"],
                    })
                continue
            if override and override["action"] == "DROP_DUPLICATE_CARD_NO_SEPARATE_DELIVERY":
                decisions.append({
                    "scene": scene, "line_id": line_id,
                    "decision": "DROP_DUPLICATE_CARD_NO_SEPARATE_DELIVERY",
                    "evidence": override["reason"],
                })
                continue

            if override and override["action"] == "SYNTHESIZE":
                timing = {"proposed_start": override["start"], "proposed_end": override["end"]}
                decision = "SYNTHESIZE"
                evidence = f"human_context_override:{override['reason']}"
            elif override and override["action"] == "KEEP_ORIGINAL":
                decision = "KEEP_ORIGINAL"
                evidence = f"human_context_override:{override['reason']}"
            elif line_id in card_rescue:
                timing = card_rescue[line_id]
                decision = "SYNTHESIZE"
                evidence = "card_seeded_dual_asr_plus_mms_plus_ac29"
            elif line_id in first_rescue:
                timing = first_rescue[line_id]
                decision = "SYNTHESIZE"
                evidence = "dual_asr_seeded_mms_plus_ac29"
            elif consensus[line_id]["status"] == "TIMING_EVIDENCE_PASS":
                timing = block[line_id]
                decision = "SYNTHESIZE"
                evidence = "official_plus_2of3_asr_plus_mms_plus_ac29"
            else:
                row = consensus[line_id]
                overlap = row.get("overlap_policy", "NONE")
                card = card_all.get(line_id, {})
                pair = best_asr_pair(card.get("matches", {}), len(tokens)) if card.get("matches") else None
                nonverbal_only = bool(tokens and not lexical)
                short_guard = bool(
                    len(tokens) > 3
                    or row["passes"]["vad_beam5"]["coverage"] >= 0.50
                    or float(row.get("mms_mean_score") or 0.0) >= 0.75
                )
                safe_overlap = overlap == "NONE"
                if pair and lexical and short_guard and safe_overlap:
                    pairs, pass_names = pair
                    if scene not in source_cache:
                        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
                        if source.ndim > 1:
                            source = source.mean(axis=1)
                        source_cache[scene] = (np.asarray(source, dtype=np.float32), sr)
                    source, sr = source_cache[scene]
                    anchor_start = min(float(item["start"]) for item in pairs)
                    anchor_end = max(float(item["end"]) for item in pairs)
                    context_start = max(0.0, anchor_start - 1.5)
                    context_end = min(len(source) / sr, anchor_end + 1.5)
                    clip = source[round(context_start * sr):round(context_end * sr)]
                    local_start, local_end, acoustic = refine_edges(
                        clip, sr, anchor_start - context_start, anchor_end - context_start
                    )
                    if acoustic["start_valley_found"] and acoustic["end_valley_found"]:
                        timing = {
                            "proposed_start": round(context_start + local_start, 3),
                            "proposed_end": round(context_start + local_end, 3),
                        }
                        decision = "SYNTHESIZE"
                        evidence = f"2of3_asr_plus_ac29_direct:{'+'.join(pass_names)}"
                if not decision:
                    decision = "KEEP_ORIGINAL"
                    if overlap != "NONE":
                        evidence = overlap.lower()
                    elif nonverbal_only or not lexical:
                        evidence = "nonverbal_or_effort"
                    elif not pair:
                        evidence = "fewer_than_two_agreeing_asr_anchors"
                    elif not short_guard:
                        evidence = "short_nonvad_hallucination_guard"
                    else:
                        evidence = "no_stable_ac29_boundary"

            output = dict(line)
            if decision == "SYNTHESIZE":
                output["start"] = float(timing["proposed_start"])
                output["end"] = float(timing["proposed_end"])
                output["force_clone"] = True
                output.pop("force_keep_original", None)
                output.pop("force_keep_reason", None)
                output["mapping_validation"] = "EXACT"
                output["mapping_validation_reason"] = evidence
                output["timing_source"] = "MULTI_EVIDENCE_DELIVERY_ALIGNMENT_V2"
                output["timing_review_required"] = False
            else:
                output["force_clone"] = False
                output["force_keep_original"] = True
                output["force_keep_reason"] = evidence
                output["mapping_validation"] = "CONTEXTUAL"
                output["mapping_validation_reason"] = evidence
                output["timing_review_required"] = False
            derived_lines.append(output)
            decisions.append({
                "scene": scene, "line_id": line_id, "decision": decision,
                "evidence": evidence, "source_text": line.get("source_text", ""),
                "old_start": float(line["start"]), "old_end": float(line["end"]),
                "new_start": float(output["start"]), "new_end": float(output["end"]),
            })
        mapping["lines"] = derived_lines
        mapping["notes"] = (
            "Delivery-aligned V2 map. Subtitle cards authorize content only; "
            "physical edges require multi-evidence timing consensus."
        )
        target = OUT_MAPS / map_path.name
        target.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = {
        "schema": "p3r_anime_delivery_map_finalization_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_maps_modified": False,
        "derived_maps": str(OUT_MAPS),
        "decisions": decisions,
        "counts": {
            name: sum(row["decision"] == name for row in decisions)
            for name in sorted({row["decision"] for row in decisions})
        },
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "counts": payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
