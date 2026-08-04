#!/usr/bin/env python3
"""Find subtitle-card windows that end while dry dialogue remains active.

This is a review queue, not an automatic map editor: energy establishes that a
card end is suspicious, while the official text/speaker context must still
authorize any timing extension.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent / "P3R_ANIME_VISUAL_DUB_20260801"
OUT = ROOT / "ANIME_CARD_TAIL_WINDOW_AUDIT.json"
HOP_SECONDS = 0.01
MAX_EXTENSION_SECONDS = 2.0
START_GRACE_SECONDS = 0.12
MIN_EXTENSION_SECONDS = 0.15


def runs(indices: np.ndarray) -> list[np.ndarray]:
    if not len(indices):
        return []
    return [part for part in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(part)]


def main() -> int:
    flagged: list[dict] = []
    scanned = 0
    for map_path in sorted((ROOT / "maps").glob("*_map.json")):
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        source, sr = sf.read(mapping["source_stem"], dtype="float32", always_2d=False)
        if source.ndim > 1:
            source = source.mean(axis=1)
        lines = sorted(mapping["lines"], key=lambda row: float(row["start"]))
        hop = max(1, round(HOP_SECONDS * sr))
        for index, line in enumerate(lines):
            scanned += 1
            end = float(line["end"])
            next_start = float(lines[index + 1]["start"]) if index + 1 < len(lines) else len(source) / sr
            stop = min(end + MAX_EXTENSION_SECONDS, next_start)
            if stop - end < MIN_EXTENSION_SECONDS:
                continue
            probe_start = max(0, int((float(line["start"]) - 0.15) * sr))
            probe_end = min(len(source), int(stop * sr))
            envelope = np.array([
                np.sqrt(np.mean(source[pos:pos + hop] ** 2))
                for pos in range(probe_start, probe_end, hop)
            ])
            if not len(envelope):
                continue
            peak = float(envelope.max(initial=0.0))
            threshold = max(0.001, peak * 10 ** (-35.0 / 20.0))
            active = np.flatnonzero(envelope >= threshold)
            for group in runs(active):
                group_start = (probe_start + int(group[0]) * hop) / sr
                group_end = (probe_start + int(group[-1] + 1) * hop) / sr
                if (
                    group_start <= end + START_GRACE_SECONDS
                    and group_end >= end + MIN_EXTENSION_SECONDS
                    and len(group) >= 4
                ):
                    flagged.append({
                        "scene": mapping["scene"],
                        "line_id": line["id"],
                        "card_start": float(line["start"]),
                        "card_end": end,
                        "next_card_start": next_start,
                        "active_tail_start": round(group_start, 3),
                        "active_tail_end": round(group_end, 3),
                        "suggested_review_end": round(min(group_end + 0.03, next_start - 0.02), 3),
                        "text_en": line.get("source_text", ""),
                        "reason": "dry_stem_activity_continues_after_card_end",
                    })
                    break
    payload = {
        "method": "dry-stem energy screen; review required before map edit",
        "scanned_cards": scanned,
        "suspect_card_endings": len(flagged),
        "items": flagged,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scanned": scanned, "suspects": len(flagged), "report": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
