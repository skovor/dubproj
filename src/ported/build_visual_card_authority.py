"""Produce immutable, per-card FMV timing evidence from the subtitled source.

This does not infer a card for every BMD entry: the output contains only text
actually visible in the reference video.  Repeated wording is retained as
ambiguous unless its local sequence can be established independently later.
"""
from __future__ import annotations
import json
from pathlib import Path

DATA = Path(r"E:\P3R_FMV_SUBTITLE_TIMING")
SRC = DATA / "visible_subtitle_cards_evidence.json"
OUT = DATA / "ANIME_VISUAL_CARD_AUTHORITY.json"

def main() -> None:
    source = json.loads(SRC.read_text(encoding="utf-8"))
    scenes = []
    totals = {"approved": 0, "context_only": 0, "rejected_noise": 0}
    for scene in source["scenes"]:
        cards = []
        for card in scene.get("cards", []):
            text = (card.get("ocr_text") or "").strip()
            score = float(card.get("similarity") or 0)
            alternatives = card.get("alternatives") or []
            margin = score - float(alternatives[0]["score"]) if alternatives else score
            if not text:
                totals["rejected_noise"] += 1
                continue
            # OCR may mangle short cards.  It is still a legitimate visible
            # card, but cannot independently unlock audio replacement.
            verdict = "APPROVED_VISUAL_TEXT" if score >= 90 and margin >= 4 else "CONTEXT_REQUIRED"
            totals["approved" if verdict.startswith("APPROVED") else "context_only"] += 1
            cards.append({
                "card_index": card["card_index"],
                "start_s_in_scene": card["start_s_in_scene"],
                "end_s_in_scene": card["end_s_in_scene"],
                "visible_ocr": text,
                "candidate_bmd_id": card.get("best_bmd_id"),
                "candidate_official_text": card.get("best_bmd_text"),
                "similarity": score,
                "margin": round(margin, 1),
                "verdict": verdict,
                "evidence": "burned official subtitle card; no ASR/audio-energy inference",
            })
        scenes.append({"scene": scene["scene"], "source_status": scene["status"], "cards": cards})
    OUT.write_text(json.dumps({"source": str(SRC), "totals": totals, "scenes": scenes}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT, totals)

if __name__ == "__main__":
    main()
