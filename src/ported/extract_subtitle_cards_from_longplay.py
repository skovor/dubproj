"""Recover subtitle-card timing from a burned-subtitle FMV compilation.

This is an evidence producer, never a generator and never a map mutator.
It treats the visible subtitle card as the authority.  ASR is deliberately
absent: audio is used elsewhere only to locate each USM in the compilation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
from rapidfuzz.fuzz import ratio

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
VIDEO = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\P3R_all_anime_cutscenes_bGdTAZ1hA4s.mp4")
ALIGN = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\scene_video_alignment.json")
INVENTORY = ROOT / "anime_all_20260725" / "anime_dialogue_inventory.json"
OUT = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\visible_subtitle_cards_evidence.json")
FRAMES = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\subtitle_card_frames")

SAMPLE_HZ = 4.0
MIN_CARD_SECONDS = 0.35
MIN_ALIGN_SCORE = 0.78
# A burned subtitle can change its glyphs without changing its placement,
# outline, or overall mask enough to cross the old 0.095 threshold.  Sample
# OCR at the same cadence as the visual scan and require two consecutive
# observations before splitting a card.  This keeps fade-in/out noise merged
# while preventing two consecutive captions from becoming one production
# window (the 100_090_M L008/L009 regression).
# rapidfuzz.fuzz.ratio returns a percentage in the 0..100 range.
OCR_CHANGE_RATIO = 82.0
OCR_CONFIRMATION_SAMPLES = 2
CARD_TAIL_MERGE_SECONDS = 0.50


def normal(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def card_mask(frame: np.ndarray) -> np.ndarray:
    """Keep white glyphs surrounded by their dark subtitle outline.

    The mask avoids treating bright scenery as a caption. The exact text is
    read only after a stable card interval is identified.
    """
    h, w = frame.shape[:2]
    crop = frame[int(h * .72): int(h * .975), int(w * .08):int(w * .92)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white = (g >= 185).astype(np.uint8)
    dark = (g <= 80).astype(np.uint8)
    outlined = white & cv2.dilate(dark, np.ones((3, 3), np.uint8))
    return cv2.resize(outlined, (160, 64), interpolation=cv2.INTER_NEAREST)


def is_caption(mask: np.ndarray) -> bool:
    # At 640x360 P3R captions generate hundreds to thousands of outlined
    # pixels. The small lower bound intentionally prefers false candidates;
    # OCR + BMD matching is the second gate.
    # Very short cards such as "Huh?" still matter: the user asked to dub
    # every displayed subtitle, not only long dialogue. OCR/BMD matching
    # rejects scenery false positives later in the pipeline.
    return int(mask.sum()) >= 3


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def ocr_card(reader: easyocr.Reader, frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    crop = frame[int(h * .70):int(h * .985), int(w * .05):int(w * .95)]
    lines = reader.readtext(crop, detail=0, paragraph=True, text_threshold=0.45,
                            low_text=0.25, link_threshold=0.25)
    return " ".join(lines).strip()


def scan_scene(cap: cv2.VideoCapture, reader: easyocr.Reader, scene: dict,
               official: list[dict]) -> list[dict]:
    start, duration = scene["reference_start_s"], scene["source_duration_s"]
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    stride = max(1, round(fps / SAMPLE_HZ))
    end_frame = int((start + duration) * fps)
    current: dict | None = None
    rows: list[dict] = []
    pending_ocr = ""
    pending_ocr_count = 0
    fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while fno < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        fno += 1
        if fno % stride:
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        mask = card_mask(frame)
        present = is_caption(mask)
        if not present:
            if current and t - current["start_s"] >= MIN_CARD_SECONDS:
                current["end_s"] = t
                rows.append(current)
            current = None
            pending_ocr = ""
            pending_ocr_count = 0
            continue
        # OCR each 250 ms sample while a card is present.  The previous
        # implementation only OCR'd the first frame *after* interval merging,
        # so text changes with identical placement were invisible to the
        # authority builder.
        sample_text = ocr_card(reader, frame)
        sample_norm = normal(sample_text)
        if current is None:
            current = {"start_s": t, "frame": frame.copy(), "mask": mask,
                       "ocr_text": sample_text}
            pending_ocr = ""
            pending_ocr_count = 0
            continue
        current_norm = normal(current.get("ocr_text", ""))
        if sample_norm and current_norm and ratio(sample_norm, current_norm) < OCR_CHANGE_RATIO:
            if sample_norm == pending_ocr:
                pending_ocr_count += 1
            else:
                pending_ocr = sample_norm
                pending_ocr_count = 1
            if (pending_ocr_count >= OCR_CONFIRMATION_SAMPLES
                    and t - current["start_s"] >= MIN_CARD_SECONDS):
                # The first confirmed change is observed one sample after the
                # actual boundary.  Put the edge halfway through that delay
                # so neither card receives the full next caption.
                boundary = max(current["start_s"] + MIN_CARD_SECONDS,
                                t - 0.5 / SAMPLE_HZ)
                current["end_s"] = boundary
                rows.append(current)
                current = {"start_s": boundary, "frame": frame.copy(),
                           "mask": mask, "ocr_text": sample_text}
                pending_ocr = ""
                pending_ocr_count = 0
                continue
        elif sample_norm and not current_norm:
            current["ocr_text"] = sample_text
            pending_ocr = ""
            pending_ocr_count = 0
        else:
            pending_ocr = ""
            pending_ocr_count = 0
        # A card must remain different for at least one sample before it
        # becomes a new event; this prevents subtitle fade-in/out duplicates.
        if distance(current["mask"], mask) > 0.095 and t - current["start_s"] >= MIN_CARD_SECONDS:
            current["end_s"] = t
            rows.append(current)
            current = {"start_s": t, "frame": frame.copy(), "mask": mask}
    if current and start + duration - current["start_s"] >= MIN_CARD_SECONDS:
        current["end_s"] = start + duration
        rows.append(current)

    out: list[dict] = []
    for idx, row in enumerate(rows, 1):
        text = ocr_card(reader, row["frame"])
        text_n = normal(text)
        candidates = sorted(
            ((ratio(text_n, normal(x["source_text"])), x) for x in official if text_n),
            key=lambda pair: pair[0], reverse=True,
        )[:3]
        best_score, best = candidates[0] if candidates else (0, None)
        status = "VISUAL_EXACT" if best_score >= 88 else "VISUAL_CONTEXT_REQUIRED"
        image = FRAMES / f"{scene['scene']}_{idx:03d}.png"
        cv2.imwrite(str(image), row["frame"])
        item = {
            "scene": scene["scene"],
            "card_index": idx,
            "start_s_in_video": round(row["start_s"], 3),
            "end_s_in_video": round(row["end_s"], 3),
            "start_s_in_scene": round(row["start_s"] - start, 3),
            "end_s_in_scene": round(row["end_s"] - start, 3),
            "ocr_text": text,
            "best_bmd_id": best["id"] if best else None,
            "best_bmd_text": best["source_text"] if best else None,
            "similarity": round(best_score, 1),
            "alternatives": [{"id": x["id"], "score": round(s, 1)} for s, x in candidates[1:]],
            "status": status,
            "frame": str(image),
        }
        # OCR can briefly return a fragmented/garbled second reading while a
        # single caption fades out.  If that fragment is immediately adjacent,
        # shorter than half a second, and resolves to the same official BMD
        # line, fold it into the preceding card.  Real consecutive cues (for
        # example the two distinct "Morning!" IDs) retain separate identities.
        duration = row["end_s"] - row["start_s"]
        same_id = bool(item["best_bmd_id"]
                       and item["best_bmd_id"] == out[-1].get("best_bmd_id")) if out else False
        contiguous_duplicate = bool(
            same_id and abs(float(item["start_s_in_scene"])
                            - float(out[-1]["end_s_in_scene"])) <= 0.05
        )
        if out and same_id and (duration < CARD_TAIL_MERGE_SECONDS or contiguous_duplicate):
            out[-1]["end_s_in_video"] = item["end_s_in_video"]
            out[-1]["end_s_in_scene"] = item["end_s_in_scene"]
            continue
        out.append(item)
    return out


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    align = {r["scene"]: r for r in json.loads(ALIGN.read_text(encoding="utf-8"))["rows"]}
    inventory = {r["scene"]: r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))}
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    cap = cv2.VideoCapture(str(VIDEO))
    output = {"method": "visible cards + OCR + official BMD fuzzy-match; no ASR", "scenes": []}
    for name, item in inventory.items():
        a = align[name]
        if a["score"] < MIN_ALIGN_SCORE:
            output["scenes"].append({"scene": name, "status": "ALIGNMENT_REQUIRED", "alignment": a, "cards": []})
            print(f"{name}: alignment required ({a['score']})", flush=True)
            continue
        cards = scan_scene(cap, reader, a, item["official_lines"])
        output["scenes"].append({"scene": name, "status": "CARD_EVIDENCE_READY", "alignment": a, "cards": cards})
        print(f"{name}: {len(cards)} visible card candidates", flush=True)
    cap.release()
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
