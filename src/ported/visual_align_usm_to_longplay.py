"""Visual USM-to-longplay alignment for subtitle timing recovery.

The reference longplay may use a different spoken language.  This script thus
uses only picture content above the subtitle band: it demuxes the original USM
video, finds several independent visual anchors in the longplay, and accepts a
scene start only when anchors agree.  It changes no map and generates no TTS.
"""
from __future__ import annotations

import json, subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
INV = ROOT / "anime_all_20260725" / "anime_dialogue_inventory.json"
LONG = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\P3R_full_no_commentary_179A4o2sIxk.mp4")
WORK = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\visual_alignment")
OUT = Path(r"E:\P3R_FMV_SUBTITLE_TIMING\full_longplay_visual_alignment.json")
WANNACRI = ROOT / "tools_cricodecs_env" / "Scripts" / "wannacri.exe"

SAMPLE_HZ = 1.0
WIDTH, HEIGHT = 64, 36
CROP_HEIGHT = 31  # discard burned subtitle band


def feature(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:CROP_HEIGHT].astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray -= gray.mean()
    gray /= gray.std() + 1e-5
    return gray.reshape(-1)


def video_index(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps / SAMPLE_HZ)); feats=[]; times=[]; n=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if n % stride == 0:
            feats.append(feature(frame)); times.append(n / fps)
        n += 1
    cap.release()
    return np.vstack(feats).astype(np.float32), np.array(times, dtype=np.float32)


def demux_usm(item: dict) -> Path:
    slug = item["scene"]
    target = WORK / "usm_demux" / slug
    existing = list(target.rglob("*.ivf")) if target.exists() else []
    if not existing:
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(WANNACRI), "extractusm", item["source_usm"], "-o", str(target)], check=True)
        existing = list(target.rglob("*.ivf"))
    if not existing: raise RuntimeError(f"no IVF for {slug}")
    return existing[0]


def anchors(ivf: Path, ref: np.ndarray, ref_times: np.ndarray) -> list[tuple[float,float]]:
    cap=cv2.VideoCapture(str(ivf)); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    # Avoid opening/closing fades and use multiple independent shots.
    probes=[duration*x for x in (.17,.31,.49,.67,.83)]
    found=[]
    for t in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)); ok, frame=cap.read()
        if not ok: continue
        q=feature(frame); scores=(ref @ q)/len(q)
        k=min(12,len(scores)); idx=np.argpartition(scores,-k)[-k:]
        for i in idx:
            found.append((float(ref_times[i]-t),float(scores[i])))
    cap.release(); return found


def resolve(candidates: list[tuple[float,float]]) -> tuple[float,int,float,float]:
    # Weighted one-second bins. A genuine film contributes several anchors to
    # one bin; coincidental similar shots do not.
    bins=defaultdict(list)
    for start,score in candidates: bins[round(start)].append((start,score))
    ranked=sorted(bins.values(),key=lambda xs:(len(xs),sum(x[1] for x in xs)),reverse=True)
    win=ranked[0]; weights=np.array([max(0,x[1]) for x in win]); values=np.array([x[0] for x in win])
    start=float(np.average(values,weights=weights)) if weights.sum() else float(values.mean())
    return start,len(win),float(np.mean(weights)),float(np.std(values))


def main() -> None:
    WORK.mkdir(parents=True,exist_ok=True)
    print("indexing longplay frames",flush=True)
    ref,times=video_index(LONG)
    np.save(WORK/"longplay_visual_features.npy",ref)
    np.save(WORK/"longplay_visual_times.npy",times)
    items=json.loads(INV.read_text(encoding="utf-8")); rows=[]
    for item in items:
        ivf=demux_usm(item); cand=anchors(ivf,ref,times); start,support,score,spread=resolve(cand)
        status="VISUAL_ALIGNED" if support>=2 and score>=0.55 and spread<=1.5 else "VISUAL_REVIEW"
        rows.append({"scene":item["scene"],"reference_start_s":round(start,3),"support":support,"mean_similarity":round(score,4),"spread_s":round(spread,3),"status":status})
        print(rows[-1],flush=True)
    OUT.write_text(json.dumps({"method":"visual USM anchors; subtitle band excluded; no ASR","rows":rows},indent=2),encoding="utf-8")
    print(f"Wrote {OUT}",flush=True)

if __name__=="__main__": main()
