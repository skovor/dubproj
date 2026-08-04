#!/usr/bin/env python3
"""Decide scream vs mild interjection from the ORIGINAL AUDIO, not the spelling.

Text cannot separate these -- both are written with the same elongation:
    "Ahhh, du kommst wegen der Zusatzlektion?"   <- calm, almost teasing
    "Aaaah! Hiihiiii! Hilfe!"                     <- panic scream
The English reference audio can: a scream is loud, high-pitched and sustained.

For each candidate line with reference audio available, measures the leading
vocalisation and scores it. Also reports whether the line is spliceable at all
(the English must scream too) and flags German words that merely LOOK like
screams ("Neeein" vs English "Nooo" -- different words, splicing is invalid).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(
    0, r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project\scripts"
)
from clean_runtime import prepare

prepare()

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
WS = Path(
    r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project"
    r"\workspace\chronological_early_20260721"
)
DATA = ROOT / "corpus_screams.json"
OUT = ROOT / "screams_acoustic.json"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# German words that are real words, not language-neutral noise: never splice
# these from the English track, the English says something else.
GERMAN_WORDS = re.compile(
    r"^(nein|neiin|neein|neeein|neiiin|neeeein|neiiiin|hilfe|hiilfe|hiiilfe|"
    r"rindfleisch|los|was|falsch|ja|bitte|mama|papa|opa|oma|warte|halt)$", re.I)


def collapse(t: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", t)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def _score_from_array(y: np.ndarray, sr: int) -> dict | None:
    """Loudness/pitch of the LEADING vocalisation in y. Call with y reversed
    to score the TRAILING vocalisation instead (see tail_measure)."""
    y = np.asarray(y, float)
    if len(y) < sr // 10:
        return None
    peak = np.abs(y).max()
    if peak <= 0:
        return None
    hop = int(sr * 0.010)
    n = (len(y) // hop) * hop
    env = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(axis=1))
    edb = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    on = next((i for i, v in enumerate(edb) if v > -40), 0)
    # leading chunk = until first sustained drop, capped at 1.2 s
    end = len(edb)
    run = -99.0
    for i in range(on + 3, len(edb)):
        run = max(run, edb[i - 1])
        if edb[i] < run - 18 and edb[i] < -30:
            end = i
            break
    end = min(end, on + 120)
    seg = y[on * hop: end * hop]
    if len(seg) < sr // 20:
        return None
    rms_lead = float(np.sqrt((seg ** 2).mean()))
    rms_rest = float(np.sqrt((y[end * hop:] ** 2).mean())) if end * hop < len(y) else 1e-9
    try:
        f0 = librosa.yin(seg.astype(np.float32), fmin=70, fmax=1000, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch = float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        pitch = 0.0
    return {
        "dur_lead": round(len(seg) / sr, 3),
        "lead_db": round(20 * np.log10(max(rms_lead, 1e-9) / peak), 1),
        "lead_vs_rest_db": round(20 * np.log10(max(rms_lead, 1e-9) / max(rms_rest, 1e-9)), 1),
        "pitch_hz": round(pitch, 1),
    }


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, float), sr


def lead_measure(path: Path) -> dict | None:
    """Loudness/pitch of the leading vocalisation in the reference."""
    y, sr = _read_mono(path)
    return _score_from_array(y, sr)


def scream_score(meas: dict) -> int:
    """0-5: how much the leading/trailing vocalisation looks like a real
    scream (loud + high-pitched + sustained) rather than a mild interjection."""
    score = 0
    if meas["lead_vs_rest_db"] >= 3.0:
        score += 2
    elif meas["lead_vs_rest_db"] >= 0.0:
        score += 1
    if meas["pitch_hz"] >= 300:
        score += 2
    elif meas["pitch_hz"] >= 200:
        score += 1
    if meas["dur_lead"] >= 0.45:
        score += 1
    return score


def tail_measure(path: Path) -> dict | None:
    """Loudness/pitch of the TRAILING vocalisation (mirrors lead_measure by
    scoring the time-reversed signal: the tail's leading edge, in reverse, is
    the line's trailing edge)."""
    y, sr = _read_mono(path)
    return _score_from_array(y[::-1], sr)


def main() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    refdirs = {}
    for p in WS.rglob("*_EN.wav"):
        refdirs.setdefault(p.parent.name, p.parent)

    rows = []
    for r in d["hibridos"]:
        # Only Main_: 17 bank names collide across series and the workspace refs
        # are all Main, so pairing any other series with them mixes lines up.
        m = re.match(r"^Main_(\d+_\d+_[A-Z])$", r["event"])
        if not m:
            continue
        dd = refdirs.get(m.group(1))
        if not dd:
            continue
        cands = list(dd.glob(f"*L{r['stream_index']:03d}_EN.wav"))
        if not cands:
            continue
        meas = lead_measure(cands[0])
        if not meas:
            continue
        lead_tok = r["gritos"][0]
        aleman = bool(GERMAN_WORDS.match(fold(collapse(lead_tok))))
        score = scream_score(meas)
        rows.append({
            "bank": m.group(1), "stream": r["stream_index"], "ref": str(cands[0]),
            "en": r["text_en"], "de": r["text_de"], "lead": lead_tok,
            "en_tiene_grito": r["en_tiene_grito"], "palabra_alemana": aleman,
            **meas, "score": score,
            "empalmable": bool(r["en_tiene_grito"] and not aleman),
        })

    rows.sort(key=lambda x: -x["score"])
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"lineas medidas: {len(rows)}")
    print(f"{'banco':<12}{'sc':>3}{'lead_dB':>9}{'vs_resto':>10}{'pitch':>7}{'dur':>6}  "
          f"{'empalm':>7}  lead")
    print("-" * 88)
    for x in rows[:32]:
        print(f"{x['bank']:<12}{x['score']:>3}{x['lead_db']:>9.1f}"
              f"{x['lead_vs_rest_db']:>10.1f}{x['pitch_hz']:>7.0f}{x['dur_lead']:>6.2f}  "
              f"{'SI' if x['empalmable'] else ('ALEMAN' if x['palabra_alemana'] else 'no'):>7}  "
              f"{x['lead']!r}  {x['de'][:40]}")
    print(f"\n-> {OUT.name}")


if __name__ == "__main__":
    main()
