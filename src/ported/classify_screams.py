#!/usr/bin/env python3
"""Isolate the vocalisations that actually break the model: screams, cries,
pain, battle cries.

Mild discourse markers ("Oh", "Hmm", "Ach", "Aeh") are excluded on purpose --
OmniVoice renders those acceptably on its own, so special handling there buys
nothing. What breaks is high-intensity voice: a sustained scream, a wail, a
grunt of pain, a battle shout.

Signals used, on the German text (what gets synthesised):
  - a scream lexeme (gyaah, hraaah, waaah, aaah, iiih, grr, argh, ...)
  - OR strong elongation (>=3 repeats) on a vocalisation token
  - OR all-caps shouting on a vocalisation
Guarded against German words with legitimate triple letters ("helllichten",
"Schifffahrt", "Bettttuch").
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
CORPUS = ROOT / "corpus_lines.jsonl"
OUT = ROOT / "corpus_screams.json"

# High-intensity vocalisation stems (collapsed form).
SCREAM_STEMS = {
    "a", "ah", "aah", "agh", "argh", "arg", "augh", "gah", "gyah", "gya",
    "hra", "hrah", "hraah", "wah", "waah", "wa", "ya", "yah", "yaah",
    "iih", "ih", "eek", "iip", "eep", "uah", "uaah", "oah",
    "ugh", "urgh", "urg", "ungh", "ngh", "hng", "nng", "grr", "gr",
    "no", "nein", "nooo", "aiee", "aie", "hiih", "hilfe",
    "haa", "hah", "ha", "keuch", "stohn", "achz", "schluchz", "winsel",
    "buhu", "aua", "au", "autsch", "oof", "ow",
}
# Mild markers explicitly NOT worth special handling.
MILD = {
    "oh", "ooh", "och", "ach", "aha", "oho", "hm", "hmm", "mh", "mhm", "mmm",
    "ahm", "ahem", "ahem", "ah", "eh", "ahm", "em", "erm", "um", "tja", "nun",
    "naja", "hey", "he", "ey", "na", "so", "ja", "puh", "hui", "huch", "wow",
    "boah", "oha", "hoppla", "ups", "oops", "je", "oje", "aha",
}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# German words where a triple letter is legitimate orthography, not a scream.
LEGIT_TRIPLE = re.compile(
    r"(helllicht|schifffahrt|bettt|ballett|stofff|kaffeee|schnellll|"
    r"fussball|balllos|rolllad|stilllleg|stillleg|volllauf|nulll)", re.I)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def collapse(t: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", t)


def strong_elong(tok: str) -> bool:
    if LEGIT_TRIPLE.search(tok):
        return False
    return bool(re.search(r"(.)\1{2,}", tok))


def is_scream(tok: str, raw: str) -> bool:
    f = fold(tok)
    c = collapse(f)
    if c in MILD and not strong_elong(tok):
        return False
    if c in SCREAM_STEMS and strong_elong(tok):
        return True
    if c in SCREAM_STEMS and f != c:      # any elongation of a scream stem
        return True
    if c in {"grr", "gr", "gyah", "hra", "hrah", "argh", "gah"}:
        return True
    # a vowel-only token stretched out is a scream by construction
    if strong_elong(tok) and re.fullmatch(r"[aeiouyäöü]+[hn]?", c or ""):
        return True
    # SHOUTED in caps and elongated
    if raw.isupper() and len(raw) >= 4 and strong_elong(tok):
        return True
    return False


def main() -> None:
    recs = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    puros, hibridos = [], []
    stats = Counter()

    for r in recs:
        de = r.get("text_de") or ""
        en = r.get("text_en") or ""
        raws = WORD.findall(de)
        if not raws:
            continue
        toks = [t.lower() for t in raws]
        hits = [i for i, t in enumerate(toks) if is_scream(t, raws[i])]
        if not hits:
            continue
        rec = {**r, "n_words": len(toks), "gritos": [raws[i] for i in hits]}
        if len(hits) == len(toks):
            stats["puro"] += 1
            puros.append(rec)
        else:
            # does the ENGLISH also scream there? only then can it be spliced
            en_toks = WORD.findall(en)
            en_scream = any(is_scream(t.lower(), t) for t in en_toks)
            rec["en_tiene_grito"] = en_scream
            rec["posicion"] = "inicial" if 0 in hits else "interna"
            stats["hibrido_" + ("empalmable" if en_scream else "SIN_grito_EN")] += 1
            hibridos.append(rec)

    hibridos.sort(key=lambda x: (not x["en_tiene_grito"], x["posicion"] != "inicial"))
    OUT.write_text(json.dumps(
        {"n_puros": len(puros), "n_hibridos": len(hibridos), "stats": dict(stats),
         "hibridos": hibridos, "puros": puros[:300]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"corpus: {len(recs)} lineas")
    print(f"  GRITO PURO (linea entera)      : {len(puros)}")
    print(f"  GRITO DENTRO DE FRASE          : {len(hibridos)}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {k:<32} {v:>5}")
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
