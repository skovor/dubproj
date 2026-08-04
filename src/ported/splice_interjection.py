#!/usr/bin/env python3
"""Splice: keep the original actor's vocalisation, synthesise only the sentence.

A grunt has no language. "Argh!" is the same sound in English and German, so
making the TTS *act* it is both unnecessary and the thing that sounds terrible.
This keeps the real actor's interjection from the English reference and
generates only the German sentence that follows, then joins them.

Steps
  1. Whisper word timestamps on the English reference locate where the leading
     interjection ends.
  2. That head is cut from the original audio -- untouched, real performance.
  3. OmniVoice (SUKAKU build) generates only the remaining German sentence.
  4. The two are joined with a short equal-power crossfade.

Only valid for language-neutral vocalisations. Elongated German words
("Neeein", "Hiiiilfe") must NOT be spliced -- the English audio says something
else. `--check-neutral` refuses those.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

SUKAKU = Path(r"C:\Users\juand\Desktop\moddeutsch\OMNIVOICE SUKAKU\Omnivoice\Omnivoice")
OFFICIAL_021 = Path(
    r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1"
    r"\source\k2-fsa-OmniVoice-5ba967c"
)
SCRIPTS = Path(
    r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project\scripts"
)
sys.path.insert(0, str(SCRIPTS))
from clean_runtime import prepare

prepare()
# Audio-splice helpers are engine-neutral. Standalone legacy calls retain
# SUKAKU as their default, while an orchestrator that has already imported
# the verified 0.2.1 runtime must not be replaced or rejected here.
if "omnivoice" not in sys.modules:
    requested_engine = Path(os.environ.get("OMNIVOICE_ENGINE_ROOT", str(SUKAKU)))
    sys.path.insert(0, str(requested_engine))

import numpy as np
import soundfile as sf
import torch
import torchaudio

import omnivoice
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

_res = Path(omnivoice.__file__).resolve()
allowed_engines = (SUKAKU.resolve(), OFFICIAL_021.resolve())
if not any(root in _res.parents for root in allowed_engines):
    raise SystemExit(
        f"ABORT: omnivoice cargo de {_res}; no es SUKAKU ni OmniVoice 0.2.1 verificado"
    )

# Vocalisations that carry no language and may be taken from the English audio.
NEUTRAL = {
    "ah", "aah", "ahh", "aaah", "agh", "argh", "arg", "augh", "au", "aua",
    "ugh", "uh", "uhh", "uff", "urgh", "urg", "ungh", "ngh", "hng", "nng",
    "gah", "grr", "grrr", "gnn", "hmpf", "pff", "pfff", "puh", "ph", "gyah",
    "oh", "ooh", "oooh", "och", "ach", "aha", "oho", "hui", "huch", "wow",
    "wah", "woah", "whoa", "boah", "oha", "hm", "hmm", "hmmm", "mh", "mhm",
    "mmm", "ahm", "ahem", "ahem", "ah", "eh", "ha", "hah", "haha", "hehe",
    "heh", "hihi", "buhu", "tsk", "psst", "shh", "eep", "iip", "wuff", "wau",
    "woof", "arf", "bark", "seufz", "keuch", "haa", "hach", "hu", "huff",
}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def collapse(t: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", t)


def is_neutral(tok: str) -> bool:
    t = fold(tok.lower())
    return t in NEUTRAL or collapse(t) in NEUTRAL


def split_lead(text: str) -> tuple[str, str]:
    """Split 'Argh! Nein, das...' -> ('Argh', 'Nein, das...')."""
    m = re.match(r"^\s*([^\w]*)([^\W\d_]+)(\s*[^\w\s]*\s*)(.*)$", text, re.UNICODE | re.S)
    if not m:
        return "", text
    _, first, _sep, rest = m.groups()
    if not is_neutral(first) or not rest.strip():
        return "", text
    return first, rest.strip()


def energy_end(y: np.ndarray, sr: int, search_to: float) -> float:
    """Boundary = the FIRST deep valley after the grunt, not the deepest one.

    Whisper does not transcribe grunts at all (it folds them into the first real
    word), so word timestamps cannot locate this edge -- measured on
    100_030_M_L015, where Whisper reported "There's" starting at 0.000 s while
    the grunt actually runs to ~0.50 s. Taking the global minimum instead lands
    inside the following speech (0.745 s there, which already says "There's").
    So: walk forward and stop at the first sustained drop relative to the level
    the grunt was holding.
    """
    hop = max(1, int(sr * 0.010))
    n = int(min(len(y), search_to * sr))
    seg = y[:n]
    pad = (-len(seg)) % hop
    if pad:
        seg = np.concatenate([seg, np.zeros(pad)])
    env = np.sqrt((seg.reshape(-1, hop) ** 2).mean(axis=1))
    peak = env.max() if len(env) else 0.0
    if peak <= 0 or len(env) < 8:
        return search_to
    edb = 20.0 * np.log10(np.maximum(env, 1e-12) / peak)

    onset = next((i for i, v in enumerate(edb) if v > -40.0), 0)
    running = -99.0
    need = max(2, int(0.020 / 0.010))  # valley must last >= 20 ms
    for i in range(onset + 3, len(edb)):
        running = max(running, edb[i - 1])
        if edb[i] < running - 18.0 and edb[i] < -30.0:
            j = i
            while j < len(edb) and edb[j] < running - 18.0:
                j += 1
            if (j - i) >= need:
                lo = int(np.argmin(edb[i:j]) + i)
                return lo * hop / sr
    return search_to


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-en", type=Path, required=True, help="original English audio")
    ap.add_argument("--text-en", required=True)
    ap.add_argument("--text-de", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-plain", type=Path, default=None,
                    help="also write the plain full-TTS version, for comparison")
    ap.add_argument("--gap-ms", type=float, default=60.0)
    args = ap.parse_args()

    lead_en, _ = split_lead(args.text_en)
    lead_de, rest_de = split_lead(args.text_de)
    if not lead_de:
        raise SystemExit(f"La linea DE no empieza por vocalizacion neutra: {args.text_de!r}")
    print(f"  interjeccion DE : {lead_de!r}   (EN: {lead_en!r})")
    print(f"  frase a generar : {rest_de!r}")

    # 1-2. locate the interjection in the English audio and cut it.
    # Energy, not Whisper: Whisper does not transcribe grunts, it folds them
    # into the following word, so its timestamps place the boundary at 0.
    y, sr = sf.read(str(args.ref_en), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    boundary = energy_end(y, sr, min(1.2, len(y) / sr * 0.6))
    print(f"  interjeccion original: 0 -> {boundary:.3f}s")
    head = y[: int(boundary * sr)]

    # verify the head kept only the vocalisation
    from faster_whisper import WhisperModel

    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    probe = args.out.parent / "_head_probe.wav"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(probe), head, sr)
    hs, _ = asr.transcribe(str(probe), language="en")
    heard = " ".join(s.text.strip() for s in hs).strip()
    probe.unlink(missing_ok=True)
    leaked = [w for w in WORD.findall(heard.lower()) if not is_neutral(w)]
    print(f"  la cabeza dice: {heard!r}" + (f"   !! PALABRAS FILTRADAS: {leaked}" if leaked else "   (limpia)"))

    # 3. generate only the sentence
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16
    )
    cfg = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)

    def gen(text_de: str) -> np.ndarray:
        arr = model.generate(
            text=[text_de], language=["de"], ref_audio=[str(args.ref_en)],
            ref_text=[args.text_en], generation_config=cfg,
        )[0]
        t = arr.detach().cpu() if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        return t.squeeze().numpy().astype(np.float64)

    body = gen(rest_de)
    msr = model.sampling_rate
    if msr != sr:
        head_t = torch.from_numpy(head).unsqueeze(0).float()
        head = torchaudio.functional.resample(head_t, sr, msr).squeeze(0).numpy()
        sr = msr

    # 4. join with an equal-power crossfade
    gap = np.zeros(int(args.gap_ms / 1000.0 * sr))
    x = int(0.015 * sr)
    if len(head) > x and len(body) > x:
        fo = np.cos(np.linspace(0, np.pi / 2, x)) ** 2
        fi = np.sin(np.linspace(0, np.pi / 2, x)) ** 2
        head = head.copy()
        head[-x:] *= fo
        body = body.copy()
        body[:x] *= fi
    out = np.concatenate([head, gap, body])
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.99
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), out, sr)
    print(f"  -> {args.out.name}  ({len(out)/sr:.2f}s)")

    if args.also_plain:
        plain = gen(args.text_de)
        sf.write(str(args.also_plain), plain, msr)
        print(f"  -> {args.also_plain.name}  ({len(plain)/msr:.2f}s)  [TTS completo, para comparar]")


if __name__ == "__main__":
    main()
