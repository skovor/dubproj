#!/usr/bin/env python3
"""Generate the scream listening set: original / plain TTS / splice.

Uses the SUKAKU OmniVoice build, Matthias settings (num_step=32, guidance=2.0,
ref_text = the English original).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
from splice_interjection import (  # noqa: E402
    OmniVoice, OmniVoiceGenerationConfig, energy_end,
)

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
BASE = Path(
    r"C:\Users\juand\Desktop\moddeutsch\OmniVoice-clean-0.2.1\persona_project"
    r"\COMPARACION_GRITOS"
)
AUD = BASE / "audio"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_PAUSE = 0.070   # minimum breath between grunt and sentence, seconds
XFADE = 0.025       # crossfade length, seconds
BED_SPAN = 0.600    # how far past the seam the room tone reaches before fading
                    # out. It is there to bridge the join, not to sit under the
                    # whole German phrase.


def crossfade_weights(
    left: np.ndarray, right: np.ndarray, curve: str = "equal_gain",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return smooth splice gains and the measured waveform correlation.

    ``equal_gain`` is the complementary raised-cosine used historically.  It
    keeps two coherent copies from getting louder, but it can thin unrelated
    room tones by roughly 3 dB at the midpoint. ``equal_power`` is appropriate
    for unrelated signals but can bump coherent material. ``correlation`` uses
    the measured overlap correlation to normalise an equal-power pair, so its
    expected energy remains continuous in either case.
    """
    n = min(len(left), len(right))
    if n <= 0:
        return np.zeros(0), np.zeros(0), 0.0
    theta = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float64)
    a = np.asarray(left[-n:], dtype=np.float64)
    b = np.asarray(right[:n], dtype=np.float64)
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    denom_corr = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    rho = float(np.dot(ac, bc) / denom_corr) if denom_corr > 1e-12 else 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    if curve == "equal_gain":
        fo = np.cos(theta) ** 2
        fi = np.sin(theta) ** 2
    elif curve == "equal_power":
        fo = np.cos(theta)
        fi = np.sin(theta)
    elif curve == "correlation":
        fo = np.cos(theta)
        fi = np.sin(theta)
        norm = np.sqrt(np.maximum(
            1e-8, fo ** 2 + fi ** 2 + 2.0 * rho * fo * fi,
        ))
        fo /= norm
        fi /= norm
    else:
        raise ValueError(f"unknown crossfade curve: {curve}")
    return fo.astype(np.float32), fi.astype(np.float32), rho


def build_leading_splice(
    y, sr, raw_body, sr_m, b, cut,
    min_pause_seconds: float = MIN_PAUSE,
    crossfade_seconds: float = XFADE,
    crossfade_curve: str = "equal_gain",
):
    """Original head (scream + the actor's pause) + synthesised sentence.

    Single implementation on purpose: this chain used to be copy-pasted across
    scripts and fixes landed in one copy only.

    Returns (audio, seam seconds, room-tone used).
    """
    head = y[: int(cut * sr)]
    if sr != sr_m:
        head = torchaudio.functional.resample(
            torch.from_numpy(head).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()

    body = trim_lead_silence(raw_body, sr_m)
    body = match_level(body, speech_rms(y[int(cut * sr):]))
    tone_pre, tone_rms_pre = room_tone(y, sr, seam=cut)
    pk_body = float(np.abs(body).max())
    if pk_body > 0.97:                       # keep headroom; match_level used to
        body = body * (0.97 / pk_body)       # leave the peak at +0.15 dBFS
    if tone_rms_pre > 0:
        body = lower_floor(body, sr_m, tone_rms_pre)
    if tone_rms_pre > 0 and len(tone_pre) and sr != sr_m:
        tone_pre = torchaudio.functional.resample(
            torch.from_numpy(tone_pre).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()

    h = head.copy()
    if (cut - b) < min_pause_seconds:
        extra = int((min_pause_seconds - (cut - b)) * sr_m)
        if tone_rms_pre > 0:
            filler = tile_tone(tone_pre, extra, sr_m)
        else:
            q = min(int(0.030 * sr_m), max(1, len(h) // 4))
            seg, best = None, None
            for i in range(0, max(1, len(h) - q), max(1, q // 2)):
                rr = float(np.sqrt((h[i:i + q] ** 2).mean()))
                if best is None or rr < best:
                    best, seg = rr, h[i:i + q]
            filler = tile_tone(seg, extra, sr_m) if seg is not None and best else np.zeros(extra)
        h = np.concatenate([h, filler])

    x = min(int(crossfade_seconds * sr_m), len(h) // 2, len(body) // 2)
    if x > 0:
        fo, fi, _ = crossfade_weights(h[-x:], body[:x], crossfade_curve)
        spl = np.concatenate([h[:-x], h[-x:] * fo + body[:x] * fi, body[x:]])
    else:
        spl = np.concatenate([h, body])
    seam_at = max(0, len(h) - x) / sr_m

    used_tone = False
    if tone_rms_pre > 0 and len(tone_pre):
        start = int(seam_at * sr_m)
        span = min(int(BED_SPAN * sr_m), max(0, len(spl) - start))
        if span > int(0.05 * sr_m):
            bed = tile_tone(tone_pre, span, sr_m)
            seg = spl[start:start + span]
            have = float(np.percentile(np.abs(seg), 5)) if len(seg) else 0.0
            deficit = max(0.0, tone_rms_pre ** 2 - have ** 2) ** 0.5
            scale = min(1.0, deficit / max(tone_rms_pre, 1e-12))
            if scale > 0.05:
                bed *= scale
                rin = min(int(0.100 * sr_m), span)
                bed[:rin] *= np.linspace(0.0, 1.0, rin)
                rout = min(int(0.250 * sr_m), span)
                bed[-rout:] *= np.linspace(1.0, 0.0, rout)
                spl = spl.copy()
                spl[start:start + span] += bed
                used_tone = True

    pk = np.abs(spl).max()
    if pk > 0.99:
        spl = spl / pk * 0.99
    return spl, seam_at, used_tone


def lay_seam_bed(spl: np.ndarray, seam_at: float, tone: np.ndarray,
                 tone_rms: float, sr: int, span_s: float = BED_SPAN,
                 pre_s: float = 0.30) -> tuple[np.ndarray, bool]:
    """Puentea el suelo de ruido a ambos lados de una costura de empalme.

    La costura se oye como un "corte" cuando el suelo de ruido salta: el original
    lleva room-tone y la síntesis es silencio digital, así que al juntarlos el
    fondo desaparece/aparece de golpe. Esto añade SOLO el room-tone que falta
    (las potencias de ruido suman) en una ventana centrada en la costura
    (`pre_s` antes y el resto después), con fundido de entrada/salida. Único
    sitio con esta lógica: antes estaba solo en el empalme inicial, y el final
    -- sin ella -- era el que sonaba cortado.
    """
    if tone_rms <= 0 or not len(tone):
        return spl, False
    start = max(0, int((seam_at - pre_s) * sr))
    span = min(int(span_s * sr), len(spl) - start)
    if span <= int(0.05 * sr):
        return spl, False
    bed = tile_tone(tone, span, sr)
    seg = spl[start:start + span]
    have = float(np.percentile(np.abs(seg), 5)) if len(seg) else 0.0
    deficit = max(0.0, tone_rms ** 2 - have ** 2) ** 0.5
    scale = min(1.0, deficit / max(tone_rms, 1e-12))
    if scale <= 0.05:
        return spl, False
    bed *= scale
    rin = min(int(0.100 * sr), span)
    bed[:rin] *= np.linspace(0.0, 1.0, rin)
    rout = min(int(0.150 * sr), span)
    bed[-rout:] *= np.linspace(1.0, 0.0, rout)
    spl = spl.copy()
    spl[start:start + span] += bed
    return spl, True


def _tail_start(y: np.ndarray, sr: int) -> float:
    """Dónde arranca la última ráfaga vocal: se camina hacia atrás hasta el valle
    que la precede. Espejo de energy_end, que camina hacia delante."""
    peak = np.abs(y).max()
    if peak <= 0:
        return 0.0
    hop = max(1, int(sr * 0.010))
    n = (len(y) // hop) * hop
    env = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(axis=1))
    edb = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    end = next((i for i in range(len(edb) - 1, -1, -1) if edb[i] > -40), len(edb) - 1)
    run = -99.0
    for i in range(end - 3, 0, -1):
        run = max(run, edb[i + 1])
        if edb[i] < run - 18 and edb[i] < -30:
            j = i
            while j > 0 and edb[j] < run - 18:
                j -= 1
            return (j + (i - j) // 2) * hop / sr
    return 0.0


def build_trailing_splice(
    y, sr, raw_body, sr_m,
    min_pause_seconds: float = MIN_PAUSE,
    crossfade_seconds: float = XFADE,
    crossfade_curve: str = "equal_gain",
):
    """Frase sintetizada + grito ORIGINAL al final, con el MISMO trato de costura
    que el empalme inicial (crossfade real + cama de room-tone que puentea el
    suelo). Espejo de build_leading_splice. Devuelve (audio, costura_s, uso_sala).
    """
    ts = _tail_start(y, sr)
    tail = y[int(ts * sr):]
    if sr != sr_m:
        tail = torchaudio.functional.resample(
            torch.from_numpy(tail).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()

    body = trim_lead_silence(raw_body, sr_m)
    body = match_level(body, speech_rms(y[: int(ts * sr)]))
    tone, tone_rms = room_tone(y, sr, seam=ts)
    if tone_rms > 0:
        body = lower_floor(body, sr_m, tone_rms)
        if sr != sr_m and len(tone):
            tone = torchaudio.functional.resample(
                torch.from_numpy(tone).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()
    pk_body = float(np.abs(body).max())
    if pk_body > 0.97:
        body = body * (0.97 / pk_body)

    pause = int(min_pause_seconds * sr_m)
    filler = tile_tone(tone, pause, sr_m) if tone_rms > 0 else np.zeros(pause)
    b = np.concatenate([body, filler])
    x = min(int(crossfade_seconds * sr_m), len(b) // 2, len(tail) // 2)
    if x > 0:
        fo, fi, _ = crossfade_weights(b[-x:], tail[:x], crossfade_curve)
        spl = np.concatenate([b[:-x], b[-x:] * fo + tail[:x] * fi, tail[x:]])
    else:
        spl = np.concatenate([b, tail])
    seam_at = max(0, len(b) - x) / sr_m
    spl, used = lay_seam_bed(spl, seam_at, tone, tone_rms, sr_m)
    pk = np.abs(spl).max()
    if pk > 0.99:
        spl = spl / pk * 0.99
    return spl, seam_at, used


def seam_notch_db(y: np.ndarray, sr: int, seam: float) -> float:
    """Depth of the level hole at the seam, in dB. Automatic QA gate.

    Two fades concatenated leave a silent notch that reads as a small cut;
    measured 47-58 dB deep before the crossfade fix. Anything beyond ~12 dB
    here means the join is audible.
    """
    peak = np.abs(y).max()
    if peak <= 0:
        return 0.0
    hop = max(1, int(sr * 0.002))
    n = (len(y) // hop) * hop
    if n <= 0:
        return 0.0
    env = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(axis=1))
    edb = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    c = int(seam * sr / hop)
    w = max(3, int(0.030 * sr / hop))
    left = edb[max(0, c - w):c]
    if len(left) < 2:
        return 0.0
    # Compare against the level the PAUSE is already sitting at, not against the
    # speech peaks: an intended pause is not a defect, and measuring it against
    # the voice reported every deliberate breath as a 30 dB hole.
    floor_before = float(np.median(left))
    core = edb[max(0, c - 3):c + 4]
    if not len(core):
        return 0.0
    return float(floor_before - core.min())


def strip_lead(text: str) -> str:
    """Drop the leading vocalisation token and any punctuation after it."""
    m = re.match(r"^\s*[^\w]*[^\W\d_]+\s*(?:[.!?,…]+|\.{2,})?\s*(.*)$", text, re.S | re.U)
    return (m.group(1).strip() if m else text).strip()


def room_tone(y: np.ndarray, sr: int, seam: float | None = None,
              win: float = 0.20) -> tuple[np.ndarray, float]:
    """Room tone to lay under the synthesised half, plus its RMS.

    The seam is audible not because the voice level jumps (+0.4 dB on average)
    but because the noise floor does: -72.7 dB on average, down to -205 dB.
    The original carries room tone, the synthesis is digital silence, so the
    room vanishes at the join.

    Taken from the pause immediately BEFORE the seam when possible: that is the
    level the join has to match. Using the globally quietest window instead
    overshot by up to +80 dB, because the quietest stretch of the file can sit
    well below the level right at the cut.
    """
    n = int(win * sr)
    if len(y) < n * 2:
        return np.zeros(0), 0.0
    peak = float(np.abs(y).max())
    if peak <= 0:
        return np.zeros(0), 0.0
    # A window only counts as room tone if it really is quiet. Without this
    # guard the 200 ms ending at the seam mostly contained the SCREAM (the
    # actor's pause can be as short as 29 ms), so the "tone" tiled under the
    # synthesis was the scream itself, audibly doubling it.
    quiet_max = peak * 10 ** (-38 / 20)

    def take(start: int, end: int):
        start, end = max(0, start), min(len(y), end)
        if end - start < n // 2:
            return None
        seg = y[start:end]
        r = float(np.sqrt((seg ** 2).mean()))
        if r <= 1e-7 or r > quiet_max:
            return None
        return seg.copy(), r

    # 1) the lead-in before the vocalisation starts: cleanest room tone there is
    onset = 0
    thr = peak * 10 ** (-45 / 20)
    nz = np.where(np.abs(y) > thr)[0]
    if len(nz):
        onset = int(nz[0])
    got = take(0, onset)
    if got:
        return got
    # 2) the pause immediately before the seam, if it is genuinely quiet
    if seam is not None:
        got = take(int(seam * sr) - n, int(seam * sr))
        if got:
            return got
    # 3) quietest window anywhere
    hop = max(1, n // 4)
    best = None
    for i in range(0, len(y) - n, hop):
        r = float(np.sqrt((y[i:i + n] ** 2).mean()))
        if r <= 1e-7 or r > quiet_max:
            continue
        if best is None or r < best[1]:
            best = (i, r)
    if best is None:
        return np.zeros(0), 0.0
    return y[best[0]:best[0] + n].copy(), best[1]


def tile_tone(tone: np.ndarray, length: int, sr: int) -> np.ndarray:
    """Repeat the room tone to `length`, crossfading and alternating direction
    so the repetition does not become an audible loop."""
    if len(tone) == 0 or length <= 0:
        return np.zeros(max(0, length))
    xf = min(int(0.020 * sr), len(tone) // 4)
    out = np.zeros(length)
    pos, flip = 0, False
    while pos < length:
        chunk = tone[::-1] if flip else tone
        flip = not flip
        take = min(len(chunk), length - pos)
        seg = chunk[:take].copy()
        if pos > 0 and xf > 0 and take > xf:
            ramp = np.linspace(0, 1, xf)
            out[pos:pos + xf] = out[pos:pos + xf] * (1 - ramp) + seg[:xf] * ramp
            out[pos + xf:pos + take] = seg[xf:take]
        else:
            out[pos:pos + take] = seg
        pos += max(1, take - xf)
    return out


def lower_floor(body: np.ndarray, sr: int, target_floor: float) -> np.ndarray:
    """Pull the synthesis's own noise floor down towards the original's.

    Laying room tone under only ever ADDS. Where the original's pause is
    quieter than the synthesis's hiss (measured: -63 dB vs -37 dB on two
    lines), the seam still jumps. This gently attenuates quiet frames -- a soft
    downward expansion, smoothed so it does not pump.
    """
    peak = np.abs(body).max()
    if peak <= 0 or target_floor <= 0:
        return body
    hop = max(1, int(sr * 0.010))
    n = (len(body) // hop) * hop
    if n <= 0:
        return body
    env = np.sqrt((body[:n].reshape(-1, hop) ** 2).mean(axis=1))
    # The threshold has to come from the ACTUAL noise floor, not from a fixed
    # offset below the peak. Measured: 34% of frames sit under -30 dB of peak --
    # that is a third of the speech (consonants, word endings, unstressed
    # syllables), and ducking it ~6 dB with a 50 ms smoother is a gate pumping
    # on the voice. That is what made it sound robotic.
    cur_floor = float(np.percentile(env, 5))
    if cur_floor <= 0 or cur_floor <= target_floor:
        return body
    thr = cur_floor * 10 ** (6 / 20)         # only within +6 dB of the floor
    if not np.any(env < thr):
        return body
    need = max(target_floor / cur_floor, 10 ** (-8 / 20))   # cap at -8 dB
    gain = np.ones(len(env))
    below = env < thr
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(1.0 - env[below] / thr, 0.0, 1.0)
    gain[below] = 1.0 - frac * (1.0 - need)
    k = 5
    gain = np.convolve(gain, np.ones(k) / k, mode="same")
    g = np.repeat(gain, hop)
    out = body.copy()
    out[:n] *= g[:n]
    return out


def match_level(body: np.ndarray, target_rms: float) -> np.ndarray:
    """Align the synthesised speech level to the original's, clamped to +-6 dB."""
    v = body[np.abs(body) > np.abs(body).max() * 0.1] if np.abs(body).max() > 0 else body
    if len(v) == 0:
        return body
    cur = float(np.sqrt((v ** 2).mean()))
    if cur <= 0 or target_rms <= 0:
        return body
    g = float(np.clip(target_rms / cur, 10 ** (-6 / 20), 10 ** (6 / 20)))
    return body * g


def speech_rms(y: np.ndarray) -> float:
    p = np.abs(y).max()
    if p <= 0:
        return 0.0
    v = y[np.abs(y) > p * 0.1]
    return float(np.sqrt((v ** 2).mean())) if len(v) else 0.0


def speech_resume(y: np.ndarray, sr: int, after: float) -> float:
    """Where the voice comes back after the grunt's valley.

    Cutting at the valley bottom and re-inserting a synthetic gap is fragile:
    the valley is found by relative drop while an onset gate uses an absolute
    threshold, and the two disagree (measured gaps came out 0-18 ms, clearly
    too tight). Cutting here instead keeps the actor's own pause inside the
    original audio, so it needs no reconstruction at all.
    """
    peak = np.abs(y).max()
    if peak <= 0:
        return after
    hop = max(1, int(sr * 0.010))
    n = (len(y) // hop) * hop
    if n <= 0:
        return after
    env = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(axis=1))
    edb = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    i0 = int(after * sr / hop)
    need = 2  # 20 ms sustained
    run = 0
    for i in range(i0, len(edb)):
        if edb[i] > -30.0:
            run += 1
            if run >= need:
                return (i - need + 1) * hop / sr
        else:
            run = 0
    return after


def onset_seconds(y: np.ndarray, sr: int, start: float = 0.0) -> float:
    """First sample above -45 dBFS(peak) at or after `start`."""
    peak = np.abs(y).max()
    if peak <= 0:
        return start
    thr = peak * (10 ** (-45.0 / 20.0))
    i0 = int(start * sr)
    idx = np.where(np.abs(y[i0:]) > thr)[0]
    return (i0 + idx[0]) / sr if len(idx) else start


def trim_lead_silence(y: np.ndarray, sr: int) -> np.ndarray:
    """Remove the model's own lead-in silence (measured at 137-206 ms, and it
    varies per generation -- leaving it in makes the splice gap unpredictable)."""
    if len(y) == 0:
        return y
    on = onset_seconds(y, sr)
    n = int(on * sr)
    out = y[n:] if n > 0 else y
    f = min(int(0.003 * sr), len(out))
    if f > 0:
        out = out.copy()
        out[:f] *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, f)))
    return out


def main() -> None:
    rows = json.loads((ROOT / "screams_acoustic.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["empalmable"]]
    AUD.mkdir(parents=True, exist_ok=True)

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
    cfg = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
    sr_m = model.sampling_rate

    def gen(text_de, ref, ref_text):
        arr = model.generate(text=[text_de], language=["de"], ref_audio=[ref],
                             ref_text=[ref_text], generation_config=cfg)[0]
        t = arr.detach().cpu() if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        return t.squeeze().numpy().astype(np.float64)

    out = []
    for r in rows:
        stem = f"{r['bank']}_L{r['stream']:03d}"
        rest = strip_lead(r["de"])
        if len(WORD.findall(rest)) < 1:
            print(f"  {stem}: sin frase tras el grito, se salta")
            continue

        y, sr = sf.read(r["ref"], always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = np.asarray(y, float)
        b = energy_end(y, sr, min(1.2, len(y) / sr * 0.6))
        # cut where the voice resumes, so the actor's own pause rides along
        cut = speech_resume(y, sr, b)
        head = y[: int(cut * sr)]
        if sr != sr_m:
            head = torchaudio.functional.resample(
                torch.from_numpy(head).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()

        plain = gen(r["de"], r["ref"], r["en"])
        body = gen(rest, r["ref"], r["en"])

        # The pause between grunt and sentence is NOT a constant: the actor sets
        # it, and it differs per line. Measure it in the original instead of
        # guessing (a fixed 60 ms read as too long on some lines, too short on
        # others). Then strip the model's own lead-in silence, which is variable
        # and would otherwise be added on top of the measured pause.
        body = trim_lead_silence(body, sr_m)

        # match the speech level of the portion being replaced
        tail_orig = y[int(cut * sr):]
        body = match_level(body, speech_rms(tail_orig))

        tone_pre, tone_rms_pre = room_tone(y, sr, seam=cut)
        # Headroom: match_level was leaving the peak at +0.15 dBFS with a clipped
        # sample. Pull back so the join never costs level.
        pk_body = float(np.abs(body).max())
        if pk_body > 0.97:
            body = body * (0.97 / pk_body)

        # A minimum breath. The actor's pause can be as short as 20 ms, and once
        # the synthesis's own lead-in silence is trimmed the sentence enters with
        # no air at all -- reported as "muy apresurado". Padded with the room's
        # own silence, never with digital zero.
        h = head.copy()
        pause_have = cut - b
        if pause_have < MIN_PAUSE:
            extra = int((MIN_PAUSE - pause_have) * sr_m)
            # Never pad with digital zero: that punches an absolute-silence hole
            # (measured 213 dB deep) which is exactly the artefact being fixed.
            # Prefer the room tone; failing that, recycle the head's own
            # quietest stretch so the pause keeps the recording's noise.
            if tone_rms_pre > 0:
                filler = tile_tone(tone_pre, extra, sr_m)
            else:
                q = min(int(0.030 * sr_m), max(1, len(h) // 4))
                seg, best = None, None
                for i in range(0, max(1, len(h) - q), max(1, q // 2)):
                    rr = float(np.sqrt((h[i:i + q] ** 2).mean()))
                    if best is None or rr < best:
                        best, seg = rr, h[i:i + q]
                filler = (tile_tone(seg, extra, sr_m) if seg is not None and best
                          else np.zeros(extra))
            h = np.concatenate([h, filler])

        # REAL crossfade, overlapping. Fading both sides and then concatenating
        # is two fades back to back: it digs a hole. Measured on the previous
        # build: a notch 47-58 dB deep, bottoming at -89 dB, right at the seam.
        # That hole was the "pequeño corte".
        bo = body.copy()
        x = min(int(XFADE * sr_m), len(h) // 2, len(bo) // 2)
        if x > 0:
            fo = np.cos(np.linspace(0, np.pi / 2, x)) ** 2
            fi = np.sin(np.linspace(0, np.pi / 2, x)) ** 2
            joint = h[-x:] * fo + bo[:x] * fi
            spl = np.concatenate([h[:-x], joint, bo[x:]])
        else:
            spl = np.concatenate([h, bo])
        seam_at = max(0, len(h) - x) / sr_m

        # lay the original's room tone under the synthesised half so the floor
        # does not fall off a cliff at the seam
        tone, tone_rms = room_tone(y, sr, seam=cut)
        if len(tone) and sr != sr_m:
            tone = torchaudio.functional.resample(
                torch.from_numpy(tone).unsqueeze(0).float(), sr, sr_m).squeeze(0).numpy()
        # The bed exists ONLY to bridge the join. Laying it under the whole
        # German phrase was altering the entire performance (+7.71 dB mean
        # change measured across the body) and reads as background appearing
        # mid-line. Restrict it to a short window around the seam and fade out.
        if len(tone):
            start = int(seam_at * sr_m)
            span = min(int(BED_SPAN * sr_m), max(0, len(spl) - start))
            if span > int(0.05 * sr_m):
                bed = tile_tone(tone, span, sr_m)
                # add only what is MISSING: noise powers add, so laying the full
                # original floor on top of the synthesis's own floor overshoots
                seg = spl[start:start + span]
                have = float(np.percentile(np.abs(seg), 5)) if len(seg) else 0.0
                deficit = max(0.0, tone_rms_pre ** 2 - have ** 2) ** 0.5
                scale = min(1.0, deficit / max(tone_rms_pre, 1e-12))
                if scale > 0.05:
                    bed *= scale
                    rin = min(int(0.100 * sr_m), span)
                    bed[:rin] *= np.linspace(0.0, 1.0, rin)
                    rout = min(int(0.250 * sr_m), span)
                    bed[-rout:] *= np.linspace(1.0, 0.0, rout)
                    spl = spl.copy()
                    spl[start:start + span] += bed
                else:
                    tone = np.zeros(0)
            else:
                tone = np.zeros(0)

        pk = np.abs(spl).max()
        if pk > 0.99:
            spl = spl / pk * 0.99

        sf.write(str(AUD / f"{stem}_EN.wav"), y, sr)
        sf.write(str(AUD / f"{stem}_PLAIN.wav"), plain, sr_m)
        sf.write(str(AUD / f"{stem}_SPLICE.wav"), spl, sr_m)
        notch = seam_notch_db(spl, sr_m, seam_at)
        out.append({**r, "stem": stem, "rest": rest, "boundary": round(seam_at, 3),
                    "valle": round(b, 3),
                    "pausa_ms": round(max(cut - b, MIN_PAUSE) * 1000, 0),
                    "notch_db": round(notch, 1),
                    "sala": bool(len(tone))})
        flag = "  <<< COSTURA AUDIBLE" if notch > 12 else ""
        print(f"  {stem}: corte {seam_at:.3f}s  pausa {int(max(cut-b, MIN_PAUSE)*1000):3d}ms  "
              f"hueco {notch:5.1f}dB  sala={'si' if len(tone) else 'no'}{flag}")

    (BASE / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOK: {len(out)} lineas -> {BASE}")


if __name__ == "__main__":
    main()
