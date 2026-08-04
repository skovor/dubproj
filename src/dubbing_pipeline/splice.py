"""Empalme A/B helpers that preserve the real actor's effort and room tone."""
from __future__ import annotations

import re
import unicodedata


def _np():
    import numpy as np
    return np


NEUTRAL_EFFORTS = {"ah", "aah", "ahh", "agh", "argh", "augh", "eh", "gah", "ha", "hmm", "huh", "ngh", "oh", "ooh", "oof", "ow", "ugh", "uh", "uff", "urgh", "wow", "woah", "whoa", "huch", "ach", "tsk", "tch", "pff", "puh", "hmpf", "grr", "wuff", "wau", "woof", "arf", "miau", "meow"}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def is_neutral(token: str) -> bool:
    value = fold(token)
    collapsed = re.sub(r"(.)\1+", r"\1", value)
    return value in NEUTRAL_EFFORTS or collapsed in NEUTRAL_EFFORTS


def split_lead(text: str) -> tuple[str, str]:
    match = re.match(r"^\s*([^\w]*)([^\W\d_]+)(\s*[^\w\s]*\s*)(.*)$", str(text), re.UNICODE | re.S)
    if not match:
        return "", str(text).strip()
    _, first, _, rest = match.groups()
    if not is_neutral(first) or not rest.strip():
        return "", str(text).strip()
    return first, rest.strip()


def energy_end(audio, sample_rate: int, search_to: float) -> float:
    """Return the first sustained energy valley after a leading effort."""
    np = _np(); hop = max(1, int(sample_rate * .010)); limit = int(min(len(audio), search_to * sample_rate))
    segment = np.asarray(audio[:limit], dtype="float64")
    if len(segment) < hop * 8:
        return search_to
    padded = np.pad(segment, (0, (-len(segment)) % hop))
    env = np.sqrt((padded.reshape(-1, hop) ** 2).mean(axis=1)); peak = float(env.max())
    if peak <= 0:
        return search_to
    db = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    onset = next((idx for idx, value in enumerate(db) if value > -40), 0)
    running, needed = -99.0, 2
    for index in range(onset + 3, len(db)):
        running = max(running, float(db[index - 1]))
        if db[index] < running - 18 and db[index] < -30:
            end = index
            while end < len(db) and db[end] < running - 18:
                end += 1
            if end - index >= needed:
                return float(index + int(np.argmin(db[index:end]))) * hop / sample_rate
    return search_to


def speech_resume(audio, sample_rate: int, after: float) -> float:
    np = _np(); value = np.asarray(audio); peak = float(np.max(np.abs(value))) if len(value) else 0.0
    if peak <= 0:
        return after
    hop = max(1, int(sample_rate * .010)); usable = (len(value) // hop) * hop
    if usable <= 0:
        return after
    env = np.sqrt((value[:usable].reshape(-1, hop) ** 2).mean(axis=1)); db = 20 * np.log10(np.maximum(env, 1e-12) / peak)
    run = 0
    for index in range(int(after * sample_rate / hop), len(db)):
        run = run + 1 if db[index] > -30 else 0
        if run >= 2:
            return float(index - run + 1) * hop / sample_rate
    return after


def crossfade_weights(left, right, seconds: float, sample_rate: int):
    np = _np(); n = min(int(seconds * sample_rate), len(left), len(right))
    if n <= 0:
        return np.zeros(0), np.zeros(0), 0.0
    theta = np.linspace(0, np.pi / 2, n)
    a, b = np.asarray(left[-n:], dtype="float64"), np.asarray(right[:n], dtype="float64")
    ac, bc = a - a.mean(), b - b.mean(); denom = np.linalg.norm(ac) * np.linalg.norm(bc)
    rho = float(np.dot(ac, bc) / denom) if denom > 1e-12 else 0.0
    rho = float(np.clip(rho, -.95, .95))
    out, inn = np.cos(theta) ** 2, np.sin(theta) ** 2
    return out.astype("float32"), inn.astype("float32"), rho


def trim_lead_silence(audio, sample_rate: int):
    from .timing import trim_lead_silence as trim
    return trim(audio, sample_rate)


def room_tone(audio, sample_rate: int, seam: float | None = None, window: float = .20):
    np = _np(); value = np.asarray(audio, dtype="float64"); n = int(window * sample_rate)
    if len(value) < n * 2:
        return np.zeros(0), 0.0
    peak = float(np.max(np.abs(value)))
    if peak <= 0:
        return np.zeros(0), 0.0
    quiet_max = peak * 10 ** (-38 / 20)

    def take(start, end):
        start, end = max(0, start), min(len(value), end)
        if end - start < n // 2:
            return None
        segment = value[start:end]; rms = float(np.sqrt((segment ** 2).mean()))
        return (segment.copy(), rms) if 1e-7 < rms <= quiet_max else None

    onset = np.where(np.abs(value) > peak * 10 ** (-45 / 20))[0]
    got = take(0, int(onset[0])) if len(onset) else None
    if got:
        return got
    if seam is not None:
        got = take(int(seam * sample_rate) - n, int(seam * sample_rate))
        if got:
            return got
    best = None
    for start in range(0, max(1, len(value) - n), max(1, n // 4)):
        segment = value[start:start + n]; rms = float(np.sqrt((segment ** 2).mean()))
        if 1e-7 < rms <= quiet_max and (best is None or rms < best[1]):
            best = (start, rms)
    return (value[best[0]:best[0] + n].copy(), best[1]) if best else (np.zeros(0), 0.0)


def tile_tone(tone, length: int, sample_rate: int):
    np = _np(); tone = np.asarray(tone); output = np.zeros(max(0, length))
    if len(tone) == 0 or length <= 0:
        return output
    crossfade = min(int(.020 * sample_rate), len(tone) // 4); position = 0; reverse = False
    while position < length:
        chunk = tone[::-1] if reverse else tone; reverse = not reverse
        take = min(len(chunk), length - position); segment = chunk[:take].copy()
        if position and crossfade and take > crossfade:
            ramp = np.linspace(0, 1, crossfade); output[position:position + crossfade] = output[position:position + crossfade] * (1 - ramp) + segment[:crossfade] * ramp; output[position + crossfade:position + take] = segment[crossfade:take]
        else:
            output[position:position + take] = segment
        position += max(1, take - crossfade)
    return output


def _match_level(body, target_rms: float):
    np = _np(); value = np.asarray(body, dtype="float32"); peak = float(np.max(np.abs(value))) if len(value) else 0.0
    active = value[np.abs(value) > peak * .1] if peak else value
    current = float(np.sqrt((active ** 2).mean())) if len(active) else 0.0
    if current <= 0 or target_rms <= 0:
        return value
    return value * float(np.clip(target_rms / current, 10 ** (-6 / 20), 10 ** (6 / 20)))


def _speech_rms(audio):
    np = _np(); value = np.asarray(audio); peak = float(np.max(np.abs(value))) if len(value) else 0.0
    active = value[np.abs(value) > peak * .1] if peak else value
    return float(np.sqrt((active ** 2).mean())) if len(active) else 0.0


def build_leading_splice(source, source_rate: int, generated, generated_rate: int, boundary: float, cut: float | None = None, min_pause_seconds: float = .070, crossfade_seconds: float = .025):
    """Empalme B: original head/effort + only the generated target body."""
    np = _np(); cut = boundary if cut is None else cut
    head = np.asarray(source[:int(cut * source_rate)], dtype="float32")
    if source_rate != generated_rate:
        head = _resample(head, source_rate, generated_rate)
    body = trim_lead_silence(generated, generated_rate)
    body = _match_level(body, _speech_rms(np.asarray(source[int(cut * source_rate):])))
    tone, tone_rms = room_tone(source, source_rate, seam=cut)
    if source_rate != generated_rate and len(tone):
        tone = _resample(tone, source_rate, generated_rate)
    if cut - boundary < min_pause_seconds:
        filler = tile_tone(tone, int((min_pause_seconds - max(0, cut - boundary)) * generated_rate), generated_rate) if tone_rms else np.zeros(0)
        head = np.concatenate([head, filler])
    out_gain, in_gain, _ = crossfade_weights(head, body, crossfade_seconds, generated_rate)
    n = len(out_gain)
    result = np.concatenate([head[:-n] if n else head, head[-n:] * out_gain + body[:n] * in_gain if n else np.zeros(0), body[n:] if n else body])
    peak = float(np.max(np.abs(result))) if len(result) else 0.0
    return result if peak <= .99 else result * (.99 / peak), max(0, len(head) - n) / generated_rate


def _resample(audio, source_rate: int, target_rate: int):
    from .audio import resample_exact
    return resample_exact(audio, source_rate, target_rate)


def seam_notch_db(audio, sample_rate: int, seam: float) -> float:
    np = _np(); value = np.asarray(audio); peak = float(np.max(np.abs(value))) if len(value) else 0.0
    if peak <= 0:
        return 0.0
    hop = max(1, int(.002 * sample_rate)); usable = (len(value) // hop) * hop
    env = np.sqrt((value[:usable].reshape(-1, hop) ** 2).mean(axis=1)); db = 20 * np.log10(np.maximum(env, 1e-12) / peak); center = int(seam * sample_rate / hop); window = max(3, int(.030 * sample_rate / hop))
    left = db[max(0, center - window):center]; core = db[max(0, center - 3):center + 4]
    return float(np.median(left) - core.min()) if len(left) >= 2 and len(core) else 0.0
