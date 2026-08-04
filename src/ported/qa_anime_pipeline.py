#!/usr/bin/env python3
"""QA, selección de tomas y montaje conservador para la escena anime 100_090.

Principios:
- El stem original manda sobre el BMD para decidir qué bloques tienen voz.
- No se comprime ni se recorta una cola mediante un umbral agresivo.
- Cada toma se alinea por el inicio de actividad vocal y se iguala al LUFS de
  su intervención original, no mediante una ganancia global arbitraria.
- Whisper large-v3-turbo/CUDA verifica el texto de candidatos y escena final.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "anime_100_090"
MANIFEST_PATH = WORK / "qa_scene.json"
CANDIDATE_DIR = WORK / "candidates_96_g20"
SELECTED_DIR = WORK / "selected_96_g20"
PROFILE = "96_g20"
SR = 48000
FFMPEG = ROOT.parent / "ffmpeg-master" / "ffmpeg-master" / "ffmpeg-master-latest-win64-gpl-shared" / "bin" / "ffmpeg.exe"


def load_mono(path: Path, target_sr: int = SR) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def activity_intervals(audio: np.ndarray, top_db: float = 35.0) -> list[tuple[int, int]]:
    raw = librosa.effects.split(audio, top_db=top_db, frame_length=1024, hop_length=256)
    if len(raw) == 0:
        return []
    # Une falsos huecos menores de 70 ms.
    merged: list[list[int]] = []
    max_gap = int(0.07 * SR)
    for start, end in raw:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return [(a, b) for a, b in merged]


def preserve_tail_crop(audio: np.ndarray, pre: float = 0.06, post: float = 0.20) -> np.ndarray:
    intervals = activity_intervals(audio)
    if not intervals:
        return audio.copy()
    start = max(0, intervals[0][0] - int(pre * SR))
    end = min(len(audio), intervals[-1][1] + int(post * SR))
    cropped = audio[start:end].copy()
    cropped_intervals = activity_intervals(cropped)
    if cropped_intervals:
        required_tail = int(post * SR)
        available_tail = len(cropped) - cropped_intervals[-1][1]
        if available_tail < required_tail:
            # La toma no trae cola natural (OmniVoice recorta el silencio final y
            # con el la caida del ultimo fonema). Pegar ceros directamente crea
            # una discontinuidad de fondo de escala = clic y "corte en seco"
            # (medido: paso de 20% del pico a 0.0% en una muestra, salto 0.9).
            # Se aplica un fundido coseno sobre la cola disponible antes de rellenar.
            fade = min(int(0.030 * SR), max(available_tail, 0), len(cropped) // 4)
            if fade > 0:
                curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
                cropped[-fade:] *= curve
            elif len(cropped) > int(0.030 * SR):
                # sin cola alguna: se funde el final de la propia voz
                fade = int(0.030 * SR)
                curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
                cropped[-fade:] *= curve
            cropped = np.pad(cropped, (0, required_tail - available_tail))
    return cropped


def impose_phrase_gaps(audio: np.ndarray, desired_gaps: list[float]) -> np.ndarray:
    """Reemplaza solo los silencios más grandes; nunca acelera la voz."""
    if not desired_gaps:
        return audio
    intervals = activity_intervals(audio)
    if len(intervals) < len(desired_gaps) + 1:
        return audio
    gaps = [(intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)]
    chosen = sorted(sorted(range(len(gaps)), key=lambda i: gaps[i][1] - gaps[i][0], reverse=True)[: len(desired_gaps)])
    boundaries = [gaps[i] for i in chosen]
    # Esta funcion trocea la onda y la vuelve a pegar; medido, dispara la planitud
    # espectral 2-5x y es lo que hace que la voz deje de sonar fluida. Si las
    # pausas que el modelo ya genero estan lo bastante cerca de las deseadas, se
    # devuelve el audio INTACTO. Solo se interviene si la desviacion es audible.
    TOLERANCIA = 0.120  # s
    naturales = [(right - left) / SR for left, right in boundaries]
    if all(abs(nat - des) <= TOLERANCIA for nat, des in zip(naturales, desired_gaps)):
        return audio
    pieces = []
    cursor = 0
    fade = int(0.010 * SR)
    for gap_index, (left, right) in enumerate(boundaries):
        phrase = audio[cursor:left].copy()
        if cursor and len(phrase) >= fade:
            phrase[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if len(phrase) >= fade:
            phrase[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        pieces.append(phrase)
        pieces.append(np.zeros(round(desired_gaps[gap_index] * SR), dtype=np.float32))
        cursor = right
    phrase = audio[cursor:].copy()
    if cursor and len(phrase) >= fade:
        phrase[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    pieces.append(phrase)
    return np.concatenate(pieces)


def integrated_lufs(audio: np.ndarray) -> float:
    minimum = int(0.5 * SR)
    measured = audio if len(audio) >= minimum else np.pad(audio, (0, minimum - len(audio)))
    value = pyln.Meter(SR).integrated_loudness(measured)
    return float(value)


def match_loudness(audio: np.ndarray, target_lufs: float) -> tuple[np.ndarray, float, float]:
    """Iguala LUFS a la linea inglesa y limita picos SIN deshacer la igualacion.

    Antes se reescalaba toda la senal si el pico pasaba -1.5 dBFS, lo que anulaba
    el match de loudness (se median -2.7 a -3.6 LU de desvio = saltos de volumen
    audibles entre lineas). Ese tope solo saltaba con tomas poco estiradas, que
    conservan picos mas altos. Ahora: techo a -0.5 dBFS (la mezcla final pica a
    -4 dBFS, hay headroom) y limitador de rodilla suave que solo comprime lo que
    asoma por encima del umbral, dejando intacto el grueso de la senal.
    """
    before = integrated_lufs(audio)
    gain_db = float(np.clip(target_lufs - before, -8.0, 8.0))
    result = audio * (10 ** (gain_db / 20))
    ceiling = 10 ** (-0.5 / 20)
    peak = float(np.max(np.abs(result)) + 1e-12)
    if peak > ceiling:
        # Limitador real: envolvente de ganancia SUAVIZADA, no waveshaping.
        # Un tanh muestra a muestra (lo que habia antes aqui) deforma la onda y
        # mete distorsion armonica: se midio +54% de discontinuidad y +54% de
        # flujo espectral = "artefactos raros" aunque no suene distorsionado.
        # Aqui la ganancia solo baja donde hace falta y cambia despacio (~15 ms),
        # asi que la forma de onda se conserva.
        need = np.minimum(1.0, ceiling / (np.abs(result) + 1e-12))
        win = int(0.015 * SR) | 1
        kernel = np.hanning(win); kernel /= kernel.sum()
        # minimo deslizante antes de suavizar, para no dejar picos sin atender
        pad = win // 2
        padded = np.pad(need, (pad, pad), mode="edge")
        rolling_min = np.minimum.reduce([padded[i:i + len(need)] for i in range(win)])
        envelope = np.convolve(rolling_min, kernel, mode="same")
        result = result * envelope.astype(np.float32)
    return result.astype(np.float32), before, integrated_lufs(result)


def fade_edges(audio: np.ndarray, seconds: float = 0.012) -> np.ndarray:
    n = min(round(seconds * SR), len(audio) // 2)
    if n:
        audio[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return audio


def rubberband_stretch(audio: np.ndarray, tempo: float) -> np.ndarray:
    if abs(tempo - 1.0) < 0.002:
        return audio.copy()
    with tempfile.TemporaryDirectory(prefix="p3r_rubberband_") as temp_dir:
        source = Path(temp_dir) / "in.wav"
        target = Path(temp_dir) / "out.wav"
        sf.write(source, audio, SR, subtype="PCM_24")
        # Ajustes para VOZ, no musica. Medido en test_stretch.py sobre una toma real:
        #   window=long + phase=laminar  -> -18% de transitorios (consonantes
        #     emborronadas: la energia percusiva se disuelve en tono = "robotico
        #     con eco"; se nota porque el HNR sube en vez de bajar).
        #   window=short + phase=independent -> -9% al mismo tempo, y +4% si el
        #     estirado se mantiene pequeno (<=6%).
        audio_filter = (
            f"rubberband=tempo={tempo:.8f}:pitch=1.0:transients=crisp:detector=percussive:"
            "phase=independent:window=short:smoothing=on:formant=preserved:pitchq=quality:channels=apart"
        )
        subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-af", audio_filter, "-ar", str(SR), "-ac", "1", str(target)],
            check=True,
        )
        return load_mono(target)


def fit_timing(audio: np.ndarray, target_span: float, phrase_gaps: list[float]) -> tuple[np.ndarray, float]:
    """Ajusta tempo con Rubber Band y después restituye las pausas exactas."""
    base = preserve_tail_crop(audio)
    cache: dict[float, tuple[np.ndarray, float]] = {}

    def evaluate(tempo: float) -> tuple[np.ndarray, float]:
        key = round(tempo, 6)
        if key not in cache:
            stretched = rubberband_stretch(base, tempo)
            processed = impose_phrase_gaps(stretched, phrase_gaps)
            intervals = activity_intervals(processed)
            span = (intervals[-1][1] - intervals[0][0]) / SR if intervals else 0.0
            cache[key] = (processed, span)
        return cache[key]

    # Limite de estirado: +-6%. Antes era +-18%, y ahi es donde se destruia la
    # voz (17% = -18% de transitorios). Si una toma no cabe en +-6% NO se fuerza:
    # se prefiere regenerarla / elegir otra toma que encaje de forma natural.
    LIM_LO, LIM_HI = 0.97, 1.03
    unmodified, initial_span = evaluate(1.0)
    direct = float(np.clip(initial_span / max(target_span, 0.2), LIM_LO, LIM_HI))
    evaluate(direct)
    low, high = LIM_LO, LIM_HI
    evaluate(low)
    evaluate(high)
    for _ in range(7):
        middle = (low + high) / 2
        _, span = evaluate(middle)
        if span > target_span:
            low = middle
        else:
            high = middle
    best_tempo, (best_audio, _) = min(cache.items(), key=lambda item: abs(item[1][1] - target_span))
    return preserve_tail_crop(best_audio), float(best_tempo)


# Alucinaciones tipicas de Whisper en aleman sobre silencio/musica: inventa
# creditos de subtitulado de television. No son fallos del doblaje, pero
# disparaban el WER (0.036) al alargar las colas naturales. Se eliminan antes
# de medir para que el QA no de falsos positivos.
WHISPER_ALUCINACIONES = (
    "untertitelung des zdf",
    "untertitelung im auftrag des zdf",
    "untertitel von",
    "untertitel im auftrag des zdf",
    "copyright wdr",
    "vielen dank fur die aufmerksamkeit",
)


def canonical(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ß", "ss")
    for frase in WHISPER_ALUCINACIONES:
        if frase in text:
            text = text.split(frase)[0]
    text = re.sub(r"\b(?:ge|g)e?kk?o[ -]?(?:u[ -]?)?(?:kan|kann|kahn)\b", "gekkoukan", text)
    return re.findall(r"[a-z]+", text)


def edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        new = [i]
        for j, right in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1]


def estimated_syllables(text: str) -> int:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return max(1, len(re.findall(r"[aeiouy]+", normalized)))


def resampled_contour(audio: np.ndarray, kind: str, points: int = 80) -> np.ndarray | None:
    intervals = activity_intervals(audio)
    if not intervals:
        return None
    active = audio[intervals[0][0]:intervals[-1][1]]
    if kind == "energy":
        values = librosa.feature.rms(y=active, frame_length=1024, hop_length=480)[0]
        values = np.log(values + 1e-7)
    else:
        f0, _, _ = librosa.pyin(active, fmin=70, fmax=500, sr=SR, frame_length=2048, hop_length=480)
        valid = np.isfinite(f0)
        if valid.sum() < 4:
            return None
        indices = np.arange(len(f0))
        values = np.interp(indices, indices[valid], np.log2(f0[valid]))
    if len(values) < 2 or float(np.std(values)) < 1e-7:
        return None
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, points)
    result = np.interp(new_x, old_x, values)
    return (result - result.mean()) / (result.std() + 1e-7)


def contour_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float | None:
    a = resampled_contour(left, kind)
    b = resampled_contour(right, kind)
    if a is None or b is None:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def transcribe(model: WhisperModel, path: Path | np.ndarray, language: str) -> dict:
    source = str(path)
    if isinstance(path, np.ndarray):
        source = librosa.resample(path.astype(np.float32), orig_sr=SR, target_sr=16000)
    segments, info = model.transcribe(
        source, language=language, beam_size=5, vad_filter=False,
        word_timestamps=True, condition_on_previous_text=False,
    )
    packed = []
    texts = []
    probabilities = []
    for segment in segments:
        texts.append(segment.text.strip())
        words = []
        for word in segment.words or []:
            probabilities.append(float(word.probability))
            words.append({"start": word.start, "end": word.end, "word": word.word.strip(), "probability": word.probability})
        packed.append({"start": segment.start, "end": segment.end, "text": segment.text.strip(), "words": words})
    return {
        "language": info.language,
        "text": " ".join(texts).strip(),
        "mean_word_probability": float(np.mean(probabilities)) if probabilities else 0.0,
        "segments": packed,
    }


def select_candidates(model: WhisperModel, manifest: dict) -> list[dict]:
    metadata_path = CANDIDATE_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_line = {line["id"]: line for line in manifest["lines"] if line.get("generated")}
    results = []
    SELECTED_DIR.mkdir(exist_ok=True)
    source_voice = load_mono(Path(manifest["source_channels_dir"]) / "ch5.wav")
    for line_id, line in by_line.items():
        expected = canonical(line["target_text"])
        source_audio = source_voice[round(line["source_start"] * SR):round(line["source_end"] * SR)]
        source_duration = line["source_end"] - line["source_start"]
        source_pitch = pitch_stats(source_audio)
        ranked = []
        for item in (entry for entry in metadata if entry["line_id"] == line_id):
            path = Path(item["path"])
            transcript = transcribe(model, path, "de")
            actual = canonical(transcript["text"])
            errors = edit_distance(expected, actual)
            wer = errors / max(1, len(expected))
            forbidden = sum(word in actual for word in line.get("forbidden_transcript_words", []))
            raw_transcript = transcript["text"].lower()
            # En alemán Whisper distingue con bastante consistencia "Kan" de
            # "Kahn". Priorizamos la vocal corta del nombre japonés.
            long_a_name = int(any(form in raw_transcript for form in ("kahn", "khan", "karn")))
            raw_audio = load_mono(path)
            processed_audio = impose_phrase_gaps(preserve_tail_crop(raw_audio), line.get("target_phrase_gaps", []))
            intervals = activity_intervals(processed_audio)
            active_duration = sum(b - a for a, b in intervals) / SR
            activity_span = (intervals[-1][1] - intervals[0][0]) / SR if intervals else 0.0
            words_per_second = len(expected) / max(active_duration, 0.2)
            syllables_per_second = estimated_syllables(line["target_text"]) / max(active_duration, 0.2)
            duration_ratio = activity_span / max(source_duration, 0.2)
            candidate_pitch = pitch_stats(processed_audio)
            pitch_ratio_penalty = 0.0
            if source_pitch["spread_semitones"] and candidate_pitch["spread_semitones"]:
                pitch_ratio_penalty = abs(math.log(candidate_pitch["spread_semitones"] / source_pitch["spread_semitones"]))
            energy_correlation = contour_correlation(source_audio, processed_audio, "energy")
            pitch_correlation = contour_correlation(source_audio, processed_audio, "pitch")
            phrase_count_ok = len(intervals) >= len(line.get("target_phrase_gaps", [])) + 1
            official_spelling_penalty = 0.03 * int("Gekkoukan" not in item["synthesis_text"])
            score = (
                100 * forbidden + 20 * wer + 3 * long_a_name + 5 * int(not phrase_count_ok)
                + 2 * max(0.0, syllables_per_second - 6.5)
                # Peso alto: con el estirado limitado a +-6%, lo que decide la
                # calidad es que la toma YA encaje en duracion. El WER no sirve
                # para elegir (es 0.00 en todas las tomas).
                + 12.0 * abs(math.log(max(duration_ratio, 0.1)))
                + 0.35 * pitch_ratio_penalty + official_spelling_penalty
                - 0.20 * (energy_correlation or 0.0) - 0.20 * (pitch_correlation or 0.0)
                - 0.1 * transcript["mean_word_probability"]
            )
            ranked.append({**item, "transcript": transcript["text"], "wer": wer, "forbidden": forbidden,
                           "long_a_name": long_a_name, "phrase_count_ok": phrase_count_ok,
                           "active_words_per_second": words_per_second,
                           "active_syllables_per_second": syllables_per_second,
                           "activity_span": activity_span, "source_span": source_duration,
                           "duration_ratio": duration_ratio, "source_pitch": source_pitch,
                           "candidate_pitch": candidate_pitch, "energy_correlation": energy_correlation,
                           "pitch_correlation": pitch_correlation, "score": score})
        ranked.sort(key=lambda item: item["score"])
        post_ranked = []
        for candidate in ranked:
            if candidate["wer"] > 0.02 or candidate["forbidden"] or candidate["long_a_name"]:
                continue
            processed, tempo = fit_timing(
                load_mono(Path(candidate["path"])), source_duration,
                line.get("target_phrase_gaps", []),
            )
            post_transcript = transcribe(model, processed, "de")
            post_words = canonical(post_transcript["text"])
            post_wer = edit_distance(expected, post_words) / max(1, len(expected))
            raw_post = post_transcript["text"].lower()
            post_long_a = int(any(form in raw_post for form in ("kahn", "khan", "karn")))
            post_score = candidate["score"] + 100 * post_wer + 3 * post_long_a + 0.5 * abs(tempo - 1.0)
            post_ranked.append({**candidate, "postprocess_transcript": post_transcript["text"],
                                "postprocess_wer": post_wer, "postprocess_long_a_name": post_long_a,
                                "postprocess_tempo": tempo, "postprocess_score": post_score})
        post_ranked.sort(key=lambda item: item["postprocess_score"])
        winner = post_ranked[0] if post_ranked else ranked[0]
        shutil.copy2(winner["path"], SELECTED_DIR / f"{line_id}.wav")
        results.append({"line_id": line_id, "winner": winner, "ranking": ranked,
                        "postprocess_ranking": post_ranked})
        checked_text = winner.get("postprocess_transcript", winner["transcript"])
        checked_wer = winner.get("postprocess_wer", winner["wer"])
        print(f"SELECT {line_id}: {Path(winner['path']).name} | {checked_text} | post-WER={checked_wer:.3f}")
    (WORK / f"candidate_qa_{PROFILE}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def pitch_stats(audio: np.ndarray) -> dict:
    if len(audio) < SR:
        audio = np.pad(audio, (0, SR - len(audio)))
    f0, voiced, _ = librosa.pyin(audio, fmin=70, fmax=500, sr=SR, frame_length=2048, hop_length=480)
    values = f0[np.isfinite(f0)]
    if len(values) < 3:
        return {"median_hz": None, "spread_semitones": None}
    semitones = 12 * np.log2(values / np.median(values))
    return {"median_hz": float(np.median(values)), "spread_semitones": float(np.std(semitones))}


def render(manifest: dict) -> list[dict]:
    channels = Path(manifest["source_channels_dir"])
    source_voice = load_mono(channels / "ch5.wav")
    bed_channels = [load_mono(channels / f"ch{i}.wav") for i in range(1, 5)]
    n = max(map(len, bed_channels))
    padded = [np.pad(ch, (0, n - len(ch))) for ch in bed_channels]
    bed_l = padded[0] + 0.5 * padded[2]
    bed_r = padded[1] + 0.5 * padded[3]
    voice = np.zeros(n, dtype=np.float32)
    metrics = []
    spoken = [line for line in manifest["lines"] if line.get("spoken", True)]
    for index, line in enumerate(spoken):
        source_start = float(line["source_start"])
        source_end = float(line["source_end"])
        source_crop = source_voice[max(0, round((source_start - 0.08) * SR)): min(len(source_voice), round((source_end + 0.20) * SR))]
        source_lufs = integrated_lufs(source_crop)
        audio = load_mono(WORK / line["audio"])
        audio, tempo_factor = fit_timing(
            audio, source_end - source_start, line.get("target_phrase_gaps", [])
        )
        audio, before_lufs, final_lufs = match_loudness(audio, source_lufs)
        audio = fade_edges(audio)
        intervals = activity_intervals(audio)
        target_phrase_gaps = line.get("target_phrase_gaps", [])
        measured_phrase_gaps = []
        if target_phrase_gaps and len(intervals) > 1:
            all_gaps = [((intervals[i + 1][0] - intervals[i][1]) / SR, i) for i in range(len(intervals) - 1)]
            chosen_indices = sorted(i for _, i in sorted(all_gaps, reverse=True)[:len(target_phrase_gaps)])
            measured_phrase_gaps = [(intervals[i + 1][0] - intervals[i][1]) / SR for i in chosen_indices]
        active_onset = intervals[0][0] / SR if intervals else 0.0
        planned_start = source_start
        if metrics and line.get("min_gap_after_previous") is not None:
            planned_start = max(planned_start, metrics[-1]["actual_end"] + float(line["min_gap_after_previous"]))
        place = max(0, round((planned_start - active_onset) * SR))
        end = place + len(audio)
        if end > len(voice):
            voice = np.pad(voice, (0, end - len(voice)))
            bed_l = np.pad(bed_l, (0, end - len(bed_l)))
            bed_r = np.pad(bed_r, (0, end - len(bed_r)))
        voice[place:end] += audio
        actual_start = place / SR + active_onset
        actual_end = place / SR + (intervals[-1][1] / SR if intervals else len(audio) / SR)
        metrics.append({
            "line_id": line["id"], "source_start": source_start, "source_end": source_end,
            "actual_start": actual_start, "actual_end": actual_end,
            "start_delta_ms": round(1000 * (actual_start - source_start), 1),
            "end_delta_ms": round(1000 * (actual_end - source_end), 1),
            "gap_to_next_ms": None,
            "allow_timing_shift": bool(line.get("allow_timing_shift")),
            "source_lufs": source_lufs, "before_lufs": before_lufs, "final_lufs": final_lufs,
            "loudness_delta_lu": final_lufs - source_lufs,
            "rubberband_tempo": tempo_factor,
            "syllables_per_second": estimated_syllables(line["target_text"]) / max(actual_end - actual_start, 0.2),
            "tail_guard_ms": round(1000 * (len(audio) / SR - intervals[-1][1] / SR), 1) if intervals else None,
            "max_sample_jump": float(np.max(np.abs(np.diff(audio)))) if len(audio) > 1 else 0.0,
            "target_phrase_gaps_ms": [round(1000 * gap, 1) for gap in target_phrase_gaps],
            "measured_phrase_gaps_ms": [round(1000 * gap, 1) for gap in measured_phrase_gaps],
            "phrase_gap_errors_ms": [round(1000 * (actual - target), 1) for actual, target in zip(measured_phrase_gaps, target_phrase_gaps)],
            "source_pitch": pitch_stats(source_crop), "dub_pitch": pitch_stats(audio),
        })
        print(f"MIX {line['id']}: {actual_start:.2f}-{actual_end:.2f}s | dLU={final_lufs-source_lufs:+.2f}")
    for index in range(len(metrics) - 1):
        metrics[index]["gap_to_next_ms"] = round(1000 * (metrics[index + 1]["actual_start"] - metrics[index]["actual_end"]), 1)
    length = max(len(voice), len(bed_l), len(bed_r))
    voice = np.pad(voice, (0, length - len(voice)))
    bed_l = np.pad(bed_l, (0, length - len(bed_l)))
    bed_r = np.pad(bed_r, (0, length - len(bed_r)))
    stereo = np.stack([bed_l + voice, bed_r + voice], axis=1)
    peak = float(np.max(np.abs(stereo)) + 1e-12)
    if peak > 0.99:
        stereo *= 0.99 / peak
        voice *= 0.99 / peak
    sf.write(WORK / f"escena_100_090_de_qa_{PROFILE}.wav", stereo, SR)
    sf.write(WORK / f"escena_100_090_solo_voz_qa_{PROFILE}.wav", voice, SR)
    (WORK / f"mix_metrics_{PROFILE}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def verify(model: WhisperModel, manifest: dict, mix_metrics: list[dict]) -> dict:
    solo = WORK / f"escena_100_090_solo_voz_qa_{PROFILE}.wav"
    transcript = transcribe(model, solo, "de")
    expected = " ".join(line["target_text"] for line in manifest["lines"] if line.get("spoken", True))
    expected_words = canonical(expected)
    actual_words = canonical(transcript["text"])
    wer = edit_distance(expected_words, actual_words) / max(1, len(expected_words))
    forbidden = [word for word in ["morning", "guten", "erlauben"] if word in actual_words]
    timing_failures = [m["line_id"] for m in mix_metrics if (abs(m["start_delta_ms"]) > 50 and not m.get("allow_timing_shift")) or abs(m["end_delta_ms"]) > 100 or abs(m["loudness_delta_lu"]) > 1.0 or m["syllables_per_second"] > 6.7 or m["max_sample_jump"] > 0.35 or (m["gap_to_next_ms"] is not None and m["gap_to_next_ms"] < 80)]
    tail_failures = [m["line_id"] for m in mix_metrics if m["tail_guard_ms"] is None or m["tail_guard_ms"] < 150]
    pause_failures = [m["line_id"] for m in mix_metrics if len(m["phrase_gap_errors_ms"]) != len(m["target_phrase_gaps_ms"]) or any(abs(error) > 30 for error in m["phrase_gap_errors_ms"])]
    prosody_failures = []
    for metric in mix_metrics:
        source_spread = metric["source_pitch"]["spread_semitones"]
        dub_spread = metric["dub_pitch"]["spread_semitones"]
        duration = metric["actual_end"] - metric["actual_start"]
        if duration > 1.0 and source_spread and dub_spread is not None and dub_spread < max(1.5, 0.45 * source_spread):
            prosody_failures.append(metric["line_id"])
    mixed, _ = sf.read(WORK / f"escena_100_090_de_qa_{PROFILE}.wav", always_2d=True)
    peak_dbfs = 20 * math.log10(float(np.max(np.abs(mixed))) + 1e-12)
    vowel_path = WORK / "name_vowel_alignment.json"
    name_vowel_alignment = json.loads(vowel_path.read_text(encoding="utf-8")) if vowel_path.exists() else None
    name_vowel_pass = bool(name_vowel_alignment and name_vowel_alignment.get("short_vowel_pass"))
    report = {
        "model": "faster-whisper large-v3-turbo / CUDA / float16",
        "transcript": transcript,
        "expected_text": expected,
        "wer": wer,
        "forbidden_words_found": forbidden,
        "timing_or_loudness_failures": timing_failures,
        "tail_failures": tail_failures,
        "pause_failures": pause_failures,
        "prosody_failures": prosody_failures,
        "name_vowel_alignment": name_vowel_alignment,
        "output_peak_dbfs": peak_dbfs,
        "mix_metrics": mix_metrics,
        "pass": wer <= 0.02 and not forbidden and not timing_failures and not tail_failures and not pause_failures and not prosody_failures and name_vowel_pass and peak_dbfs <= -0.05,
    }
    (WORK / f"qa_report_{PROFILE}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# QA escena 100_090", "", f"- Modelo: {report['model']}",
        f"- WER global: {wer:.3f}", f"- Palabras prohibidas: {forbidden or 'ninguna'}",
        f"- Fallos de timing/loudness: {timing_failures or 'ninguno'}", f"- Resultado automático: {'PASS' if report['pass'] else 'FAIL'}",
        f"- Fallos de cola: {tail_failures or 'ninguno'}", f"- Fallos de pausas: {pause_failures or 'ninguno'}", f"- Fallos de prosodia: {prosody_failures or 'ninguno'}", f"- Vocal corta de Gekkoukan: {'PASS' if name_vowel_pass else 'FAIL/no medida'}", f"- Pico de mezcla: {peak_dbfs:.2f} dBFS",
        "", "## Transcripción", "", transcript["text"], "", "## Líneas", "",
    ]
    for metric in mix_metrics:
        summary.append(f"- {metric['line_id']}: inicio {metric['start_delta_ms']:+.1f} ms, final {metric['end_delta_ms']:+.1f} ms, ΔLU {metric['loudness_delta_lu']:+.2f}, pausa siguiente {metric['gap_to_next_ms']} ms")
    (WORK / f"qa_report_{PROFILE}.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"QA {'PASS' if report['pass'] else 'FAIL'} | WER={wer:.3f} | forbidden={forbidden} | timing={timing_failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = None
    if args.select or args.verify:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from asr_gpu import cargar_modelo
        model = cargar_modelo("large-v3-turbo")
    if args.select:
        select_candidates(model, manifest)
    metrics = render(manifest) if args.render else json.loads((WORK / f"mix_metrics_{PROFILE}.json").read_text(encoding="utf-8"))
    if args.verify:
        verify(model, manifest, metrics)


if __name__ == "__main__":
    main()
