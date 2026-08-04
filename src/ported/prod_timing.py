#!/usr/bin/env python3
"""Corrección de longitud para líneas CINEMÁTICAS de P3R.

Regla acordada con el usuario (validada oyendo A/B en líneas cinemáticas reales):

  * Solo se corrige la longitud de líneas CINEMÁTICAS (vp sin bup): esas van
    clavadas a la secuencia y se cortan si la voz alemana se pasa. Las líneas de
    caja de diálogo NO se tocan (la caja espera a la voz) -> ver cine_index.json
    y build_cine_index.py.
  * Método base: `duration=` nativo de OmniVoice (genera a la longitud objetivo,
    natural, sin estirar).
  * Ventana aceptable del fin de voz alemán respecto al fin de voz inglés:
        [-0.35 s (corto),  +0.35 s (largo)]
    Dentro de la ventana -> se deja la toma de `duration=` tal cual. Un exceso
    leve cabe en el hueco natural entre frases, y quedarse un poco corto solo
    deja silencio/labios al final; ambos son inofensivos.
  * Fuera de la ventana -> rescate raro con atempo DIRECTO (elección del usuario:
    simple y determinista, sin re-roll), asimétrico:
        - se pasó (> +0.35): comprimir hasta caer en +0.35.
        - se quedó corto (< -0.35): estirar hasta -0.35.
    Toda salida vuelve a QA general + cola; una toma deformada se regenera.

Este módulo no carga el modelo: recibe la toma ya generada. El orquestador
(prod_dub.py) es quien llama a `model.generate(..., duration=...)` y luego pasa
por aquí.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Ventana aceptable del desajuste fin_voz_DE - fin_voz_EN (segundos).
UNDER_TOL = 0.35   # cuánto puede quedarse CORTA la voz alemana
OVER_TOL = 0.35    # cuánto puede PASARSE (se come el hueco natural entre frases)


def speech_end(y: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    """Último instante por encima de floor_db respecto al pico."""
    p = float(np.abs(y).max())
    if p <= 0:
        return 0.0
    idx = np.where(np.abs(y) > p * 10 ** (floor_db / 20.0))[0]
    return float(idx[-1]) / sr if len(idx) else 0.0


def atempo_chain(ratio: float) -> str:
    """Cadena de atempo: ffmpeg solo acepta 0.5-2.0 por etapa."""
    stages, r = [], ratio
    while r > 2.0:
        stages.append(2.0)
        r /= 2.0
    while r < 0.5:
        stages.append(0.5)
        r /= 0.5
    stages.append(r)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def within_window(diff: float) -> bool:
    """diff = fin_voz_DE - fin_voz_EN."""
    return -UNDER_TOL <= diff <= OVER_TOL


def target_end_for(diff: float, voz_en: float) -> float:
    """A qué fin de voz llevar la toma cuando está fuera de ventana."""
    if diff > OVER_TOL:
        return voz_en + OVER_TOL      # se pasó: comprimir al borde del hueco
    return voz_en - UNDER_TOL         # se quedó corta: estirar al borde inferior


def correct_length(
    audio: np.ndarray,
    sr: int,
    voz_en: float,
    ffmpeg: str | Path,
    tmpdir: str | Path,
    max_ratio_deviation: float | None = None,
    under_tol: float = UNDER_TOL,
    over_tol: float = OVER_TOL,
) -> tuple[np.ndarray, dict]:
    """Aplica el rescate atempo si la toma cae fuera de la ventana.

    `voz_en` = fin de voz de la referencia inglesa (el objetivo). `audio` = la
    toma alemana ya generada con `duration=` (idealmente con el silencio de
    entrada ya recortado por el orquestador). Devuelve (audio_corregido, info).
    """
    end = speech_end(audio, sr)
    diff = end - voz_en
    info = {"voz_en": round(voz_en, 3), "voz_de": round(end, 3),
            "diff": round(diff, 3), "metodo": "duration"}
    if (-float(under_tol) <= diff <= float(over_tol)) or end <= 0:
        return audio, info

    target = (
        voz_en + float(over_tol)
        if diff > float(over_tol)
        else voz_en - float(under_tol)
    )
    if target <= 0.05:
        return audio, info                      # nada sensato que estirar
    ratio = end / target                        # >1 comprime, <1 estira
    if (
        max_ratio_deviation is not None
        and abs(ratio - 1.0) > max_ratio_deviation
    ):
        info.update({
            "metodo": "atempo_refused_ratio",
            "ratio": round(ratio, 4),
            "max_ratio_deviation": max_ratio_deviation,
        })
        return audio, info
    tmp_in = Path(tmpdir) / "_tc_in.wav"
    tmp_out = Path(tmpdir) / "_tc_out.wav"
    sf.write(str(tmp_in), audio, sr)
    subprocess.run([str(ffmpeg), "-y", "-i", str(tmp_in), "-af",
                    atempo_chain(ratio), "-loglevel", "error", str(tmp_out)],
                   capture_output=True)
    ya, sra = sf.read(str(tmp_out), always_2d=False)
    if ya.ndim > 1:
        ya = ya.mean(axis=1)
    ya = np.asarray(ya, float)
    info.update({"metodo": "atempo", "ratio": round(ratio, 4),
                 "voz_de_corregida": round(speech_end(ya, sra), 3)})
    return ya, info


def fit_speech_end(
    audio: np.ndarray,
    sr: int,
    target_end: float,
    ffmpeg: str | Path,
    tmpdir: str | Path,
    max_ratio_deviation: float = 0.20,
) -> tuple[np.ndarray, dict]:
    """Compress a synthetic take just enough to fit a later source onset.

    This is deliberately separate from ``correct_length``: the normal
    cinematic tolerance can be satisfied while the complete active span still
    cannot be shifted to the English onset inside a short cue.  The caller must
    pass only the synthetic body, never a preserved effort or a movie mix.
    """
    end = speech_end(audio, sr)
    info = {
        "voz_de": round(end, 3),
        "target_end": round(float(target_end), 3),
        "metodo": "onset_fit_not_needed",
    }
    if end <= 0 or target_end <= 0.05 or end <= target_end:
        return audio, info
    ratio = end / target_end
    if abs(ratio - 1.0) > max_ratio_deviation:
        info.update({
            "metodo": "onset_fit_refused_ratio",
            "ratio": round(ratio, 4),
            "max_ratio_deviation": max_ratio_deviation,
        })
        return audio, info
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp_in = tmpdir / "_onset_fit_in.wav"
    tmp_out = tmpdir / "_onset_fit_out.wav"
    sf.write(str(tmp_in), audio, sr)
    completed = subprocess.run(
        [
            str(ffmpeg), "-y", "-i", str(tmp_in), "-af",
            atempo_chain(ratio), "-loglevel", "error", str(tmp_out),
        ],
        capture_output=True,
    )
    if completed.returncode or not tmp_out.is_file():
        info.update({
            "metodo": "onset_fit_ffmpeg_failed",
            "ratio": round(ratio, 4),
            "returncode": completed.returncode,
        })
        return audio, info
    corrected, corrected_sr = sf.read(str(tmp_out), always_2d=False)
    if corrected.ndim > 1:
        corrected = corrected.mean(axis=1)
    corrected = np.asarray(corrected, float)
    info.update({
        "metodo": "atempo_onset_fit",
        "ratio": round(ratio, 4),
        "voz_de_corregida": round(speech_end(corrected, corrected_sr), 3),
    })
    return corrected, info


__all__ = ["correct_length", "speech_end", "atempo_chain", "within_window",
           "UNDER_TOL", "OVER_TOL"]
