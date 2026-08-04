#!/usr/bin/env python3
"""Reaudita los 235 WAV exactos después de reparaciones transparentes."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT / "exact_ryo_deploy_report.json"
OUTPUT = ROOT / "exact_generation_technical_audit_final.json"


def longest_true_run(mask: np.ndarray) -> int:
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return int(np.max(indices[1::2] - indices[::2], initial=0))


def inspect(unit_id: str, source: Path) -> dict:
    with wave.open(str(source), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if width != 2:
        raise RuntimeError(f"{source}: se esperaba PCM16, llegó width={width}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    normalized = audio / 32768.0
    absolute = np.abs(normalized)
    peak = float(np.max(absolute, initial=0.0))
    clipped = absolute >= (32760.0 / 32768.0)
    clipping_pct = float(np.mean(clipped) * 100.0) if len(clipped) else 0.0
    longest = longest_true_run(clipped)
    active = absolute[absolute >= 10 ** (-45 / 20)]
    active_rms = float(np.sqrt(np.mean(active**2))) if len(active) else 0.0
    tail = normalized[-min(len(normalized), int(rate * 0.08)) :]
    tail_rms = float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.0
    tail_ratio = tail_rms / max(active_rms, 1e-9)
    passed = bool(
        len(normalized)
        # Los picos aislados a 0 dBFS no son clipping audible. El gate
        # canónico rechaza mesetas de ocho muestras o más, o una densidad
        # anormal de muestras saturadas; no normaliza masivamente el corpus.
        and longest < 8
        and clipping_pct <= 0.15
        and tail_ratio < 0.10
    )
    return {
        "id": unit_id.removeprefix("VU_"),
        "status": "PASS" if passed else "REVIEW",
        "seconds": len(normalized) / rate,
        "peak": peak,
        "longest_clipped_run": longest,
        "clipping_pct": clipping_pct,
        "tail_ratio": tail_ratio,
    }


def main() -> None:
    deploy = json.loads(DEPLOY.read_text(encoding="utf-8"))
    details = deploy["details"]
    if len(details) != 234 or len({row["unit_id"] for row in details}) != 234:
        raise RuntimeError("El manifiesto exacto desplegado no contiene 234 rutas únicas")
    rows = [
        inspect(row["unit_id"], Path(row["source"]))
        for row in details
    ]
    # La referencia PoolFallback es segura como audio pero no tiene destino físico.
    orphan = next(
        row for row in deploy["excluded"]
        if row["unit_id"] == "VU_PoolFallback_dungeon_L707"
    )
    source = ROOT / "produccion" / "PoolFallback_dungeon_L707.wav"
    if not source.is_file():
        raise RuntimeError(f"Falta referencia exacta huérfana: {source}")
    rows.append(inspect(orphan["unit_id"], source))
    if len(rows) != 235:
        raise RuntimeError("La reauditoría no cubrió 235 WAV exactos")
    OUTPUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "audited": len(rows),
        "pass": sum(row["status"] == "PASS" for row in rows),
        "review": sum(row["status"] != "PASS" for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
