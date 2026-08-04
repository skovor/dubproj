#!/usr/bin/env python3
"""Valida, codifica y despliega los assets compartidos de frase completa.

El despliegue es transaccional: todos los HCA se construyen y verifican en
staging antes de tocar el mod. Los cues preexistentes se respaldan.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROWS = ROOT / "shared_physical_assets.jsonl"
SOURCE = ROOT / "shared_production"
EXACT_BANKS = ROOT / "shared_exact_banks"
STAGING = ROOT / "shared_ryo_staging"
REPORT = ROOT / "shared_ryo_deploy_report.json"
MOD = Path(r"C:\Users\juand\Desktop\Reloaded-II\Mods\p3r.doblaje.aleman")
RYO = MOD / "ryo" / "P3R"
FFMPEG = Path(
    r"C:\Users\juand\Desktop\moddeutsch\ffmpeg7"
    r"\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin\ffmpeg.exe"
)
VGAUDIO = Path(
    r"C:\Users\juand\Desktop\moddeutsch\packaging_tools\VGAudioCli.exe"
)
VGMSTREAM = Path(
    r"C:\Users\juand\Desktop\moddeutsch\vgmstream\vgmstream-cli.exe"
)


def run(command: list[Path | str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def wav_contract(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as stream:
        return (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getnframes(),
        )


def hca_contract(path: Path) -> tuple[int, int, int] | None:
    result = run([VGMSTREAM, "-m", path])
    text = result.stdout + "\n" + result.stderr
    if result.returncode:
        return None
    import re

    rate = re.search(r"sample rate:\s*(\d+)", text)
    channels = re.search(r"channels:\s*(\d+)", text)
    samples = re.search(r"(?:stream total samples|play duration):\s*(\d+)", text)
    if not (rate and channels and samples):
        return None
    return int(rate.group(1)), int(channels.group(1)), int(samples.group(1))


def read_ready_rows() -> list[dict]:
    rows = [
        json.loads(line)
        for line in ROWS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ready = [row for row in rows if row["asset_action"] == "READY_TO_SYNTHESIZE"]
    if len(ready) != 1_269:
        raise RuntimeError(f"Contrato roto: {len(ready)} assets != 1269")
    return ready


def bank_for_variant(variant: str) -> str:
    manifest = EXACT_BANKS / variant / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not payload.get("validated"):
        raise RuntimeError(f"Banco exacto no validado: {variant}")
    return str(payload["bank"])


def source_for(row: dict, promote_failed: bool) -> tuple[Path, bool]:
    variant, awb = row["physical_key"].split(":", 1)
    normal = SOURCE / variant / f"{awb}.wav"
    if normal.is_file():
        return normal, False
    failed = SOURCE / "_review_failed" / (
        row["physical_key"].replace(":", "__") + ".wav"
    )
    if promote_failed and failed.is_file():
        normal.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(failed, normal)
        return normal, True
    raise FileNotFoundError(f"Falta salida para {row['physical_key']}")


def encode(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p3r_shared_hca_") as temp_name:
        converted = Path(temp_name) / "contract.wav"
        conversion = run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source,
                "-ar",
                "48000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                converted,
            ]
        )
        if conversion.returncode or not converted.is_file():
            raise RuntimeError(f"ffmpeg falló: {conversion.stderr[-500:]}")
        expected = wav_contract(converted)
        encoded = run([VGAUDIO, "-c", converted, "-o", target])
        if encoded.returncode or not target.is_file():
            diagnostic = (encoded.stdout + "\n" + encoded.stderr)[-800:]
            raise RuntimeError(
                f"VGAudio falló para {source} (bytes={source.stat().st_size}): "
                f"{diagnostic}"
            )
        got = hca_contract(target)
        if got != expected:
            raise RuntimeError(
                f"Contrato HCA incorrecto para {source.name}: {got} != {expected}"
            )
        return {"rate": got[0], "channels": got[1], "samples": got[2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="publicar en el mod después de construir y validar staging",
    )
    parser.add_argument(
        "--promote-best-after-max4",
        action="store_true",
        help="usar el mejor candidato guardado si los cuatro intentos fallaron",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reutilizar HCA ya validados en staging",
    )
    args = parser.parse_args()

    rows = read_ready_rows()
    routes: dict[tuple[str, int], dict] = {}
    promoted = []
    for row in rows:
        variant, _ = row["physical_key"].split(":", 1)
        bank = bank_for_variant(variant)
        source, was_promoted = source_for(row, args.promote_best_after_max4)
        if was_promoted:
            promoted.append(row["physical_key"])
        for cue in row["resolved_cue_ids"]:
            key = (bank, int(cue))
            if key in routes:
                raise RuntimeError(f"Colisión banco/cue: {key}")
            routes[key] = {"row": row, "source": source}

    if len(routes) != 1_274:
        raise RuntimeError(f"Contrato roto: {len(routes)} rutas != 1274")
    if STAGING.exists() and not args.resume:
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    encoded_assets: dict[str, tuple[Path, dict]] = {}
    details = []
    for number, ((bank, cue), item) in enumerate(sorted(routes.items()), 1):
        physical_key = item["row"]["physical_key"]
        if physical_key not in encoded_assets:
            target = STAGING / "_encoded_assets" / (
                physical_key.replace(":", "__") + ".hca"
            )
            existing_contract = hca_contract(target) if target.is_file() else None
            if (
                args.resume
                and existing_contract is not None
                and existing_contract[0] == 48_000
                and existing_contract[1] == 1
            ):
                contract = {
                    "rate": existing_contract[0],
                    "channels": existing_contract[1],
                    "samples": existing_contract[2],
                }
            else:
                contract = encode(item["source"], target)
            encoded_assets[physical_key] = (target, contract)
        encoded, contract = encoded_assets[physical_key]
        destination = STAGING / f"{bank}.acb" / f"{cue}.hca"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(encoded, destination)
        details.append(
            {
                "physical_key": physical_key,
                "bank": bank,
                "cue": cue,
                "source": str(item["source"]),
                "staged": str(destination),
                "contract": contract,
            }
        )
        if number % 100 == 0:
            print(f"[{number}/{len(routes)}] rutas codificadas", flush=True)

    for bank_dir in STAGING.glob("*.acb"):
        (bank_dir / "config.yaml").write_text(
            "volume: 1.0\nuse_player_volume: false\n",
            encoding="utf-8",
        )

    deployed = False
    backup = None
    if args.deploy:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MOD / "_backup_shared_ryo" / stamp
        for bank_dir in sorted(STAGING.glob("*.acb")):
            destination = RYO / bank_dir.name
            if destination.exists():
                old_files = list(destination.glob("*.hca"))
                if old_files:
                    old_backup = backup / bank_dir.name
                    old_backup.mkdir(parents=True, exist_ok=True)
                    for old in old_files:
                        shutil.copy2(old, old_backup / old.name)
            destination.mkdir(parents=True, exist_ok=True)
            for item in bank_dir.iterdir():
                shutil.copy2(item, destination / item.name)
        deployed = True

    payload = {
        "assets": len(encoded_assets),
        "routes": len(routes),
        "banks": len({bank for bank, _ in routes}),
        "promoted_best_after_max4": promoted,
        "deployed": deployed,
        "backup": str(backup) if backup else None,
        "details": details,
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "details"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
