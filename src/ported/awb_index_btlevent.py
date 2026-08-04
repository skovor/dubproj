#!/usr/bin/env python3
"""Indice evento -> banco de voz .awb para la familia BtlEvent (dialogo real
de cutscenes de combate, BMD_BtlEvent700.. BMD_BtlEvent802).

A diferencia de CmmuNPC/Facility/Dungeon (pool compartido, ver
resuelve_pool_compartido.py), BtlEvent SI mantiene la misma convencion 1
banco = 1 evento que la narrativa: `BtlEvent<N>.awb` bajo `Stream/en/`
corresponde exactamente a `BMD_BtlEvent<N>`. Confirmado por conteo: 25 bancos
ingleses encontrados para 25 de los 28 eventos de battle_lines.jsonl (los 3
restantes son BtlSupportCommon/Fuka/Mituru, que SI son pool compartido y
caen dentro de Voice_Dungeon).

Salida: awb_index_btlevent.json -- { "BtlEvent700": {"pak":..., "internal":...}, ... }
Mismo formato que awb_index.json para que prod_dub.py lo consuma igual.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPAK = Path(r"C:\Users\juand\Desktop\moddeutsch\repak\repak.exe")
AES_KEY = "0x92BADFE2921B376069D3DE8541696D230BA06B5E4320084DD34A26D117D2FFEE"
PAKS_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\P3R\P3R\Content\Paks")
PAK_CHUNKS = ["pakchunk4-WindowsNoEditor.pak", "pakchunk5-WindowsNoEditor.pak"]
OUT = Path(__file__).resolve().parent / "awb_index_btlevent.json"

BTLEVENT_RE = re.compile(r"Stream/(?:Astrea/)?en/(BtlEvent\d+)\.awb$")


def _list_pak(pak: Path) -> list[str]:
    out = subprocess.run([str(REPAK), "-a", AES_KEY, "list", str(pak)],
                          capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def build() -> dict:
    index: dict[str, dict] = {}
    for name in PAK_CHUNKS:
        pak = PAKS_DIR / name
        for line in _list_pak(pak):
            line = line.strip()
            m = BTLEVENT_RE.search(line)
            if not m:
                continue
            index[m.group(1)] = {"pak": str(pak), "internal": line}
    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} eventos BtlEvent con banco de voz EN -> {OUT.name}")
    return index


if __name__ == "__main__":
    build()
