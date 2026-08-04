#!/usr/bin/env python3
"""Indice evento -> banco de voz .awb, construido con repak.

El audio NUNCA sale de retoc (retoc solo toca IoStore/.utoc; ver CLAUDE.md,
seccion "Herramientas instaladas"): los bancos .awb viven en los .pak LEGACY
hermanos y se listan/extraen con repak.exe.

Cubre el patron principal `Voice_Event_<event>.awb` bajo `Stream/en/`, que en
pakchunk4 (50 eventos) y pakchunk5 (775 eventos) suma 825 eventos con voz
inglesa: familias Main/Cmmu/Extr/Fild/Qest (BMD_Event_*, lo que build_corpus.py
y build_misc_corpus.py llaman "narrativa"). Antes solo un punado de eventos
Main_1XX tenian su ref extraida a mano en un workspace fijo; este indice cubre
los 825 de una vez, construido leyendo el listado real de los paks (no
supuesto por convencion de nombre).

También cubre `Stream/Astrea/en/Voice_Event_<event>.awb`: un lote de eventos
(familias Extr_5xx/Fild_3xx) cuyo AUDIO vive bajo ese subfolder "Astrea"
aunque su TEXTO (BMD) está en el árbol Xrd777 normal -- build_misc_corpus.py
excluye la carpeta Astrea al extraer texto asumiendo que es contenido
duplicado, pero para audio no lo es: es la única ubicación real de esas
voces. Encontrado al
investigar por qué 248 eventos narrativos no tenían banco indexado (91 de
ellos resultaron estar aquí).

Bancos de voz con OTROS esquemas de nombre, encontrados pero NO indexados aqui
a proposito porque su mapeo evento<->banco<->stream no esta verificado por
transcripcion (la regla del proyecto: "no se da por buena ninguna regla de
mapeo"): `Voice_NPC_*.awb` (69, probablemente NPC_*/CmmuNPC_*/kfev*_NPC_*),
`Voice_Facility_*.awb` (5, tiendas), `BtlEvent###.awb` + `btl_pc/btl_boss/
btl_ast/mob/mob_mis` (bancos de batalla, compartidos entre muchos eventos por
tipo de personaje en vez de 1 banco por evento BMD). Verificar cada esquema
por transcripcion antes de confiar en un indice para ellos.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPAK = Path(r"C:\Users\juand\Desktop\moddeutsch\repak\repak.exe")
AES_KEY = "0x92BADFE2921B376069D3DE8541696D230BA06B5E4320084DD34A26D117D2FFEE"
PAKS_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\P3R\P3R\Content\Paks")
# Los bancos de voz EN viven bajo Stream/en/ SOLO en los chunks de parche 4 y 5.
# 0/1 tienen las MISMAS rutas pero SIN carpeta de idioma -- contenido distinto
# (hash distinto, confirmado con `repak hash-list`), no una copia redundante.
PAK_CHUNKS = ["pakchunk4-WindowsNoEditor.pak", "pakchunk5-WindowsNoEditor.pak"]
OUT = Path(__file__).resolve().parent / "awb_index.json"

VOICE_EVENT_RE = re.compile(r"Stream/(?:Astrea/)?en/Voice_Event_(.+)\.awb$")


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
            m = VOICE_EVENT_RE.search(line)
            if not m:
                continue
            event = m.group(1)
            # Si el evento aparece en ambos chunks, gana el ultimo escaneado
            # (pakchunk5 tras pakchunk4 = convencion UE de parche posterior).
            index[event] = {"pak": str(pak), "internal": line}
    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} eventos con banco de voz EN -> {OUT.name}")
    return index


if __name__ == "__main__":
    build()
