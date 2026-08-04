#!/usr/bin/env python3
"""Empaqueta las líneas ya generadas (produccion/*.wav) en un .pak de mod
listo para copiar a la carpeta Paks del juego.

Mecanismo (validado a mano línea por línea antes de escribir esto -- ver
HANDOFF_AGENTE.md sección 5 para el detalle completo de la investigación):

  1. Extraer el AWB original desde su PAK legacy.
  2. Leer directamente su tabla AFS2 y extraer los HCA sin depender de
     AcbEditor. Se preservan versión, anchos de offsets/IDs, IDs, alineación
     y subkey del contenedor original.
  3. Codificar cada WAV alemán frame-exacto a HCA con la misma versión,
     cifrado y contrato del slot, y reemplazar el HCA del índice que
     corresponda (stream_index del proyecto es 1-indexed).
  4. Reconstruir AFS2 directamente conservando la representación original.
  5. Verificar conteo, sample rate, canales, muestras, versión y cifrado.
  6. Copiar el .awb reconstruido a un árbol de carpetas que imite la ruta
     interna original, y empaquetar todo con `repak pack` a un .pak final.

OJO: este script asume que produccion/*.wav YA EXISTE (prod_dub.py --run ya
corrió). Si esa carpeta está vacía no hay nada que empaquetar -- se
detecta y se avisa, no se genera un .pak vacío.

NUNCA toca los 2 openings + 2 endings: esos son videos .usm puros, nunca
entraron a ningún corpus de este proyecto, así que ni siquiera aparecen
aquí -- no hace falta un filtro explícito, es una imposibilidad estructural
(no hay ninguna entrada de indice que apunte a ellos), pero se deja esta
nota por si el mapeo cambia en el futuro.

Uso:
  python empaqueta_mod.py                    # arma el .pak con todo lo generado
  python empaqueta_mod.py --solo-preparar     # solo hasta el paso 6 (deja los
                                              # .awb reconstruidos en awb_rebuilt/,
                                              # sin armar el .pak final) -- util
                                              # para revisar antes del paso final
"""
from __future__ import annotations

import argparse
import array
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "produccion"
REBUILD_DIR = ROOT / "awb_rebuilt"          # .awb reconstruidos, listos para el pak
MOD_TREE_DIR = ROOT / "mod_pak_tree"        # arbol de carpetas que imita el pak
FINAL_PAK = ROOT / "P3R_doblaje_aleman_pakchunk900-WindowsNoEditor.pak"

REPAK = Path(r"C:\Users\juand\Desktop\moddeutsch\repak\repak.exe")
RETOC = Path(r"C:\Users\juand\Desktop\moddeutsch\retoc\retoc.exe")
VGMSTREAM = Path(r"C:\Users\juand\Desktop\moddeutsch\vgmstream\vgmstream-cli.exe")
FFMPEG = Path(
    r"C:\Users\juand\Desktop\moddeutsch\ffmpeg7"
    r"\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin\ffmpeg.exe"
)
AES_KEY = "0x92BADFE2921B376069D3DE8541696D230BA06B5E4320084DD34A26D117D2FFEE"
PAKS_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\P3R\P3R\Content\Paks")
PAK_CHUNKS_LEGACY = [f"pakchunk{i}-WindowsNoEditor.pak" for i in range(6)]
UTOC_CHUNKS = [f"pakchunk{i}-WindowsNoEditor.utoc" for i in range(6)]

# Herramientas de empaquetado externas (VGAudioCli, AcbEditor) -- ver
# HANDOFF_AGENTE.md seccion 7 para las URLs de descarga si no estan aqui.
TOOLS_DIR = Path(r"C:\Users\juand\Desktop\moddeutsch\packaging_tools")
VGAUDIO = TOOLS_DIR / "VGAudioCli.exe"
ACBEDITOR = TOOLS_DIR / "PersonaVCE" / "AcbEditor.exe"
CRICODECS = (
    ROOT / "tools" / "cricodecs-1.1.2-cli-windows-x86_64" / "cricodecs.exe"
)

sys.path.insert(0, str(ROOT))
import prod_dub as pd  # reutiliza _load_awb_index, _load_btlevent_index, etc.


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, **kw)


def find_cuesheet_uasset(bank_name: str) -> str | None:
    """Busca en el listado IoStore de pakchunk4/5 el .uasset del cue sheet
    de `bank_name` (sin extension), sin asumir categoria fija."""
    pattern = re.compile(re.escape(bank_name) + r"\.uasset$")
    for chunk in UTOC_CHUNKS:
        utoc = PAKS_DIR / chunk
        if not utoc.exists():
            continue
        r = _run([RETOC, "-a", AES_KEY, "list", "--path", str(utoc)], text=True)
        for line in r.stdout.splitlines():
            line = line.strip()
            if "CueSheet" in line and pattern.search(line):
                return line
    return None


def carve_acb(uexp_path: Path, dst_acb: Path) -> bool:
    data = uexp_path.read_bytes()
    idx = data.find(b"@UTF")
    if idx < 0:
        return False
    size = struct.unpack(">I", data[idx + 4:idx + 8])[0]
    dst_acb.write_bytes(data[idx:idx + 8 + size])
    return True


def extract_cuesheet(bank_name: str, work_dir: Path) -> Path | None:
    """retoc to-legacy -f <bank> -> carvea el .acb. Devuelve la ruta del
    .acb carveado, o None si no se encontro el cue sheet."""
    acb_out = work_dir / f"{bank_name}.acb"
    if acb_out.exists():
        return acb_out
    uasset_path = find_cuesheet_uasset(bank_name)
    if uasset_path is None:
        return None
    legacy_out = work_dir / "_legacy_tmp"
    legacy_out.mkdir(parents=True, exist_ok=True)
    _run([RETOC, "-a", AES_KEY, "to-legacy", "-f", bank_name, PAKS_DIR, legacy_out])
    matches = list(legacy_out.rglob(f"{bank_name}.uexp"))
    if not matches:
        return None
    if not carve_acb(matches[0], acb_out):
        return None
    return acb_out


def acb_editor_run(arg_path: Path, timeout_s: int = 60) -> bool:
    """Invoca AcbEditor.exe como si `arg_path` se hubiera arrastrado sobre
    el ejecutable (dump si es .acb, repack si es una carpeta). Es una app
    WinForms pero no bloquea: termina sola (probado, exit code 0 en
    segundos). Si no termina en el timeout se mata el proceso."""
    p = subprocess.Popen([str(ACBEDITOR), str(arg_path)])
    try:
        p.wait(timeout=timeout_s)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        p.kill()
        return False


def _audio_contract(
    path: Path, stream_index: int | None = None,
) -> tuple[int, int, int] | None:
    """Return (sample_rate, channels, total_samples) from vgmstream."""
    command = [VGMSTREAM]
    if stream_index is not None:
        command += ["-s", str(stream_index)]
    command += ["-m", str(path)]
    result = _run(command, text=True)
    metadata = (result.stdout or "") + (result.stderr or "")
    rate = re.search(r"sample rate:\s*(\d+)\s*Hz", metadata)
    channels = re.search(r"channels:\s*(\d+)", metadata)
    samples = re.search(r"stream total samples:\s*(\d+)", metadata)
    if result.returncode or not rate or not channels or not samples:
        return None
    return int(rate.group(1)), int(channels.group(1)), int(samples.group(1))


def _hca_cipher_type(path: Path) -> int | None:
    """Read the HCA ciph chunk, whose tag may have CRI's high-bit mask."""
    data = path.read_bytes()[:0x200]
    unmasked = bytes(value & 0x7F for value in data)
    offset = unmasked.find(b"ciph")
    if offset < 0 or offset + 6 > len(data):
        return None
    return int.from_bytes(data[offset + 4:offset + 6], "big")


def _hca_header_version(path: Path) -> int | None:
    data = path.read_bytes()[:6]
    if len(data) < 6 or bytes(value & 0x7F for value in data[:4]) != b"HCA\x00":
        return None
    return int.from_bytes(data[4:6], "big") >> 8


def _wav_active_span(path: Path) -> tuple[int, int, int, int]:
    """Return first/last active frame, total frames and rate for PCM16 WAV."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        total = source.getnframes()
        width = source.getsampwidth()
        payload = source.readframes(total)
    if width != 2 or not payload or total == 0:
        return 0, total, total, rate
    values = array.array("h")
    values.frombytes(payload)
    peak = max(abs(value) for value in values)
    threshold = max(2, int(peak * 0.01))  # relative -40 dB activity gate
    first = None
    last = None
    for frame in range(total):
        begin = frame * channels
        active = any(
            abs(values[begin + channel]) >= threshold
            for channel in range(channels)
        )
        if active:
            if first is None:
                first = frame
            last = frame + 1
    if first is None or last is None:
        return 0, total, total, rate
    return first, last, total, rate


def _wav_contract(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as source:
        return (
            source.getframerate(),
            source.getnchannels(),
            source.getnframes(),
        )


def _awb_subkey(path: Path) -> int:
    """AFS2 stores the HCA subkey as little-endian uint16 at 0x0E."""
    header = path.read_bytes()[:0x10]
    if len(header) < 0x10 or header[:4] != b"AFS2":
        raise ValueError(f"not an AFS2 AWB: {path}")
    return int.from_bytes(header[0x0E:0x10], "little")


def _parse_awb(path: Path) -> tuple[dict, list[int], list[bytes]]:
    """Parse AFS2 without normalizing any of its container contract fields."""
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != b"AFS2":
        raise ValueError(f"not an AFS2 AWB: {path}")
    version = data[4]
    offset_size = data[5]
    id_size = int.from_bytes(data[6:8], "little")
    count = int.from_bytes(data[8:12], "little")
    alignment = int.from_bytes(data[12:14], "little")
    subkey = int.from_bytes(data[14:16], "little")
    if offset_size not in (2, 4, 8) or id_size not in (1, 2, 4, 8):
        raise ValueError("unsupported AFS2 table width")
    if not count or not alignment:
        raise ValueError("invalid empty/unaligned AFS2")
    cursor = 16
    ids = []
    for _ in range(count):
        ids.append(int.from_bytes(data[cursor:cursor + id_size], "little"))
        cursor += id_size
    offsets = []
    for _ in range(count + 1):
        offsets.append(
            int.from_bytes(data[cursor:cursor + offset_size], "little")
        )
        cursor += offset_size
    payloads = []
    for index in range(count):
        begin = ((offsets[index] + alignment - 1) // alignment) * alignment
        end = offsets[index + 1]
        if end < begin or end > len(data):
            raise ValueError(f"invalid AFS2 entry {index}: {begin}:{end}")
        payloads.append(data[begin:end])
    contract = {
        "version": version,
        "offset_size": offset_size,
        "id_size": id_size,
        "alignment": alignment,
        "subkey": subkey,
    }
    return contract, ids, payloads


def _write_awb_preserving_contract(
    path: Path, contract: dict, ids: list[int], payloads: list[bytes],
) -> None:
    """Build AFS2 while retaining the source header/table representation."""
    if len(ids) != len(payloads) or not payloads:
        raise ValueError("AWB IDs/payload count mismatch")
    count = len(payloads)
    offset_size = contract["offset_size"]
    id_size = contract["id_size"]
    alignment = contract["alignment"]
    header_raw = 16 + id_size * count + offset_size * (count + 1)
    current = ((header_raw + alignment - 1) // alignment) * alignment
    actual_offsets = [current]
    stored_offsets = [header_raw]
    for index, payload in enumerate(payloads):
        raw_end = current + len(payload)
        stored_offsets.append(raw_end)
        current = raw_end
        if index + 1 < count:
            current = ((current + alignment - 1) // alignment) * alignment
        actual_offsets.append(current)
    output = bytearray(actual_offsets[-1])
    output[:4] = b"AFS2"
    output[4] = contract["version"]
    output[5] = offset_size
    output[6:8] = id_size.to_bytes(2, "little")
    output[8:12] = count.to_bytes(4, "little")
    output[12:14] = alignment.to_bytes(2, "little")
    output[14:16] = contract["subkey"].to_bytes(2, "little")
    cursor = 16
    for wave_id in ids:
        output[cursor:cursor + id_size] = wave_id.to_bytes(id_size, "little")
        cursor += id_size
    for offset in stored_offsets:
        output[cursor:cursor + offset_size] = offset.to_bytes(
            offset_size, "little"
        )
        cursor += offset_size
    for begin, payload in zip(actual_offsets, payloads):
        output[begin:begin + len(payload)] = payload
    staging = path.with_suffix(path.suffix + ".rebuilding")
    staging.write_bytes(output)
    os.replace(staging, path)


def _recover_hca_base_key(path: Path) -> int | None:
    """Recover the bank's base key while accounting for its embedded subkey."""
    if not CRICODECS.is_file():
        return None
    result = _run(
        [CRICODECS, "--recover-key", "-f", "hca", path, "--json"],
        text=True,
    )
    if result.returncode:
        return None
    try:
        payload = json.loads(result.stdout)
        best = payload["candidates"][0]
        if float(best["score"]) < 0.90:
            return None
        return int(best["key"], 16)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def encode_to_hca(
    wav_path: Path,
    hca_path: Path,
    expected: tuple[int, int, int],
    cipher_type: int = 0,
    keycode: int = 0,
    subkey: int = 0,
    header_version: int = 2,
    allow_time_stretch: bool = False,
    use_cricodecs_hca_encoder: bool = False,
) -> bool:
    """Encode without violating the ACB waveform's original audio contract.

    OmniVoice outputs 24 kHz. P3R's event HCA streams are normally 48 kHz and
    their ACB waveform rows retain that rate. Replacing a slot with a 24 kHz
    HCA may decode in vgmstream but fail or play incorrectly through CRIAtom.
    Probe the original slot, resample a temporary WAV to its exact rate/channel
    contract, encode to a staging HCA, validate it, then replace atomically.
    """
    expected_rate, expected_channels, expected_samples = expected
    converted = hca_path.with_suffix(".contract.wav")
    # VGAudio infers the output codec from the final extension.
    staging = hca_path.with_name(f"{hca_path.stem}.encoding.hca")
    encrypted = hca_path.with_name(f"{hca_path.stem}.encrypted.hca")
    decoded_check = hca_path.with_name(f"{hca_path.stem}.ciphercheck.wav")
    original_check = hca_path.with_name(f"{hca_path.stem}.originalcheck.wav")
    try:
        if _wav_contract(wav_path) == expected:
            # Advanced in-engine QA already delivered a frame-exact cue.
            # Preserve it bit-for-bit; never time-stretch an approved voice.
            shutil.copy2(wav_path, converted)
        else:
            if not allow_time_stretch:
                # Fail closed: the advanced producer must deliver an exact
                # cue. Stretching a whole voice can make it audibly robotic.
                return False
            # Compatibility fallback for legacy production WAVs. Match the
            # cue's original active timing as well as its total sample count.
            if cipher_type:
                original_decode = _run([
                    CRICODECS, hca_path, "-f", "hca",
                    "--key", hex(keycode), "--subkey", hex(subkey),
                    "-o", original_check,
                ])
            else:
                original_decode = _run([
                    VGMSTREAM, "-o", original_check, hca_path,
                ])
            if original_decode.returncode or not original_check.is_file():
                return False
            source_first, source_last, _, source_rate = _wav_active_span(wav_path)
            target_first, target_last, _, _ = _wav_active_span(original_check)
            source_active = max(1, source_last - source_first)
            target_active = max(1, target_last - target_first)
            tempo = (
                (source_active / source_rate)
                / (target_active / expected_rate)
            )
            audio_filter = (
                f"atrim=start_sample={source_first}:end_sample={source_last},"
                "asetpts=PTS-STARTPTS,"
                f"aresample={expected_rate},"
                f"rubberband=tempo={tempo:.9f}:pitch=1.0,"
                f"adelay=delays={target_first}S:all=1,"
                f"apad,atrim=end_sample={expected_samples}"
            )
            conversion = _run([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", wav_path, "-af", audio_filter, "-ar", str(expected_rate),
                "-ac", str(expected_channels), "-c:a", "pcm_s16le", converted,
            ])
            if conversion.returncode or not converted.is_file():
                return False
        if cipher_type:
            if cipher_type != 56 or not keycode or not CRICODECS.is_file():
                return False
            if use_cricodecs_hca_encoder:
                encoded = _run([
                    CRICODECS, "--encode", "-f", "hca", converted,
                    "-o", encrypted,
                    "--header-version", f"{header_version}.00",
                    "--bitrate", "131000",
                    "--cipher-type", str(cipher_type),
                    "--key", hex(keycode), "--subkey", hex(subkey),
                ])
            else:
                # VGAudio's HCA bitstream has already proven compatible with
                # P3R/vgmstream. CriCodecs remains responsible only for the
                # reversible type56 transform; its newer HCA v3 encoder made
                # CRIAtom produce a demonic/garbled first cue in the real game.
                plain_encode = _run([
                    VGAUDIO, "-c", str(converted), "-o", str(staging),
                ])
                if (
                    plain_encode.returncode
                    or not staging.is_file()
                    or _audio_contract(staging) != expected
                ):
                    return False
                encoded = _run([
                    CRICODECS, staging, "--encrypt", "-f", "hca",
                    "--cipher-type", str(cipher_type),
                    "--key", hex(keycode), "--subkey", hex(subkey),
                    "-o", encrypted,
                ])
            if (
                encoded.returncode
                or not encrypted.is_file()
                or _hca_cipher_type(encrypted) != cipher_type
                or _audio_contract(encrypted) != expected
            ):
                return False
            # Round-trip through the same key contract before touching the slot.
            check = _run([
                CRICODECS, encrypted, "-f", "hca",
                "--key", hex(keycode), "--subkey", hex(subkey),
                "-o", decoded_check,
            ])
            if (
                check.returncode
                or not decoded_check.is_file()
                or _audio_contract(decoded_check) != expected
            ):
                return False
            final_hca = encrypted
        else:
            encoded = _run([VGAUDIO, "-c", str(converted), "-o", str(staging)])
            if (
                encoded.returncode
                or not staging.is_file()
                or _audio_contract(staging) != expected
            ):
                return False
            final_hca = staging
        os.replace(final_hca, hca_path)
        return True
    finally:
        converted.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
        encrypted.unlink(missing_ok=True)
        decoded_check.unlink(missing_ok=True)
        original_check.unlink(missing_ok=True)


def rebuild_bank(bank_name: str, pak_entry: dict, replacements: dict[int, Path],
                 work_dir: Path) -> Path | None:
    """replacements: stream_index (1-based, como en el proyecto) -> wav
    aleman generado. Devuelve la ruta del .awb reconstruido, o None si
    algo fallo (nunca se arriesga un .awb a medio escribir).

    pak_entry = {"pak": <ruta al .pak legacy>, "internal": <ruta interna del
    .awb dentro de ese pak>} (mismo formato que awb_index.json /
    awb_index_btlevent.json). OJO: _extract_awb_streams/_extract_btlevent_streams
    de prod_dub.py BORRAN el .awb tras decodificar los streams (solo hacia
    falta para esa extraccion) -- por eso aqui se vuelve a extraer fresco
    con repak en vez de asumir que queda un .awb cacheado en disco."""
    bank_dir = work_dir / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)

    awb_dst = bank_dir / f"{bank_name}.awb"
    extracted_ok = False
    for pak_cand in [pak_entry["pak"]] + [str(PAKS_DIR / f"pakchunk{i}-WindowsNoEditor.pak") for i in range(6)]:
        for path_cand in [pak_entry["internal"], pak_entry["internal"].replace("/en/", "/")]:
            with awb_dst.open("wb") as fh:
                r = subprocess.run([str(REPAK), "-a", AES_KEY, "get", pak_cand, path_cand],
                                    stdout=fh, stderr=subprocess.PIPE)
            if awb_dst.exists() and awb_dst.stat().st_size > 0:
                extracted_ok = True
                break
        if extracted_ok:
            break

    if not extracted_ok:
        print(f"  [{bank_name}] repak no pudo extraer el .awb original, se salta")
        return None

    try:
        awb_contract, awb_ids, original_payloads = _parse_awb(awb_dst)
    except ValueError as exc:
        print(f"  [{bank_name}] no se pudo analizar el AWB: {exc}")
        return None
    dump_dir = bank_dir / bank_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(original_payloads):
        (dump_dir / f"{index:05d}_streaming.hca").write_bytes(payload)

    expected_before = len(list(dump_dir.glob("*_streaming.hca")))
    original_hcas = sorted(dump_dir.glob("*_streaming.hca"))
    encrypted_bank = any(_hca_cipher_type(path) == 56 for path in original_hcas)
    awb_subkey = _awb_subkey(awb_dst)
    hca_keycode = _recover_hca_base_key(awb_dst) if encrypted_bank else 0
    if encrypted_bank and hca_keycode is None:
        print(f"  [{bank_name}] no se pudo recuperar la clave HCA tipo 56; se salta")
        return None
    if encrypted_bank:
        print(
            f"  [{bank_name}] contrato HCA cifrado: type56, "
            f"key={hca_keycode:#x}, subkey={awb_subkey:#x}"
        )
    encoded_streams: list[int] = []
    failed_streams: list[int] = []
    for stream_index, wav_path in replacements.items():
        slot = stream_index - 1  # proyecto 1-indexed, AcbEditor 0-indexed
        hca_name = f"{slot:05d}_streaming.hca"
        hca_path = dump_dir / hca_name
        if not hca_path.exists():
            print(f"    aviso: {bank_name} stream {stream_index} -> {hca_name} no existe, se salta esa linea")
            failed_streams.append(stream_index)
            continue
        expected_contract = _audio_contract(awb_dst, stream_index)
        if expected_contract is None:
            print(f"    aviso: no se pudo leer contrato original de {bank_name} stream {stream_index}")
            failed_streams.append(stream_index)
            continue
        # CRI event banks use one encryption contract for the whole AWB.
        # Never let a stale/mixed dump silently produce a partly plain bank:
        # CRIAtom can mute the entire cue sheet even though vgmstream decodes it.
        cipher_type = 56 if encrypted_bank else 0
        header_version = _hca_header_version(hca_path) or 2
        if not encode_to_hca(
            wav_path,
            hca_path,
            expected_contract,
            cipher_type=cipher_type,
            keycode=hca_keycode or 0,
            subkey=awb_subkey,
            header_version=header_version,
        ):
            print(f"    aviso: fallo codificando {wav_path.name} a HCA, se salta esa linea")
            failed_streams.append(stream_index)
            continue
        encoded_streams.append(stream_index)

    if failed_streams:
        failed = ", ".join(str(index) for index in failed_streams)
        print(f"  [{bank_name}] ALERTA: fallaron streams solicitados ({failed}); "
              "no se reempaqueta un banco parcial")
        return None

    final_cipher_types = {_hca_cipher_type(path) or 0 for path in original_hcas}
    expected_cipher_types = {56} if encrypted_bank else {0}
    if final_cipher_types != expected_cipher_types:
        print(
            f"  [{bank_name}] ALERTA: contrato HCA inconsistente "
            f"({sorted(final_cipher_types)} != {sorted(expected_cipher_types)}); "
            "no se reempaqueta"
        )
        return None

    try:
        rebuilt_payloads = [
            path.read_bytes() for path in sorted(dump_dir.glob("*_streaming.hca"))
        ]
        _write_awb_preserving_contract(
            awb_dst, awb_contract, awb_ids, rebuilt_payloads
        )
    except (OSError, OverflowError, ValueError) as exc:
        print(f"  [{bank_name}] no se pudo reconstruir el AWB: {exc}")
        return None

    # verificacion: mismo stream count que antes de tocar nada
    r = _run([VGMSTREAM, "-m", str(awb_dst)], text=True)
    metadata = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"stream count:\s*(\d+)", metadata)
    got = int(m.group(1)) if m else -1
    if got != expected_before:
        print(f"  [{bank_name}] ALERTA: stream count cambio ({expected_before} -> {got}), "
              f"no se usa este resultado")
        return None
    print(f"  [{bank_name}] OK, {len(encoded_streams)} lineas reemplazadas, "
          f"{got} streams preservados")
    return awb_dst


def group_outputs_by_bank() -> dict[str, dict[int, Path]]:
    """Recorre produccion/*.wav y agrupa por banco de origen, usando la
    MISMA logica de resolve_ref de prod_dub.py para saber de que banco
    salio cada linea (evita reimplementar el despacho por familia)."""
    groups: dict[str, dict[int, Path]] = {}
    recs = pd.load_corpus()
    by_key = {(r["event"], r["stream_index"]): r for r in recs}

    for wav in OUT_DIR.glob("*.wav"):
        m = re.match(r"^(.+)_L(\d+)\.wav$", wav.name)
        if not m:
            continue
        event, stream_index = m.group(1), int(m.group(2))
        rec = by_key.get((event, stream_index))
        if rec is None:
            continue
        entry = pd._load_awb_index().get(event)
        if entry is not None:
            bank_name = f"Voice_Event_{event}"
            groups.setdefault(bank_name, {})[stream_index] = wav
            continue
        entry_b = pd._load_btlevent_index().get(event)
        if entry_b is not None:
            groups.setdefault(event, {})[stream_index] = wav
            continue
        # pool compartido y fallback de audio directo: PENDIENTE -- requiere
        # saber de que .awb especifico salio el clip de referencia usado,
        # no solo el pool_key. Dejar para una iteracion futura del script
        # (ver HANDOFF_AGENTE.md); no se empaqueta esta familia todavia.
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-preparar", action="store_true",
                     help="detenerse tras reconstruir los .awb, sin armar el .pak final")
    args = ap.parse_args()

    if not OUT_DIR.exists() or not any(OUT_DIR.glob("*.wav")):
        print(f"no hay nada en {OUT_DIR} -- corre prod_dub.py --run primero")
        return

    groups = group_outputs_by_bank()
    if not groups:
        print("ninguna linea generada corresponde a un banco narrativa/BtlEvent "
              "reconocido (el pool compartido aun no esta soportado por este script)")
        return

    REBUILD_DIR.mkdir(exist_ok=True)
    print(f"{len(groups)} bancos a reconstruir")
    rebuilt: dict[str, tuple[Path, str]] = {}   # bank_name -> (awb reconstruido, ruta interna original)
    for bank_name, replacements in groups.items():
        entry = pd._load_awb_index().get(bank_name.replace("Voice_Event_", "", 1))
        if entry is None:
            entry = pd._load_btlevent_index().get(bank_name)
        if entry is None:
            continue
        result = rebuild_bank(bank_name, entry, replacements, REBUILD_DIR)
        if result is not None:
            rebuilt[bank_name] = (result, entry["internal"])

    if args.solo_preparar or not rebuilt:
        print(f"\n{len(rebuilt)} bancos reconstruidos en {REBUILD_DIR}")
        return

    print(f"\narmando arbol de carpetas para {len(rebuilt)} bancos...")
    if MOD_TREE_DIR.exists():
        shutil.rmtree(MOD_TREE_DIR)
    for bank_name, (awb_rebuilt, internal_path) in rebuilt.items():
        # internal_path viene como "P3R/Content/.../en/Voice_Event_X.awb" (el
        # mismo formato que devuelve `repak list`, mount point por defecto de
        # repak es "../../../" -- verificado contra `repak -a <key> info
        # <pak original>`: mount point real "../../../P3R/Content/Xrd777/
        # CriData/Stream/" + entrada relativa "en/<archivo>" concatenan al
        # MISMO path final que mount por defecto + esta ruta completa, asi
        # que no hace falta pasar -m explicito.
        dest = MOD_TREE_DIR / internal_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(awb_rebuilt, dest)

        # También duplicar la ruta sin '/en/' para cubrir si el juego está en modo audio Japonés
        if "/en/" in internal_path:
            internal_jp = internal_path.replace("/en/", "/")
            dest_jp = MOD_TREE_DIR / internal_jp
            dest_jp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(awb_rebuilt, dest_jp)

    print(f"empaquetando en {FINAL_PAK.name}...")
    seed = str(int("EA6EE1AD", 16))
    r = subprocess.run([str(REPAK), "pack", "--version", "V11", "-p", seed, str(MOD_TREE_DIR), str(FINAL_PAK)],
                        capture_output=True, text=True)
    if r.returncode != 0 or not FINAL_PAK.exists():
        print(f"ERROR empaquetando: {r.stderr[:500]}")
        return
    print(f"OK -> {FINAL_PAK}")

    # verificacion: el pak resultante debe listar exactamente esas rutas internas
    check = subprocess.run([str(REPAK), "list", str(FINAL_PAK)], capture_output=True, text=True)
    listed = set(l.strip() for l in check.stdout.splitlines() if l.strip())
    expected = {internal for _, internal in rebuilt.values()}
    faltantes = expected - listed
    if faltantes:
        print(f"AVISO: {len(faltantes)} rutas esperadas no aparecen en el pak final: "
              f"{list(faltantes)[:5]}")
    else:
        print(f"verificado: las {len(expected)} rutas esperadas estan en el pak final")


if __name__ == "__main__":
    main()
