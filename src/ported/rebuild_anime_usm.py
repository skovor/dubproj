#!/usr/bin/env python3
"""Replace one ADX audio track in a CRI USM without remuxing its container.

P3R anime movies contain two 5.1 ADX tracks: Japanese (channel 0) and English
(channel 1). The German dub replaces channel 1. The replacement WAV must have
the exact sample rate, channel count and frame count declared by that track.

Run this script with the dedicated ``tools_cricodecs_env`` Python. WannaCRI is
used only to read contracts. Its serializer is deliberately not used: it
rewrites every USM chunk and CRI Mana may reject the result even when vgmstream
accepts it. Since the replacement ADX is byte-exact in length, only the
original @SFA STREAM payload regions are patched in place.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from wannacri.usm import Usm


def element(page, name: str) -> int:
    return int(page._dict[name].val)


def read_stream(stream) -> tuple[bytes, list[int]]:
    chunks: list[bytes] = []
    sizes: list[int] = []
    for item in stream:
        data = item[0] if isinstance(item, tuple) else item
        data = bytes(data)
        chunks.append(data)
        sizes.append(len(data))
    return b"".join(chunks), sizes


def adx_header_size(data: bytes) -> int:
    if len(data) < 4 or data[:2] != b"\x80\x00":
        raise ValueError("VGAudio output is not a CRI ADX stream")
    return int.from_bytes(data[2:4], "big") + 4


def stream_payload_regions(
    raw: bytes, signature: bytes, channel_number: int,
) -> list[tuple[int, int]]:
    """Return exact payload regions for one channel without repacking chunks."""
    regions: list[tuple[int, int]] = []
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 0x20:
            raise ValueError(f"truncated USM chunk header at 0x{offset:x}")
        chunk_size = int.from_bytes(raw[offset + 4:offset + 8], "big")
        total_size = 8 + chunk_size
        if chunk_size < 0x18 or offset + total_size > len(raw):
            raise ValueError(
                f"invalid USM chunk size {chunk_size} at 0x{offset:x}"
            )
        payload_offset = raw[offset + 9]
        padding_size = int.from_bytes(raw[offset + 10:offset + 12], "big")
        payload_size = chunk_size - payload_offset - padding_size
        if payload_size < 0:
            raise ValueError(f"negative USM payload size at 0x{offset:x}")
        payload_type = raw[offset + 15] & 0x03
        channel = raw[offset + 12]
        if (
            raw[offset:offset + 4] == signature
            and channel == channel_number
            and payload_type == 0
        ):
            begin = offset + 8 + payload_offset
            regions.append((begin, payload_size))
        offset += total_size
    if offset != len(raw):
        raise ValueError(f"USM chunk walk ended at {offset}, file has {len(raw)} bytes")
    return regions


def wav_contract(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getnchannels(), wav.getnframes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-usm", required=True, type=Path)
    parser.add_argument("--german-wav", required=True, type=Path)
    parser.add_argument("--output-usm", required=True, type=Path)
    parser.add_argument("--vgaudio", required=True, type=Path)
    parser.add_argument("--track", type=int, default=1, help="0-based USM audio track")
    args = parser.parse_args()

    for path in (args.original_usm, args.german_wav, args.vgaudio):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.original_usm.resolve() == args.output_usm.resolve():
        raise ValueError("refusing to overwrite the original USM")

    movie = Usm.open(str(args.original_usm), encoding="shift-jis")
    if not 0 <= args.track < len(movie.audios):
        raise ValueError(f"track {args.track} does not exist; movie has {len(movie.audios)}")

    original_audio = movie.audios[args.track]
    header = original_audio.header_page
    expected = (
        element(header, "sampling_rate"),
        element(header, "num_channels"),
        element(header, "total_samples"),
    )
    actual = wav_contract(args.german_wav)
    if actual != expected:
        raise ValueError(f"German WAV contract mismatch: expected {expected}, got {actual}")
    if element(header, "audio_codec") != 2:
        raise ValueError("target track is not CRI ADX (audio_codec != 2)")

    original_bytes, packet_sizes = read_stream(original_audio._stream)
    original_header_size = adx_header_size(original_bytes)
    declared_size = element(original_audio.crid_page, "filesize")
    if len(original_bytes) != declared_size:
        raise ValueError(
            f"original ADX size mismatch: stream={len(original_bytes)}, table={declared_size}"
        )

    args.output_usm.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p3r_adx_") as temp:
        encoded = Path(temp) / "german.adx"
        result = subprocess.run(
            [
                str(args.vgaudio), "-c", str(args.german_wav), "-o", str(encoded),
                "--version", "4", "--framesize", "18",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode or not encoded.is_file():
            raise RuntimeError(
                f"VGAudio ADX encode failed ({result.returncode}): "
                f"{result.stderr[-1000:]}"
            )
        encoded_bytes = encoded.read_bytes()

    encoded_header_size = adx_header_size(encoded_bytes)
    # P3R's SFA carries a 304-byte ADX header while VGAudio emits the compact
    # 40-byte equivalent. Preserve the game's original header (its declared
    # sample count already equals the exact-frame German WAV) and replace only
    # the ADPCM frames.
    replacement_bytes = (
        original_bytes[:original_header_size] + encoded_bytes[encoded_header_size:]
    )
    if len(replacement_bytes) != len(original_bytes):
        raise ValueError(
            "encoded ADX payload does not fit the original stream exactly: "
            f"replacement={len(replacement_bytes)}, original={len(original_bytes)}"
        )

    original_raw = args.original_usm.read_bytes()
    regions = stream_payload_regions(
        original_raw, b"@SFA", original_audio._channel_number
    )
    region_sizes = [size for _, size in regions]
    if region_sizes != packet_sizes:
        raise ValueError(
            "raw @SFA packet layout differs from parsed audio stream: "
            f"raw={region_sizes[:8]}... parsed={packet_sizes[:8]}..."
        )
    if sum(region_sizes) != len(replacement_bytes):
        raise ValueError(
            f"replacement has {len(replacement_bytes)} bytes but raw regions "
            f"hold {sum(region_sizes)}"
        )
    patched = bytearray(original_raw)
    source_offset = 0
    for begin, size in regions:
        patched[begin:begin + size] = replacement_bytes[source_offset:source_offset + size]
        source_offset += size

    partial = args.output_usm.with_suffix(args.output_usm.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.write(patched)
    if partial.stat().st_size != args.original_usm.stat().st_size:
        raise ValueError(
            "in-place patch changed container size: "
            f"{args.original_usm.stat().st_size} -> {partial.stat().st_size}"
        )

    check = Usm.open(str(partial), encoding="shift-jis")
    if len(check.videos) != len(movie.videos) or len(check.audios) != len(movie.audios):
        raise ValueError("rebuilt USM stream count changed")
    check_header = check.audios[args.track].header_page
    rebuilt_contract = (
        element(check_header, "sampling_rate"),
        element(check_header, "num_channels"),
        element(check_header, "total_samples"),
    )
    if rebuilt_contract != expected:
        raise ValueError(
            f"rebuilt track contract mismatch: expected {expected}, got {rebuilt_contract}"
        )
    checked_video_count = len(check.videos)
    checked_audio_count = len(check.audios)

    # Exact-container invariant: no byte outside the selected @SFA STREAM
    # regions may change.
    allowed = bytearray(len(original_raw))
    for begin, size in regions:
        allowed[begin:begin + size] = b"\x01" * size
    for index, (before, after) in enumerate(zip(original_raw, patched)):
        if before != after and not allowed[index]:
            raise ValueError(
                f"byte outside selected @SFA payload changed at 0x{index:x}"
            )

    # Close WannaCRI's lazy file handles before Windows performs the rename.
    for media in [*check.audios, *check.videos]:
        stream = getattr(media, "_stream", None)
        frame = getattr(stream, "gi_frame", None)
        owner = frame.f_locals.get("usmfile") if frame is not None else None
        if owner is not None:
            owner.close()
    os.replace(partial, args.output_usm)
    print(
        f"PASS: {args.output_usm} | video={checked_video_count} | "
        f"audio_tracks={checked_audio_count} | German track={args.track} {expected}"
    )


if __name__ == "__main__":
    main()
