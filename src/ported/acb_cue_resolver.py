"""Resolve CRI ACB cue IDs to their physical AWB waveform IDs.

Read-only.  This follows the actual ACB graph:
Cue -> Sequence -> Track -> TrackEvent command -> Synth -> Waveform.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from cri_utf import UtfTable, is_utf_blob


@dataclass(frozen=True)
class ResolvedWaveform:
    cue_id: int
    cue_name: str
    cue_index: int
    sequence_index: int
    track_index: int
    synth_index: int
    waveform_index: int
    streaming: int
    stream_awb_port: int
    stream_awb_id: int
    memory_awb_id: int
    encode_type: int
    channels: int
    sample_rate: int
    num_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def command_records(payload: bytes) -> Iterator[tuple[int, bytes]]:
    cursor = 0
    while cursor + 3 <= len(payload):
        command, size = struct.unpack_from(">HB", payload, cursor)
        cursor += 3
        if cursor + size > len(payload):
            raise ValueError(
                f"Truncated ACB command 0x{command:04x}: "
                f"needs {size}, has {len(payload) - cursor}"
            )
        parameters = payload[cursor : cursor + size]
        cursor += size
        yield command, parameters
        if command == 0:
            break


def u16_array(payload: bytes, limit: int | None = None) -> list[int]:
    if len(payload) % 2:
        raise ValueError(f"Odd-sized uint16 array ({len(payload)} bytes)")
    values = [value[0] for value in struct.iter_unpack(">H", payload)]
    return values if limit is None else values[:limit]


class AcbCueResolver:
    def __init__(self, acb: str | Path):
        self.path = Path(acb)
        self.root = UtfTable.from_file(self.path)
        root = self.root.rows[0]
        self.tables: dict[str, UtfTable] = {
            name: UtfTable(value)
            for name, value in root.items()
            if is_utf_blob(value)
        }
        required = {
            "CueTable",
            "CueNameTable",
            "WaveformTable",
            "SynthTable",
            "TrackTable",
            "SequenceTable",
            "TrackEventTable",
        }
        missing = required - self.tables.keys()
        if missing:
            raise ValueError(f"{self.path}: missing ACB tables {sorted(missing)}")
        self.cue_names = {
            row["CueIndex"]: row["CueName"]
            for row in self.tables["CueNameTable"].rows
        }
        self._by_id: dict[int, list[int]] = {}
        for index, cue in enumerate(self.tables["CueTable"].rows):
            self._by_id.setdefault(cue["CueId"], []).append(index)

    @property
    def cue_ids(self) -> set[int]:
        return set(self._by_id)

    def _synth_waveforms(self, synth_index: int) -> list[int]:
        synths = self.tables["SynthTable"].rows
        if not 0 <= synth_index < len(synths):
            raise ValueError(f"Synth index outside table: {synth_index}")
        references = synths[synth_index]["ReferenceItems"]
        result: list[int] = []
        for reference_type, reference_index in struct.iter_unpack(">HH", references):
            # Reference type 1 is a waveform.  Type 2 points to another synth
            # in some CRI banks, so recurse while guarding the trivial cycle.
            if reference_type == 1:
                result.append(reference_index)
            elif reference_type == 2 and reference_index != synth_index:
                result.extend(self._synth_waveforms(reference_index))
        return result

    def _sequence_paths(
        self, sequence_index: int
    ) -> list[tuple[int, int, int]]:
        sequences = self.tables["SequenceTable"].rows
        tracks = self.tables["TrackTable"].rows
        events = self.tables["TrackEventTable"].rows
        if not 0 <= sequence_index < len(sequences):
            raise ValueError(f"Sequence index outside table: {sequence_index}")
        sequence = sequences[sequence_index]
        track_indices = u16_array(
            sequence["TrackIndex"], int(sequence.get("NumTracks", 0))
        )
        paths: list[tuple[int, int, int]] = []
        for track_index in track_indices:
            track = tracks[track_index]
            event_index = track["EventIndex"]
            if event_index == 0xFFFF:
                continue
            command_blob = events[event_index]["Command"]
            for command, parameters in command_records(command_blob):
                if command != 0x07D0 or len(parameters) < 4:
                    continue
                reference_type, reference_index = struct.unpack_from(
                    ">HH", parameters
                )
                if reference_type == 2:
                    for waveform_index in self._synth_waveforms(reference_index):
                        paths.append(
                            (track_index, reference_index, waveform_index)
                        )
        return paths

    def resolve(self, cue_id: int) -> list[ResolvedWaveform]:
        waveforms = self.tables["WaveformTable"].rows
        result: list[ResolvedWaveform] = []
        for cue_index in self._by_id.get(cue_id, []):
            cue = self.tables["CueTable"].rows[cue_index]
            cue_name = self.cue_names.get(cue_index, f"Cue{cue_id}")
            reference_type = cue["ReferenceType"]
            reference_index = cue["ReferenceIndex"]
            if reference_type == 3:
                paths = self._sequence_paths(reference_index)
                sequence_index = reference_index
            elif reference_type == 8:
                paths = [
                    (-1, reference_index, waveform_index)
                    for waveform_index in self._synth_waveforms(reference_index)
                ]
                sequence_index = -1
            else:
                raise ValueError(
                    f"Cue {cue_id} uses unsupported ReferenceType "
                    f"{reference_type}"
                )
            for track_index, synth_index, waveform_index in paths:
                waveform = waveforms[waveform_index]
                result.append(
                    ResolvedWaveform(
                        cue_id=cue_id,
                        cue_name=cue_name,
                        cue_index=cue_index,
                        sequence_index=sequence_index,
                        track_index=track_index,
                        synth_index=synth_index,
                        waveform_index=waveform_index,
                        streaming=int(waveform["Streaming"]),
                        stream_awb_port=int(waveform["StreamAwbPortNo"]),
                        stream_awb_id=int(waveform["StreamAwbId"]),
                        memory_awb_id=int(waveform["MemoryAwbId"]),
                        encode_type=int(waveform["EncodeType"]),
                        channels=int(waveform["NumChannels"]),
                        sample_rate=int(waveform["SamplingRate"]),
                        num_samples=int(waveform["NumSamples"]),
                    )
                )
        # Stable de-duplication: multiple sequence paths may intentionally
        # reference the same waveform.
        unique = {}
        for item in result:
            key = (
                item.cue_index,
                item.track_index,
                item.synth_index,
                item.waveform_index,
            )
            unique[key] = item
        return list(unique.values())
