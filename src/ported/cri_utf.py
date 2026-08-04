"""Minimal read-only parser for CRI @UTF tables used inside ACB files.

This module intentionally does not write or rebuild ACB/AWB data.  It exists so
the dubbing inventory can follow a cue through the ACB tables instead of
guessing that an AWB ordinal is the cue number.

The layout follows the public MIT-licensed `acb.py` parser by dnaroma / The Holy
Constituency of the Summer Triangle:
https://gist.github.com/dnaroma/fa17c383696c59f4a77d40ddfd573779
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


STORAGE_MASK = 0xF0
STORAGE_ZERO = 0x10
STORAGE_CONSTANT = 0x30
STORAGE_PER_ROW = 0x50
STORAGE_CONSTANT_ALT = 0x70

TYPE_MASK = 0x0F
TYPE_U8 = 0x00
TYPE_S8 = 0x01
TYPE_U16 = 0x02
TYPE_S16 = 0x03
TYPE_U32 = 0x04
TYPE_S32 = 0x05
TYPE_U64 = 0x06
TYPE_FLOAT = 0x08
TYPE_STRING = 0x0A
TYPE_DATA = 0x0B

TYPE_FORMAT = {
    TYPE_U8: "B",
    TYPE_S8: "b",
    TYPE_U16: "H",
    TYPE_S16: "h",
    TYPE_U32: "I",
    TYPE_S32: "i",
    TYPE_U64: "Q",
    TYPE_FLOAT: "f",
    TYPE_STRING: "I",
    TYPE_DATA: "II",
}


@dataclass(frozen=True)
class UtfColumn:
    name: str
    storage: int
    value_type: int
    constant: Any = None


class CriUtfError(ValueError):
    pass


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise CriUtfError(f"Unexpected EOF: wanted {count}, got {len(data)}")
    return data


def _read_c_string(data: bytes, offset: int) -> str:
    if not 0 <= offset < len(data):
        raise CriUtfError(f"String offset outside table: {offset}")
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def _zero_for_type(value_type: int) -> Any:
    if value_type == TYPE_STRING:
        return ""
    if value_type == TYPE_DATA:
        return b""
    if value_type == TYPE_FLOAT:
        return 0.0
    return 0


class UtfTable:
    def __init__(self, source: bytes | bytearray | memoryview | BinaryIO):
        if hasattr(source, "read"):
            raw = source.read()
        else:
            raw = bytes(source)
        self.raw = raw
        if len(raw) < 0x20 or raw[:4] != b"@UTF":
            raise CriUtfError("Not a CRI @UTF table")

        (
            self.table_size,
            self.unknown,
            row_offset,
            string_offset,
            data_offset,
            table_name_offset,
            self.column_count,
            self.row_size,
            self.row_count,
        ) = struct.unpack_from(">IHHIIIHHI", raw, 4)

        # CRI offsets are relative to byte 8 of the @UTF object.
        self.row_offset = row_offset + 8
        self.string_offset = string_offset + 8
        self.data_offset = data_offset + 8
        self.table_name = _read_c_string(raw, self.string_offset + table_name_offset)
        self.columns = self._parse_columns()
        self.rows = self._parse_rows()

    def _read_encoded_value(self, cursor: int, value_type: int) -> tuple[Any, int]:
        try:
            fmt = TYPE_FORMAT[value_type]
        except KeyError as exc:
            raise CriUtfError(f"Unsupported @UTF value type 0x{value_type:02x}") from exc
        size = struct.calcsize(">" + fmt)
        values = struct.unpack_from(">" + fmt, self.raw, cursor)
        cursor += size
        if value_type == TYPE_STRING:
            return _read_c_string(self.raw, self.string_offset + values[0]), cursor
        if value_type == TYPE_DATA:
            offset, count = values
            start = self.data_offset + offset
            end = start + count
            if start < 0 or end > len(self.raw):
                raise CriUtfError(
                    f"Data reference outside table: offset={offset}, size={count}"
                )
            return self.raw[start:end], cursor
        return values[0], cursor

    def _parse_columns(self) -> list[UtfColumn]:
        columns: list[UtfColumn] = []
        cursor = 0x20
        for _ in range(self.column_count):
            flags = self.raw[cursor]
            cursor += 1
            (name_offset,) = struct.unpack_from(">I", self.raw, cursor)
            cursor += 4
            storage = flags & STORAGE_MASK
            value_type = flags & TYPE_MASK
            name = _read_c_string(self.raw, self.string_offset + name_offset)
            if storage in (STORAGE_CONSTANT, STORAGE_CONSTANT_ALT):
                constant, cursor = self._read_encoded_value(cursor, value_type)
            elif storage == STORAGE_ZERO:
                constant = _zero_for_type(value_type)
            elif storage == STORAGE_PER_ROW:
                constant = None
            else:
                raise CriUtfError(
                    f"Unsupported storage 0x{storage:02x} for column {name!r}"
                )
            columns.append(UtfColumn(name, storage, value_type, constant))
        return columns

    def _parse_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_index in range(self.row_count):
            cursor = self.row_offset + row_index * self.row_size
            row: dict[str, Any] = {}
            for column in self.columns:
                if column.storage == STORAGE_PER_ROW:
                    value, cursor = self._read_encoded_value(cursor, column.value_type)
                else:
                    value = column.constant
                row[column.name] = value
            expected_end = self.row_offset + (row_index + 1) * self.row_size
            # Recent ACB root tables may declare a row size that excludes a
            # fixed extension area even though its columns remain tagged as
            # per-row.  There is only one root row in those files and the
            # actual row safely ends at the string table.  Nested multi-row
            # tables still require the declared stride to be exact.
            root_extension_case = (
                self.row_count == 1 and cursor <= self.string_offset
            )
            if cursor > expected_end and not root_extension_case:
                raise CriUtfError(
                    f"Decoded row {row_index} exceeds declared row size "
                    f"({cursor - expected_end:+d} bytes)"
                )
            rows.append(row)
        return rows

    @classmethod
    def from_file(cls, path: str | Path) -> "UtfTable":
        return cls(Path(path).read_bytes())

    def child(self, row: int, column: str) -> "UtfTable":
        value = self.rows[row][column]
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise CriUtfError(f"{column!r} is not a nested data table")
        return UtfTable(value)


def is_utf_blob(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview)) and bytes(value).startswith(
        b"@UTF"
    )
