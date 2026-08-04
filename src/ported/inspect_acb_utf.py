"""Dump ACB @UTF schemas and compact row samples as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cri_utf import UtfTable, is_utf_blob


def compact(value: Any) -> Any:
    if isinstance(value, bytes):
        if is_utf_blob(value):
            nested = UtfTable(value)
            return {
                "nested_utf": nested.table_name,
                "rows": nested.row_count,
                "columns": [column.name for column in nested.columns],
            }
        return {"bytes": len(value), "hex": value[:32].hex()}
    return value


def describe(table: UtfTable, sample_rows: int) -> dict[str, Any]:
    return {
        "name": table.table_name,
        "row_count": table.row_count,
        "row_size": table.row_size,
        "columns": [
            {
                "name": column.name,
                "storage": hex(column.storage),
                "type": hex(column.value_type),
            }
            for column in table.columns
        ],
        "sample": [
            {key: compact(value) for key, value in row.items()}
            for row in table.rows[:sample_rows]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("acb", type=Path)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = UtfTable.from_file(args.acb)
    report = {"root": describe(root, args.rows), "children": {}}
    for name, value in root.rows[0].items():
        if is_utf_blob(value):
            report["children"][name] = describe(UtfTable(value), args.rows)

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
