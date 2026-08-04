#!/usr/bin/env python3
"""Read-only runtime overlay for the final human/LLM-reviewed line policy.

The reviewed CSV remains the source of truth.  Generators import this module
instead of rewriting the original corpus or physical voice manifests.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "revision_lineas_final_reviewed.csv"
UNIT_ID = re.compile(r"^(?:VU_)?(.+)_L(\d+)$")


@lru_cache(maxsize=1)
def reviewed_rows() -> dict[tuple[str, int], dict[str, str]]:
    with REVIEW.open(encoding="utf-8-sig", newline="") as stream:
        return {
            (row["event"], int(row["stream"])): row
            for row in csv.DictReader(stream)
        }


def review_for(event: str, stream: int) -> dict[str, str] | None:
    return reviewed_rows().get((event, int(stream)))


def review_for_unit_id(unit_id: str) -> dict[str, str] | None:
    match = UNIT_ID.match(unit_id)
    if not match:
        return None
    return review_for(match.group(1), int(match.group(2)))


def is_ready(row: dict[str, str] | None) -> bool:
    return bool(row and row.get("policy_ready") == "1")


def reviewed_action(row: dict[str, str]) -> str:
    return row.get("accion_final") or row.get("accion") or ""


def reviewed_delivery(row: dict[str, str]) -> str:
    return (
        row.get("delivery_text_de_final")
        or row.get("DE")
        or ""
    ).strip()


def physical_asset_override(asset: dict) -> dict | None:
    """Return an override only when the physical delivery is unambiguous.

    Shared bark assets may be associated with many full subtitle pages.  A
    final subtitle rewrite must never replace their short physical delivery.
    The overlay is therefore allowed only when the current physical German
    delivery exactly equals one reviewed member's original German text.
    """

    current = (asset.get("delivery_text_de") or "").strip()
    if not current:
        return None
    matches = []
    for unit_id in asset.get("unit_ids", []):
        row = review_for_unit_id(unit_id)
        if is_ready(row) and (row.get("DE") or "").strip() == current:
            matches.append(row)
    if not matches:
        return None
    signatures = {
        (
            reviewed_action(row),
            reviewed_delivery(row),
            row.get("montage_hint") or "",
            row.get("preserve_original_component") or "",
        )
        for row in matches
    }
    if len(signatures) != 1:
        return None
    action, delivery, montage, preserve = next(iter(signatures))
    return {
        "action": action,
        "delivery_text_de": delivery,
        "montage_hint": montage,
        "preserve_original_component": preserve,
        "reviewed_units": [
            f"{row['event']}_L{int(row['stream']):03d}" for row in matches
        ],
    }
