#!/usr/bin/env python3
"""ASR-audit every deployed exact route against its claimed English text."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "exact_reference_alignment_audit.json"


def words(text: str) -> list[str]:
    value = unicodedata.normalize("NFKD", (text or "").casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", value)


def distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(
                min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b))
            )
        row = nxt
    return row[-1]


def main() -> None:
    master = {
        row["unit_id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "pending_voice_master_manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }
    deploy = json.loads(
        (ROOT / "exact_ryo_deploy_report.json").read_text(encoding="utf-8")
    )
    model = WhisperModel(
        "large-v3-turbo", device="cuda", compute_type="float16"
    )
    transcript_cache = {}
    rows = []
    for index, detail in enumerate(deploy["details"], 1):
        unit = master[detail["unit_id"]]
        reference = unit.get("reference_wav")
        if not reference:
            rows.append(
                {
                    "unit_id": detail["unit_id"],
                    "bank": detail["bank"],
                    "cue": detail["cue"],
                    "status": "NO_REFERENCE",
                }
            )
            continue
        if reference not in transcript_cache:
            segments, _ = model.transcribe(
                reference,
                language="en",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            transcript_cache[reference] = " ".join(
                segment.text.strip() for segment in segments
            ).strip()
        transcript = transcript_cache[reference]
        expected = words(unit["text_en"])
        heard = words(transcript)
        wer = distance(expected, heard) / max(1, len(expected))
        rows.append(
            {
                "unit_id": detail["unit_id"],
                "event": unit["event"],
                "bank": detail["bank"],
                "cue": detail["cue"],
                "reference_wav": reference,
                "expected_en": unit["text_en"],
                "transcript": transcript,
                "wer": wer,
                "status": "PASS" if wer <= 0.45 else "MISMATCH",
            }
        )
        if index % 25 == 0:
            print(f"[{index}/{len(deploy['details'])}]")
    payload = {
        "routes": len(rows),
        "pass": sum(row["status"] == "PASS" for row in rows),
        "mismatch": sum(row["status"] == "MISMATCH" for row in rows),
        "no_reference": sum(row["status"] == "NO_REFERENCE" for row in rows),
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2)
    )


if __name__ == "__main__":
    main()
