"""Materialize every cue in an exact ACB as an analysis-only mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from acb_cue_resolver import AcbCueResolver
from extract_singleword_references import atomic_jsonl


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("prefix")
    args = parser.parse_args()
    acbs = list((ROOT / "shared_exact_banks" / args.variant).glob("*.acb"))
    if len(acbs) != 1:
        raise RuntimeError(f"{args.variant}: expected one ACB, got {acbs}")
    resolver = AcbCueResolver(acbs[0])
    rows = []
    status = Counter()
    physical = set()
    for cue_id in sorted(resolver.cue_ids):
        mappings = resolver.resolve(cue_id)
        if not mappings:
            verdict = "ACB_GRAPH_UNRESOLVED"
        elif len(mappings) == 1:
            verdict = "ACB_MAPPED"
        else:
            verdict = "ACB_MULTI_WAVEFORM"
        status[verdict] += 1
        for mapping in mappings:
            physical.add(
                (
                    args.variant,
                    mapping.stream_awb_port,
                    mapping.stream_awb_id,
                )
            )
        rows.append(
            {
                "unit_id": f"ACB_{args.variant}_Cue{cue_id}",
                "event": args.variant,
                "text_en": "",
                "shared_bank_variant": args.variant,
                "resolved_cue_id": cue_id,
                "acb_status": verdict,
                "physical_waveforms": [
                    mapping.to_dict() for mapping in mappings
                ],
                "promoted": False,
            }
        )
    output = ROOT / f"shared_{args.prefix}_all_acb_cues.jsonl"
    atomic_jsonl(output, rows)
    summary = {
        "variant": args.variant,
        "cues": len(rows),
        "status": dict(sorted(status.items())),
        "unique_physical_waveforms": len(physical),
        "analysis_only": True,
        "output": str(output),
    }
    (ROOT / f"shared_{args.prefix}_all_acb_cues_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
