#!/usr/bin/env python3
"""QA-gated production for a P3R in-engine voice bank."""
from __future__ import annotations

import argparse
import gc
import html
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from clean_runtime import prepare

prepare()

from audio_contracts import active_span, resample_exact, spec, write_exact
from line_policy import KEEP_ORIGINAL, classify_line
from produce_anime_scene import (
    CONFIG_PATH, PROJECT, GenerationRuntime, OmniVoice, evaluate, generate_round,
    read, write_state,
)


def load_generation_runtime(profile: dict) -> GenerationRuntime:
    """Keep one OmniVoice model/prompt cache for all rounds of one VN bank."""
    model = OmniVoice.from_pretrained(
        profile.get("model", "k2-fsa/OmniVoice"),
        device_map="cuda", dtype=torch.float16,
    )
    return GenerationRuntime(model)


def retryable_ids(focused: dict[str, list[dict]]) -> set[str]:
    """Do not spend retries on deterministic mapping/window failures."""
    return {
        line_id for line_id, rows in focused.items()
        if rows and not any(row.get("pass") for row in rows)
        and any(row.get("failure_class") == "RANDOM_TTS" for row in rows)
    }


def build_scene(map_path: Path, refs: Path, out: Path, bank: str, prefix: str) -> tuple[dict, np.ndarray, int, dict[str, Path]]:
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    chunks, lines, cue_paths = [], [], {}
    sr = None
    cursor = 0
    for item in mapping["voiced_lines"]:
        stream = int(item["stream_index"])
        line_id = f"{prefix}_L{stream:03d}"
        cue = refs / f"{line_id}_EN.wav"
        audio, cue_sr = read(cue)
        if audio.ndim != 1:
            raise ValueError(f"cue must be mono: {cue}")
        if sr is None:
            sr = cue_sr
        if cue_sr != sr:
            raise ValueError("mixed sample rates in bank")
        start, end = cursor, cursor + len(audio)
        line = {
            "id": line_id, "speaker": item["speaker_en"],
            "start": start / sr, "end": end / sr,
            "source_text": item["text_en"], "target_text": item["text_de"],
            "source_frames": len(audio), "stream_index": stream,
        }
        for optional in (
            "reference_segments",
            "rejected_candidates",
            "preferred_candidate",
        ):
            if optional in item:
                line[optional] = item[optional]
        stage_free = re.sub(r"\*[^*]+\*", "", line["target_text"]).strip()
        if stage_free != line["target_text"] and stage_free:
            line["delivery_text"] = stage_free
            line["delivery_note"] = "Stage direction removed from speech; subtitle remains unchanged."
        if bank == "100_040_B" and stream in (9, 12):
            line["delivery_text"] = "Unterschreib!"
            line["delivery_note"] = "Timing adaptation; subtitle remains 'Unterschreib das.'"
        if bank == "100_070_C" and stream == 1:
            line.update({
                "delivery_text": "Dein Zimmer.",
                "delivery_note": "Exact-window adaptation; subtitle remains 'Hier, dein Zimmer.'",
                "synthesis_text_override": "Hier, dein Zimmer ... Ich wollte dir noch etwas sagen.",
                "delivery_word_start": 1,
                "delivery_word_count": 2,
            })
        if bank == "100_070_C" and stream == 18:
            line.update({
                "delivery_text": "Nacht.",
                "delivery_note": "Exact-window adaptation; subtitle remains 'Gute Nacht.'",
                "synthesis_text_override": "Gute Nacht ... Ich wollte dir noch etwas sagen.",
                "delivery_word_start": 1,
                "delivery_word_count": 1,
            })
        if bank == "100_130_C" and stream == 19:
            line.update({
                "delivery_text": "Ja, klar.",
                "delivery_note": "Exact-window adaptation; subtitle remains 'Schicksal? Ja, klar.'",
                "synthesis_text_override": "Schicksal? Ja, klar ... Ich wollte dir noch etwas sagen.",
                "delivery_word_start": 1,
                "delivery_word_count": 2,
            })
        if bank == "100_130_C" and stream == 17:
            line.update({
                "delivery_text": "Witzig?",
                "delivery_note": "Exact-window adaptation; subtitle remains 'Witzig, oder?'",
                "synthesis_text_override": "Witzig, oder ... Sag mir bitte, was du meinst.",
                "delivery_word_start": 0,
                "delivery_word_count": 1,
            })
        lines.append(line)
        cue_paths[line_id] = cue
        chunks.append(audio)
        cursor = end
    stem = np.concatenate(chunks).astype(np.float32)
    input_dir = out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    composite = input_dir / f"{bank}_EN_cues_concatenated.wav"
    write_exact(composite, stem, int(sr), len(stem))
    scene = {
        "scene": bank, "kind": "in_engine", "source_stem": str(composite),
        "container_frames": len(stem), "sample_rate": int(sr), "lines": lines,
        "notes": "Composite exists only for shared generation/QA; delivery is one exact WAV per cue.",
    }
    (out / f"{bank}_contract_map.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    return scene, stem, int(sr), cue_paths


def select_cues(
    scene: dict,
    cue_paths: dict[str, Path],
    out: Path,
    rankings: dict,
    strict_ids: set[str] | None = None,
) -> dict:
    selected_dir = out / "selected_exact"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in scene["lines"]:
        decision = classify_line(line["source_text"], line.get("delivery_text", line["target_text"]))
        source = cue_paths[line["id"]]
        expected = spec(source)
        target = selected_dir / f"{line['id']}_DE.wav"
        # A focused repair must not remount or rejudge unrelated historical
        # cues. Preserve their current selected waveform exactly and enforce
        # fail-closed only on the IDs explicitly requested.
        if strict_ids is not None and line["id"] not in strict_ids:
            if not target.exists():
                raise RuntimeError(
                    f"focused repair cannot preserve missing cue: {line['id']}"
                )
            current = spec(target)
            rows.append({
                "id": line["id"], "action": decision.action,
                "source": str(source), "output": str(target),
                "preserved_unrelated": True,
                "contract": expected.__dict__,
                "contract_pass": current == expected,
                "peak": 0.0, "clipped_samples": 0,
            })
            continue
        if decision.action == KEEP_ORIGINAL:
            shutil.copy2(source, target)
            original_audio, _ = read(target)
            rows.append({
                "id": line["id"], "action": KEEP_ORIGINAL, "source": str(source),
                "output": str(target), "contract": expected.__dict__,
                "contract_pass": spec(target) == expected,
                "peak": float(np.max(np.abs(original_audio), initial=0.0)),
                "clipped_samples": int(np.sum(np.abs(original_audio) >= 0.999)),
            })
            continue
        rejected = set(line.get("rejected_candidates", []))
        candidates = sorted(
            (
                row for row in rankings[line["id"]]
                if row.get("file") not in rejected
            ),
            key=lambda row: (not row["pass"], row["score"]),
        )
        # Never turn a QA failure into a mountable delivery merely because it
        # is the least-bad take.  In particular, an exact cue whose active
        # voice reaches the right edge used to be faded/cropped below and
        # could sound as if the final syllable had been choked off.  Such a
        # take must be regenerated; packaging is not an audio repair stage.
        winner = next(
            (row for row in candidates if row.get("processed") and row.get("pass")),
            None,
        )
        if winner is None:
            reasons = sorted({
                key
                for row in candidates
                for key, passed in (row.get("hard_gates") or {}).items()
                if not passed
            })
            raise RuntimeError(
                f"no QA-passing mountable candidate: {line['id']} "
                f"(failed gates: {', '.join(reasons) or 'processing'})"
            )
        audio, sr = read(winner["processed"])
        audio = resample_exact(audio, sr, expected.sample_rate) if sr != expected.sample_rate else audio
        resample_gain_db = 0.0
        dc_offset_removed = 0.0
        resampled_peak = float(np.max(np.abs(audio), initial=0.0))
        if resampled_peak > 0.98:
            factor = 0.98 / resampled_peak
            audio *= factor
            resample_gain_db = float(20.0 * np.log10(factor))
        # Remove only material DC drift, and only inside the detected active
        # span.  A short ramp keeps the original zero/silent cue boundaries
        # untouched, unlike subtracting one constant from the whole file.
        dc = float(np.mean(audio))
        if abs(dc) >= 0.0008:
            active_start, active_end = active_span(audio, expected.sample_rate)
            if active_end > active_start:
                correction = np.zeros(len(audio), dtype=np.float32)
                correction[active_start:active_end] = 1.0
                ramp = min(round(0.010 * expected.sample_rate), (active_end - active_start) // 2)
                if ramp > 0:
                    correction[active_start:active_start + ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                    correction[active_end - ramp:active_end] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
                weight = float(np.sum(correction))
                if weight > 0.0:
                    dc_offset_removed = float(np.sum(audio) / weight)
                    audio -= dc_offset_removed * correction
        adjustment = len(audio) - expected.frames
        active_end = active_span(audio, expected.sample_rate)[1]
        tail_was_active = bool(adjustment > 0 and active_end > expected.frames)
        if len(audio) < expected.frames:
            audio = np.pad(audio, (0, expected.frames - len(audio)))
        elif len(audio) > expected.frames:
            discarded = audio[expected.frames:]
            peak = float(np.max(np.abs(audio), initial=0.0))
            discarded_peak = float(np.max(np.abs(discarded), initial=0.0))
            if peak > 0.0 and discarded_peak >= peak * 0.01:
                raise RuntimeError(
                    f"refusing active-tail crop for {line['id']}: "
                    f"{len(audio) - expected.frames} frames exceed cue"
                )
            fade = min(round(0.010 * expected.sample_rate), expected.frames)
            audio[expected.frames - fade:expected.frames] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            audio = audio[:expected.frames]
        write_exact(target, audio, expected.sample_rate, expected.frames)
        actual = spec(target)
        rows.append({
            "id": line["id"], "action": decision.action, "source": str(source),
            "output": str(target), "winner": winner["file"], "round": winner["round"],
            "pass": winner["pass"], "score": winner["score"], "wer": winner.get("wer"),
            "transcript": winner.get("transcript"), "hard_gates": winner.get("hard_gates"),
            "alignment_fallback": winner.get("alignment_fallback"),
            "resample_frame_adjustment": adjustment, "tail_adjustment_touched_active": tail_was_active,
            "post_resample_gain_db": resample_gain_db,
            "post_resample_dc_offset_removed": dc_offset_removed,
            "contract": expected.__dict__, "contract_pass": actual == expected,
            "peak": float(np.max(np.abs(audio), initial=0.0)),
            "clipped_samples": int(np.sum(np.abs(audio) >= 0.999)),
        })
    report = {
        "bank": scene["scene"], "lines": rows,
        "all_contracts_pass": all(row["contract_pass"] for row in rows),
        "clipped_samples": sum(row.get("clipped_samples", 0) for row in rows if row["action"] != KEEP_ORIGINAL),
        "inherited_original_clipped_samples": sum(row.get("clipped_samples", 0) for row in rows if row["action"] == KEEP_ORIGINAL),
        "tail_adjustments_touching_active": [row["id"] for row in rows if row.get("tail_adjustment_touched_active")],
    }
    (out / "FINAL_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def make_html(scene: dict, report: dict, out: Path) -> None:
    by_id = {row["id"]: row for row in report["lines"]}
    cards = []
    for line in scene["lines"]:
        row = by_id[line["id"]]
        source = Path(os.path.relpath(row.get("source", row["output"]), out)).as_posix()
        target = Path(os.path.relpath(row["output"], out)).as_posix()
        cards.append(
            f"<section><h3>{html.escape(line['id'])}</h3><p>EN: {html.escape(line['source_text'])}<br>"
            f"DE subtitle: {html.escape(line['target_text'])}<br>DE spoken: {html.escape(line.get('delivery_text', line['target_text']))}</p><audio controls src='{html.escape(source)}'></audio> "
            f"<audio controls src='{html.escape(target)}'></audio><p>PASS={row.get('pass', True)} | "
            f"WER={row.get('wer', 0):.3f} | ASR={html.escape(str(row.get('transcript', 'original')))}</p></section>"
        )
    page = f"<!doctype html><meta charset='utf-8'><title>{html.escape(scene['scene'])} QA</title><style>body{{background:#17191d;color:#eee;font:16px system-ui;margin:24px}}section{{border:1px solid #667;padding:12px;margin:10px}}audio{{width:46%}}</style><h1>{html.escape(scene['scene'])} exact cues: EN / DE</h1>" + "".join(cards)
    (out / "QA_LISTEN.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=PROJECT / "workspace" / "evento_100_040_map.json")
    parser.add_argument("--refs", type=Path, default=PROJECT / "workspace" / "hour_benchmark_20260721" / "refs")
    parser.add_argument(
        "--bank",
        help="Explicit bank name when the source map uses a generic event label.",
    )
    parser.add_argument(
        "--profile", choices=("anime", "vn"), default="vn",
        help="Generation/timing profile. Both profiles generate one take per "
             "line and only retry QA failures.",
    )
    parser.add_argument("--mount-only", action="store_true")
    parser.add_argument("--qa-ids", nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Write a non-destructive production branch outside PROJECT/outputs.",
    )
    parser.add_argument(
        "--regenerate-ids", nargs="+",
        help="Generate one fresh take for these IDs and retry only QA failures "
             "up to the selected profile's max_rounds.",
    )
    parser.add_argument("--next", dest="next_bank")
    args = parser.parse_args()
    map_path = args.map if args.map.is_absolute() else (PROJECT / args.map).resolve()
    refs = args.refs if args.refs.is_absolute() else (PROJECT / args.refs).resolve()
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    bank = args.bank or re.sub(r"_(?:en|de)$", "", mapping["event"], flags=re.I)
    prefix_match = re.match(r"(\d+_\d+)", bank)
    if not prefix_match:
        raise ValueError(f"cannot derive line prefix from {bank}")
    prefix = prefix_match.group(1)
    next_bank = args.next_bank or {"100_040_B": "100_050_B", "100_050_B": "100_060_C"}.get(bank, "next chronological bank")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT / "outputs"
    )
    out = output_root / bank
    out.mkdir(parents=True, exist_ok=True)
    scene, stem, stem_sr, cue_paths = build_scene(map_path, refs, out, bank, prefix)
    profile = dict(config[args.profile])
    profile["model"] = config["model"]
    keep = sum(classify_line(line["source_text"], line.get("delivery_text", line["target_text"])).action == KEEP_ORIGINAL for line in scene["lines"])
    total = len(scene["lines"]) - keep
    if args.regenerate_ids:
        wanted = set(args.regenerate_ids)
        known = {line["id"] for line in scene["lines"]}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"unknown line IDs: {unknown}")
        metadata_path = out / "candidates" / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists() else []
        )
        next_round = 1 + max(
            (int(row["round"]) for row in metadata if row["line_id"] in wanted),
            default=0,
        )
        generated_rounds = []
        runtime = load_generation_runtime(profile)
        try:
            generate_round(scene, stem, stem_sr, out, profile, next_round, wanted, runtime=runtime)
            generated_rounds.append(next_round)
            rankings_path = out / "QA_RANKING.json"
            rankings = (
                json.loads(rankings_path.read_text(encoding="utf-8"))
                if rankings_path.exists() else {}
            )
            focused, failed = evaluate(
                scene, stem, stem_sr, out, config["qa"], wanted,
                use_asr=False,
            )
            rankings.update(focused)
            failed = retryable_ids(focused)
            max_rounds = int(profile.get("max_rounds", 2))
            while failed and len(generated_rounds) < max_rounds:
                next_round += 1
                generate_round(scene, stem, stem_sr, out, profile, next_round, failed, runtime=runtime)
                generated_rounds.append(next_round)
                focused, failed = evaluate(
                    scene, stem, stem_sr, out, config["qa"], wanted,
                    use_asr=False,
                )
                rankings.update(focused)
                failed = retryable_ids(focused)
            rankings_path.write_text(
                json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            if failed:
                raise RuntimeError(
                    f"no QA-passing take after {len(generated_rounds)} focused "
                    f"attempts: {sorted(failed)}"
                )
        finally:
            runtime.model = None
            runtime.prompt_cache.clear()
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
        report = select_cues(scene, cue_paths, out, rankings, strict_ids=wanted)
        make_html(scene, report, out)
        print(json.dumps({
            "bank": bank,
            "regenerated_ids": sorted(wanted),
            "profile": args.profile,
            "rounds": generated_rounds,
            "contracts": report["all_contracts_pass"],
        }, indent=2))
        return
    if args.qa_ids:
        rankings_path = out / "QA_RANKING.json"
        rankings = json.loads(rankings_path.read_text(encoding="utf-8")) if rankings_path.exists() else {}
        focused, _ = evaluate(
            scene, stem, stem_sr, out, config["qa"], set(args.qa_ids),
            use_asr=False,
        )
        rankings.update(focused)
        rankings_path.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
        report = select_cues(
            scene, cue_paths, out, rankings, strict_ids=set(args.qa_ids),
        )
        make_html(scene, report, out)
        passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
        review = total - passed
        write_state(scene["scene"], f"Completed {bank}; next chronological bank {next_bank}", {"generated_lines": total, "keep_original": keep, "qa_pass": passed, "qa_review": review})
        print(json.dumps({"bank": bank, "focused_qa": args.qa_ids, "pass": passed, "review": review, "keep": keep}, indent=2))
        return
    if args.mount_only:
        rankings = json.loads((out / "QA_RANKING.json").read_text(encoding="utf-8"))
        report = select_cues(scene, cue_paths, out, rankings)
        make_html(scene, report, out)
        passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
        review = total - passed
        write_state(scene["scene"], f"Completed {bank}; next chronological bank {next_bank}", {"generated_lines": total, "keep_original": keep, "qa_pass": passed, "qa_review": review})
        print(json.dumps({"bank": scene["scene"], "mount_only": True, "pass": passed, "review": review, "contracts": report["all_contracts_pass"], "clipped": report["clipped_samples"]}, indent=2))
        return
    write_state(scene["scene"], f"Generating VN initial take for {bank}", {"generated_lines": 0, "keep_original": keep, "qa_pass": 0, "qa_review": 0})
    rankings = {}
    failed = {
        line["id"] for line in scene["lines"]
        if classify_line(
            line["source_text"], line.get("delivery_text", line["target_text"]),
        ).action != KEEP_ORIGINAL
    }
    max_rounds = int(profile.get("max_rounds", 4))
    runtime = load_generation_runtime(profile)
    try:
        for round_index in range(max_rounds):
            if not failed:
                break
            write_state(
                scene["scene"],
                f"Generating adaptive round {round_index + 1}/{max_rounds} "
                f"for {len(failed)} cues",
            )
            generate_round(
                scene, stem, stem_sr, out, profile, round_index,
                failed if round_index else None, runtime=runtime,
            )
            write_state(
                scene["scene"],
                f"Advanced GPU QA round {round_index + 1}/{max_rounds} for {bank}",
            )
            focused, failed = evaluate(
                scene, stem, stem_sr, out, config["qa"],
                only_ids=failed, use_asr=False,
            )
            rankings.update(focused)
            failed = retryable_ids(focused)
            (out / "retry_ids.json").write_text(
                json.dumps(sorted(failed), indent=2), encoding="utf-8",
            )
    finally:
        runtime.model = None
        runtime.prompt_cache.clear()
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    (out / "QA_RANKING.json").write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
    report = select_cues(scene, cue_paths, out, rankings)
    make_html(scene, report, out)
    passed = sum(bool(row.get("pass", True)) for row in report["lines"] if row["action"] != KEEP_ORIGINAL)
    review = total - passed
    write_state(scene["scene"], f"Completed {bank}; next chronological bank {next_bank}", {"generated_lines": total, "keep_original": keep, "qa_pass": passed, "qa_review": review})
    print(json.dumps({"bank": scene["scene"], "pass": passed, "review": review, "keep": keep, "contracts": report["all_contracts_pass"], "active_tail_adjustments": report["tail_adjustments_touching_active"]}, indent=2))


if __name__ == "__main__":
    main()
