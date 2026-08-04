#!/usr/bin/env python3
"""Auditoría reproducible de todo el doblaje de P3R.

Fase ``technical`` (CPU): inventario y métricas acústicas baratas sobre todo
el plan, sin sobrescribir producción. Fase ``asr`` (GPU): contenido alemán con
Whisper large-v3-turbo. El resultado es un manifiesto de regeneración; este
script nunca promueve ni reemplaza audio.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "full_game_qa_20260724"
TECH = OUT / "technical.jsonl"
ASR = OUT / "asr.jsonl"
DEEP = OUT / "deep.jsonl"
SUMMARY = OUT / "summary.json"
RETRY = OUT / "retry_ids.json"
TECH_RETRY = OUT / "retry_technical_ids.json"
ASR_RETRY = OUT / "retry_asr_ids.json"
DEEP_RETRY = OUT / "retry_deep_ids.json"

sys.path.insert(0, str(ROOT))
import line_policy as lp  # noqa: E402
import prod_dub  # noqa: E402


def mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return np.asarray(y.mean(axis=1), dtype=np.float32), int(sr)


def mono_tail(path: Path, seconds: float = 1.0) -> tuple[np.ndarray, int]:
    """Read only the endpoint needed by the release-only refresh phase."""
    with sf.SoundFile(path) as handle:
        sr = int(handle.samplerate)
        frames = min(len(handle), max(1, round(seconds * sr)))
        handle.seek(len(handle) - frames)
        y = handle.read(frames, dtype="float32", always_2d=True)
    return np.asarray(y.mean(axis=1), dtype=np.float32), sr


def rms_db(value: float, peak: float) -> float:
    return float(20.0 * math.log10(max(value, 1e-12) / max(peak, 1e-12)))


def endpoint_release(y: np.ndarray, sr: int, peak: float) -> dict:
    """Detect a cut hidden by postprocess-added digital silence.

    ``physical_tail_ms`` (computed from the -40 dB activity envelope) catches
    files that simply end while still active.  It cannot distinguish a real
    decay from a short hard fade followed by appended zeros.  Measure the
    signal immediately before the digital-silence boundary as a second,
    independent gate.

    Thresholds were calibrated against the known-bad old 100_040 L010/L011
    renders and their repaired counterparts.  Natural short endings in the
    same bank can still be energetic 15 ms before the boundary, but decay
    progressively; the bad 12 ms emergency fade loses >=24 dB in its last
    12.5 ms.  A genuinely hard edge is also rejected when its final 2.5 ms
    remain louder than -26 dB.
    """
    if not len(y) or peak <= 1e-8:
        return {
            "digital_padding_ms": 0.0,
            "edge_rms_db": {"2.5": -120.0, "15": -120.0},
            "edge_collapse_db": 0.0,
            "abrupt_release": False,
            "abrupt_release_kind": None,
        }
    digital_floor = max(1e-8, peak * (10.0 ** (-90.0 / 20.0)))
    audible = np.flatnonzero(np.abs(y) > digital_floor)
    boundary = int(audible[-1]) + 1 if len(audible) else 0
    padding_ms = max(0.0, (len(y) - boundary) / sr * 1000.0)

    def edge_db(ms: float) -> float:
        frames = max(1, round(ms * sr / 1000.0))
        section = y[max(0, boundary - frames):boundary]
        if not len(section):
            return -120.0
        value = math.sqrt(
            float(np.mean(section.astype(np.float64) ** 2)) + 1e-18
        )
        return rms_db(value, peak)

    edge_2p5 = edge_db(2.5)
    edge_15 = edge_db(15.0)
    collapse_db = max(0.0, edge_15 - edge_2p5)
    has_padding = padding_ms >= 35.0
    hard_step = has_padding and edge_2p5 > -26.0
    emergency_fade = bool(
        has_padding
        and edge_15 > -32.0
        and edge_2p5 <= -48.0
        and collapse_db >= 24.0
    )
    kind = (
        "hard_step" if hard_step
        else "emergency_fade" if emergency_fade
        else None
    )
    return {
        "digital_padding_ms": padding_ms,
        "edge_rms_db": {"2.5": edge_2p5, "15": edge_15},
        "edge_collapse_db": collapse_db,
        "abrupt_release": bool(hard_step or emergency_fade),
        "abrupt_release_kind": kind,
    }


def intervals(y: np.ndarray, sr: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """10 ms RMS / 5 ms hop intervals, relative -40 dB activity."""
    frame = max(32, round(0.010 * sr))
    hop = max(16, round(0.005 * sr))
    if not len(y):
        return [], np.zeros(0, dtype=np.float64)
    padded = np.pad(y.astype(np.float64), (0, max(0, frame - len(y))))
    starts = np.arange(0, max(1, len(padded) - frame + 1), hop)
    env = np.asarray([
        math.sqrt(float(np.mean(padded[s:s + frame] ** 2)) + 1e-18)
        for s in starts
    ])
    peak = float(np.max(np.abs(y), initial=0.0))
    active = env >= max(peak * 0.01, 1e-5)
    result: list[tuple[int, int]] = []
    start = None
    for i, yes in enumerate(active):
        if yes and start is None:
            start = int(starts[i])
        if not yes and start is not None:
            result.append((start, min(len(y), int(starts[i - 1] + frame))))
            start = None
    if start is not None:
        result.append((start, min(len(y), int(starts[-1] + frame))))
    return result, env


def profile(path: Path) -> dict:
    y, sr = mono(path)
    peak = float(np.max(np.abs(y), initial=0.0))
    ints, _ = intervals(y, sr)
    if not ints or peak <= 1e-8:
        return {
            "empty": True, "sample_rate": sr, "frames": len(y),
            "duration": len(y) / sr if sr else 0.0, "peak": peak,
        }
    onset = ints[0][0] / sr
    active_end = ints[-1][1]
    span = (active_end - ints[0][0]) / sr
    active_seconds = sum(end - start for start, end in ints) / sr
    gaps = [
        (ints[i + 1][0] - ints[i][1]) / sr
        for i in range(len(ints) - 1)
        if (ints[i + 1][0] - ints[i][1]) / sr >= 0.075
    ]
    tail_db = {}
    for ms in (5, 10, 20, 40, 80, 120):
        n = min(len(y), max(1, round(sr * ms / 1000)))
        value = math.sqrt(float(np.mean(y[-n:].astype(np.float64) ** 2)) + 1e-18)
        tail_db[str(ms)] = rms_db(value, peak)
    physical_tail_ms = max(0.0, (len(y) - active_end) / sr * 1000.0)
    end_sample_db = rms_db(abs(float(y[-1])), peak)
    # A natural close may have little literal padding, but its last 20/40 ms
    # must already be quiet. Conversely, appended zeros do not absolve a loud
    # edge: the active-tail metric below measures before the final interval.
    guard = min(active_end, max(1, round(0.080 * sr)))
    active_tail = y[max(0, active_end - guard):active_end]
    active_tail_rms = math.sqrt(
        float(np.mean(active_tail.astype(np.float64) ** 2)) + 1e-18
    )
    active_tail_db = rms_db(active_tail_rms, peak)
    endpoint = endpoint_release(y, sr, peak)
    release_ok = bool(
        not endpoint["abrupt_release"]
        and (
        physical_tail_ms >= 35.0
        or (tail_db["20"] <= -30.0 and end_sample_db <= -40.0)
        )
    )
    clipped = np.abs(y) >= 0.999
    longest_clip_run = 0
    current_clip_run = 0
    for value in clipped:
        if value:
            current_clip_run += 1
            longest_clip_run = max(longest_clip_run, current_clip_run)
        else:
            current_clip_run = 0
    return {
        "empty": False, "sample_rate": sr, "frames": len(y),
        "duration": len(y) / sr, "peak": peak,
        "clipping_pct": float(np.mean(clipped) * 100.0),
        "longest_clipped_run": int(longest_clip_run),
        "dc": float(np.mean(y)), "onset": onset, "span": span,
        "active_seconds": active_seconds, "physical_tail_ms": physical_tail_ms,
        "tail_rms_db": tail_db, "active_tail_rms_db": active_tail_db,
        "end_sample_db": end_sample_db, "gaps": gaps,
        "endpoint_release": endpoint,
        "active_rms_dbfs": float(20.0 * math.log10(
            math.sqrt(float(np.mean(
                np.concatenate([y[a:b] for a, b in ints]).astype(np.float64) ** 2
            )) + 1e-18) + 1e-12
        )),
        "release_ok": release_ok,
    }


def pause_error(left: list[float], right: list[float]) -> float:
    a = sorted(left, reverse=True)[:3]
    b = sorted(right, reverse=True)[:3]
    count = max(len(a), len(b), 1)
    a += [0.0] * (count - len(a))
    b += [0.0] * (count - len(b))
    return sum(abs(x - y) for x, y in zip(a, b)) / count


def has_strong_pause(text: str) -> bool:
    return bool(__import__("re").search(r"\.{2,}|[;:!?]", text or ""))


def technical_row(rec: dict, cine: dict) -> dict:
    stem = f"{rec['event']}_L{int(rec['stream_index']):03d}"
    wav = ROOT / "produccion" / f"{stem}.wav"
    planned = prod_dub.plan_line(rec, cine)
    row = {
        "id": stem, "event": rec["event"], "stream": int(rec["stream_index"]),
        "text_en": rec.get("text_en") or "", "text_de": rec.get("text_de") or "",
        "policy": planned["accion"], "cine": bool(planned.get("corrige_timing")),
        "output": str(wav), "exists": wav.exists(),
    }
    if planned["accion"] == "conservar_original":
        # KEEP_ORIGINAL means the mod deliberately ships no replacement and
        # lets the game's original bank play.  A missing production WAV is
        # therefore correct, not a missing deliverable.
        row.update({"status": "KEEP_ORIGINAL", "failure_codes": []})
        if wav.exists():
            row["metrics"] = profile(wav)
        return row
    if not wav.exists():
        row.update({"status": "MISSING", "failure_codes": ["MISSING_OUTPUT"]})
        return row
    failures: list[str] = []
    try:
        generated = profile(wav)
    except Exception as exc:  # corrupt/unreadable file
        row.update({
            "status": "FAIL", "failure_codes": ["UNREADABLE"],
            "error": str(exc),
        })
        return row
    if generated.get("empty"):
        failures.append("EMPTY")
    else:
        # One isolated PCM full-scale sample is not automatically audible
        # clipping. Fail sustained flattening or a material percentage.
        if (
            generated["clipping_pct"] > 0.15
            or generated["longest_clipped_run"] >= 8
        ):
            failures.append("CLIPPING")
        if abs(generated["dc"]) > 0.010:
            failures.append("DC")
        if not generated["release_ok"]:
            failures.append("TAIL_RELEASE")
    ref_path = prod_dub.resolve_ref(rec)
    source = None
    if ref_path and Path(ref_path).exists():
        try:
            source = profile(Path(ref_path))
        except Exception:
            source = None
    if source and not source.get("empty") and not generated.get("empty"):
        onset_error_ms = (generated["onset"] - source["onset"]) * 1000.0
        span_error_ms = abs(generated["span"] - source["span"]) * 1000.0
        pmae_ms = pause_error(source["gaps"], generated["gaps"]) * 1000.0
        level_delta_db = generated["active_rms_dbfs"] - source["active_rms_dbfs"]
        comparison = {
            "source": str(ref_path), "source_profile": source,
            "onset_error_ms": onset_error_ms, "span_error_ms": span_error_ms,
            "pause_mae_ms": pmae_ms, "active_level_delta_db": level_delta_db,
        }
        if abs(level_delta_db) > 6.0:
            failures.append("LEVEL")
        # A quiet/natural lead may exist in both recordings. Artificial delay
        # is specifically extra delay introduced relative to the actor.
        if onset_error_ms > 100.0:
            failures.append("ARTIFICIAL_DELAY")
        if row["cine"]:
            if abs(onset_error_ms) > 80.0:
                failures.append("CINE_ONSET")
            if span_error_ms > 650.0:
                failures.append("CINE_SPAN")
            if (
                has_strong_pause(row["text_en"])
                and has_strong_pause(row["text_de"])
                and pmae_ms > 200.0
            ):
                failures.append("PAUSE")
        row["comparison"] = comparison
    else:
        row["comparison"] = {"source": ref_path}
        failures.append("SIN_REF_QA")
        if not generated.get("empty") and generated["onset"] > 0.150:
            failures.append("ARTIFICIAL_DELAY")
    row.update({
        "status": "FAIL_TECH" if failures else "PENDING_ASR",
        "failure_codes": sorted(set(failures)), "metrics": generated,
        "oder_ok": bool(generated.get("release_ok", False)),
        "general_metric_technical_ok": not any(
            code != "TAIL_RELEASE" for code in failures
        ),
    })
    return row


def write_summary(rows: list[dict], phase: str, elapsed: float) -> None:
    status = Counter(row["status"] for row in rows)
    failures = Counter(
        code for row in rows for code in row.get("failure_codes", [])
    )
    reviews = Counter(
        code for row in rows for code in row.get("review_codes", [])
    )
    summary = {
        "phase": phase, "rows": len(rows), "status": dict(status),
        "failure_codes": dict(failures), "elapsed_seconds": elapsed,
        "review_codes": dict(reviews),
        "technical_manifest": str(TECH), "asr_manifest": str(ASR),
        "deep_manifest": str(DEEP),
        "retry_manifest": str(RETRY),
        "technical_retry_manifest": str(TECH_RETRY),
        "asr_retry_manifest": str(ASR_RETRY),
        "deep_retry_manifest": str(DEEP_RETRY),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def write_retry_manifests(extra: list[str], phase_path: Path) -> None:
    """Keep technical rejects when a later QA phase writes its own failures."""
    phase_ids = sorted(set(extra))
    phase_path.write_text(
        json.dumps(phase_ids, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    technical_ids: list[str] = []
    if TECH.exists():
        for line in TECH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("status") == "FAIL_TECH"
                and any(
                    code != "SIN_REF_QA"
                    for code in row.get("failure_codes", [])
                )
            ):
                technical_ids.append(row["id"])
    TECH_RETRY.write_text(
        json.dumps(
            sorted(set(technical_ids)), ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    RETRY.write_text(
        json.dumps(
            sorted(set(technical_ids) | set(phase_ids)),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def run_technical(limit: int | None) -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cine = json.loads(prod_dub.CINE.read_text(encoding="utf-8"))
    records = prod_dub._sorted_by_chronology(prod_dub.load_corpus())
    if limit:
        records = records[:limit]
    rows = []
    with TECH.open("w", encoding="utf-8") as handle:
        for index, rec in enumerate(records, 1):
            row = technical_row(rec, cine)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 250 == 0:
                handle.flush()
                print(
                    f"TECH {index}/{len(records)} "
                    f"fail={sum(r['status']=='FAIL_TECH' for r in rows)} "
                    f"missing={sum(r['status']=='MISSING' for r in rows)}",
                    flush=True,
                )
    retry = [
        row["id"] for row in rows
        if row["status"] == "FAIL_TECH"
        and any(code != "SIN_REF_QA" for code in row["failure_codes"])
    ]
    write_retry_manifests(retry, TECH_RETRY)
    write_summary(rows, "technical", time.perf_counter() - started)


def run_release(limit: int | None) -> None:
    """Refresh only the global ``oder`` gate in an existing technical scan."""
    started = time.perf_counter()
    rows = [
        json.loads(line) for line in TECH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = [
        row for row in rows
        if (
            row.get("exists")
            and row.get("policy") != "conservar_original"
            and not row.get("metrics", {}).get("empty", True)
        )
    ]
    selected = eligible[:limit] if limit else eligible
    selected_ids = {row["id"] for row in selected}
    done = 0
    for row in rows:
        if row["id"] not in selected_ids:
            continue
        done += 1
        metrics = row["metrics"]
        try:
            y, sr = mono_tail(Path(row["output"]))
            peak = float(metrics.get("peak", 0.0))
            if peak <= 1e-8:
                peak = float(np.max(np.abs(y), initial=0.0))
            endpoint = endpoint_release(y, sr, peak)
        except Exception as exc:
            endpoint = {
                "abrupt_release": True,
                "abrupt_release_kind": "unreadable_endpoint",
                "error": str(exc),
            }
        metrics["endpoint_release"] = endpoint
        base_release_ok = bool(
            metrics.get("physical_tail_ms", 0.0) >= 35.0
            or (
                metrics.get("tail_rms_db", {}).get("20", 0.0) <= -30.0
                and metrics.get("end_sample_db", 0.0) <= -40.0
            )
        )
        release_ok = bool(
            base_release_ok and not endpoint.get("abrupt_release", False)
        )
        metrics["release_ok"] = release_ok
        failures = [
            code for code in row.get("failure_codes", [])
            if code != "TAIL_RELEASE"
        ]
        if not release_ok:
            failures.append("TAIL_RELEASE")
        row["failure_codes"] = sorted(set(failures))
        row["oder_ok"] = release_ok
        row["general_metric_technical_ok"] = not any(
            code != "TAIL_RELEASE" for code in row["failure_codes"]
        )
        row["status"] = "FAIL_TECH" if row["failure_codes"] else "PENDING_ASR"
        if done % 500 == 0:
            print(f"RELEASE {done}/{len(selected)}", flush=True)

    backup = TECH.with_name("technical.pre_endpoint_release.jsonl")
    if not backup.exists():
        shutil.copy2(TECH, backup)
    temporary = TECH.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(TECH)
    retry = [
        row["id"] for row in rows
        if row["status"] == "FAIL_TECH"
        and any(code != "SIN_REF_QA" for code in row["failure_codes"])
    ]
    write_retry_manifests(retry, TECH_RETRY)
    write_summary(rows, "release", time.perf_counter() - started)


def run_policy_reclassify() -> None:
    """Fix KEEP_ORIGINAL semantics in an existing technical manifest.

    This is a manifest-only operation: it does not rescan audio and does not
    touch production WAVs.
    """
    started = time.perf_counter()
    rows = [
        json.loads(line) for line in TECH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        if row.get("policy") != "conservar_original":
            continue
        if row.get("status") != "KEEP_ORIGINAL" or row.get("failure_codes"):
            changed += 1
        row["status"] = "KEEP_ORIGINAL"
        row["failure_codes"] = []
        row["oder_ok"] = True
        row["general_metric_technical_ok"] = True

    backup = TECH.with_name("technical.pre_keep_original_fix.jsonl")
    if not backup.exists():
        shutil.copy2(TECH, backup)
    temporary = TECH.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(TECH)
    retry = [
        row["id"] for row in rows
        if row["status"] == "FAIL_TECH"
        and any(code != "SIN_REF_QA" for code in row.get("failure_codes", []))
    ]
    write_retry_manifests(retry, TECH_RETRY)
    write_summary(rows, "policy_reclassify", time.perf_counter() - started)
    print(f"KEEP_ORIGINAL reclassified: {changed}", flush=True)


def asr_selection_reason(row: dict, mode: str) -> str | None:
    if mode == "all":
        return "all"
    # A technical reject is regenerated regardless of its transcript. Running
    # Whisper on the discarded take wastes GPU; content QA belongs on the new
    # candidate after it clears the acoustic gates.
    if row["status"] == "FAIL_TECH":
        return None
    bucket = int(hashlib.sha1(row["id"].encode("utf-8")).hexdigest()[:8], 16)
    if row.get("policy") in {"empalme_inicial", "empalme_final"}:
        return "splice"
    # Proper names, numbers and identical EN/DE spoken text are where copied
    # English or pronunciation failures matter most. German capitalises every
    # noun, so uppercase German tokens alone are *not* a name detector. Use
    # capitalised mid-sentence English tokens that survive into German.
    target_tokens = re.findall(
        r"[^\W\d_]+|\d+", row.get("text_de") or "", re.UNICODE,
    )
    source_tokens = re.findall(
        r"[^\W\d_]+|\d+", row.get("text_en") or "", re.UNICODE,
    )
    target_folded = {token.casefold() for token in target_tokens}
    shared_english_name = any(
        len(token) >= 3
        and token[:1].isupper()
        and token.casefold() in target_folded
        for token in source_tokens[1:]
    )
    if lp.words(row.get("text_en") or "") == lp.words(row.get("text_de") or ""):
        # ASR cannot prove that an acoustically identical word was re-cloned;
        # speaker/reference checks are the useful gate. Keep only a control
        # sample here instead of transcribing every "Okay"/name call.
        if mode == "targeted" or bucket % 20 == 0:
            return "identical_spoken_text"
        return None
    word_count = len(lp.words(row.get("text_de") or ""))
    if word_count <= 3:
        if mode == "targeted" or bucket % 20 == 0:
            return "short_line"
        return None
    if shared_english_name or any(
        any(char.isdigit() for char in token) for token in target_tokens
    ):
        if mode == "targeted" or bucket % 10 == 0:
            return "name_or_number"
        return None
    # ``sampled`` is the scalable default for very large games: 5% of short
    # lines, 10% of name/number risks, all rare splices/identical text and 1%
    # of other clean dialogue. Expand only the affected bank/speaker if this
    # control sample reveals a real systematic content problem.
    modulus = 20 if mode == "targeted" else 100
    if bucket % modulus == 0:
        return (
            "long_clean_sample_5pct"
            if mode == "targeted"
            else "long_clean_sample_1pct"
        )
    return None


def asr_content_decision(
    row: dict, expected: list[str], got: list[str], wer: float,
) -> tuple[str, dict]:
    """Separate confident content failures from ordinary ASR ambiguity."""
    expected_joined = "".join(expected)
    got_joined = "".join(got)
    similarity = difflib.SequenceMatcher(
        None, expected_joined, got_joined,
    ).ratio()
    expected_count = max(len(expected), 1)
    length_ratio = len(got) / expected_count
    source = lp.words(row.get("text_en") or "")
    source_wer = (
        prod_dub._qa_edit_distance(source, got) / max(len(source), 1)
        if source else None
    )
    # The English take survived when its transcript is materially closer to
    # EN than DE. Do not apply this to identical/cognate targets.
    english_leak = bool(
        source
        and source != expected
        and source_wer is not None
        and source_wer + 0.25 < wer
    )
    gross_length = bool(
        len(got) == 0
        or length_ratio > 3.0
        or (len(expected) >= 5 and length_ratio < 0.45)
    )
    gross_mismatch = bool(
        len(expected) >= 4 and wer > 0.65 and similarity < 0.45
    )
    details = {
        "character_similarity": similarity,
        "word_length_ratio": length_ratio,
        "source_wer": source_wer,
        "english_leak": english_leak,
        "gross_length": gross_length,
        "gross_mismatch": gross_mismatch,
    }
    if wer == 0.0:
        return "PASS", details
    # Short commands, vocalisations and names are precisely where Whisper is
    # least stable. Preserve the warning, but never regenerate from it alone.
    if row.get("asr_selection_reason") in {"short_line", "name_or_number"}:
        return "REVIEW", details
    if english_leak or gross_length or gross_mismatch:
        return "FAIL", details
    return "REVIEW", details


def run_asr(limit: int | None, resume: bool, mode_select: str) -> None:
    from asr_gpu import cargar_modelo

    started = time.perf_counter()
    rows = [
        json.loads(line) for line in TECH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = []
    for row in rows:
        if (
            row["status"] not in {"PENDING_ASR", "FAIL_TECH"}
            or not row["exists"]
            or row["policy"] == "conservar_original"
        ):
            continue
        reason = asr_selection_reason(row, mode_select)
        if reason:
            row["asr_selection_reason"] = reason
            candidates.append(row)
    if limit:
        candidates = candidates[:limit]
    completed: dict[str, dict] = {}
    if resume and ASR.exists():
        completed = {
            row["id"]: row
            for row in (
                json.loads(line) for line in ASR.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    mode = "a" if completed else "w"
    model = cargar_modelo("large-v3-turbo")
    with ASR.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(candidates, 1):
            if row["id"] in completed:
                continue
            segments, _ = model.transcribe(
                row["output"], language="de", beam_size=1, vad_filter=False,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments)
            expected = lp.words(row["text_de"])
            got = lp.words(transcript)
            wer = prod_dub._qa_edit_distance(expected, got) / max(len(expected), 1)
            limit_wer = 0.0 if len(expected) <= 3 else 0.08
            last_word_ok = bool(got) and bool(expected) and got[-1] == expected[-1]
            # Beam 5 is a confirmation pass, not the default cost. This keeps
            # Whisper hallucinations/greedy misses from triggering needless
            # OmniVoice retries.
            if wer > limit_wer or not last_word_ok:
                segments, _ = model.transcribe(
                    row["output"], language="de", beam_size=5,
                    vad_filter=False, condition_on_previous_text=False,
                )
                transcript = " ".join(
                    segment.text.strip() for segment in segments
                )
                got = lp.words(transcript)
                wer = prod_dub._qa_edit_distance(
                    expected, got,
                ) / max(len(expected), 1)
                last_word_ok = (
                    bool(got) and bool(expected) and got[-1] == expected[-1]
                )
            failures = [
                code for code in row.get("failure_codes", [])
                if code not in {"TEXT", "LAST_WORD", "ASR_CONTENT"}
            ]
            content_status, content_details = asr_content_decision(
                row, expected, got, wer,
            )
            review_codes = []
            if content_status == "FAIL":
                failures.append("ASR_CONTENT")
            elif content_status == "REVIEW":
                review_codes.append("ASR_AMBIGUOUS")
            row.update({
                "transcript": transcript, "wer": wer,
                "last_word_ok": last_word_ok,
                "asr_content_status": content_status,
                "asr_content_details": content_details,
                "review_codes": review_codes,
                "failure_codes": sorted(set(failures)),
            })
            # ``oder`` is the acoustic release gate. Whisper is deliberately
            # independent: a wrong ASR last word must not turn into a fake
            # tail-cut failure.
            row["oder_ok"] = bool(row.get("oder_ok", False))
            row["general_metric_ok"] = not any(
                code != "TAIL_RELEASE"
                for code in row["failure_codes"]
            )
            row["status"] = (
                "FAIL"
                if not row["general_metric_ok"] or not row["oder_ok"]
                else "PASS"
            )
            completed[row["id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                handle.flush()
            if index % 100 == 0:
                print(
                    f"ASR {index}/{len(candidates)} "
                    f"fail={sum(r['status']=='FAIL' for r in completed.values())}",
                    flush=True,
                )
    final = list(completed.values())
    retry = [
        row["id"] for row in final
        if row["status"] == "FAIL"
        and any(code != "SIN_REF_QA" for code in row["failure_codes"])
    ]
    write_retry_manifests(retry, ASR_RETRY)
    write_summary(final, "asr", time.perf_counter() - started)


def run_asr_reclassify() -> None:
    """Apply current ASR semantics without repeating GPU transcription."""
    started = time.perf_counter()
    rows = [
        json.loads(line) for line in ASR.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        expected = lp.words(row.get("text_de") or "")
        got = lp.words(row.get("transcript") or "")
        wer = prod_dub._qa_edit_distance(
            expected, got,
        ) / max(len(expected), 1)
        content_status, details = asr_content_decision(
            row, expected, got, wer,
        )
        failures = [
            code for code in row.get("failure_codes", [])
            if code not in {"TEXT", "LAST_WORD", "ASR_CONTENT"}
        ]
        reviews = []
        if content_status == "FAIL":
            failures.append("ASR_CONTENT")
        elif content_status == "REVIEW":
            reviews.append("ASR_AMBIGUOUS")
        row.update({
            "wer": wer,
            "last_word_ok": bool(got) and bool(expected)
            and got[-1] == expected[-1],
            "asr_content_status": content_status,
            "asr_content_details": details,
            "review_codes": reviews,
            "failure_codes": sorted(set(failures)),
        })
        row["oder_ok"] = bool(row.get("metrics", {}).get(
            "release_ok", row.get("oder_ok", False),
        ))
        row["general_metric_ok"] = not any(
            code != "TAIL_RELEASE" for code in row["failure_codes"]
        )
        row["status"] = (
            "PASS"
            if row["general_metric_ok"] and row["oder_ok"]
            else "FAIL"
        )
    backup = ASR.with_name("asr.pre_reclassification.jsonl")
    if not backup.exists():
        shutil.copy2(ASR, backup)
    temporary = ASR.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(ASR)
    retry = [row["id"] for row in rows if row["status"] == "FAIL"]
    write_retry_manifests(retry, ASR_RETRY)
    write_summary(rows, "asr_reclassify", time.perf_counter() - started)


def deep_features(path: Path) -> dict:
    """Pitch/energy/spectral descriptors for cross-language fidelity QA."""
    import librosa

    y, sr = mono(path)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    if len(y) < round(0.20 * sr) or float(np.max(np.abs(y), initial=0.0)) < 1e-5:
        return {"enough_voice": False}
    ints, _ = intervals(y, sr)
    if not ints:
        return {"enough_voice": False}
    start, end = ints[0][0], ints[-1][1]
    z = y[start:end]
    if len(z) < round(0.20 * sr):
        return {"enough_voice": False}
    frame_length, hop = 1024, 256
    f0 = librosa.yin(
        z, fmin=65.0, fmax=500.0, sr=sr,
        frame_length=frame_length, hop_length=hop,
    )
    rms = librosa.feature.rms(
        y=z, frame_length=frame_length, hop_length=hop,
    )[0]
    count = min(len(f0), len(rms))
    f0, rms = f0[:count], rms[:count]
    voiced = (
        np.isfinite(f0)
        & (f0 > 65.0)
        & (f0 < 490.0)
        & (rms >= max(float(np.max(rms, initial=0.0)) * 0.03, 1e-5))
    )
    pitch = f0[voiced]
    centroid = librosa.feature.spectral_centroid(
        y=z, sr=sr, n_fft=1024, hop_length=hop,
    )[0]
    result = {
        "enough_voice": len(pitch) >= 5,
        "voiced_frames": int(len(pitch)),
        "f0_median_hz": float(np.median(pitch)) if len(pitch) else None,
        "spectral_centroid_median_hz": (
            float(np.median(centroid)) if len(centroid) else None
        ),
    }
    if len(pitch) >= 8:
        low, high = np.percentile(pitch, [10, 90])
        result["pitch_range_st"] = float(
            12.0 * math.log2(max(high, 1e-6) / max(low, 1e-6))
        )
    else:
        result["pitch_range_st"] = None
    # Normalised energy contour is a soft acting cue, not a hard gate.
    if len(rms):
        x_old = np.linspace(0.0, 1.0, len(rms))
        x_new = np.linspace(0.0, 1.0, 50)
        contour = np.interp(x_new, x_old, rms)
        contour = (contour - np.mean(contour)) / (np.std(contour) + 1e-8)
        result["energy_contour"] = contour.astype(float).tolist()
    return result


def correlation(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    a, b = np.asarray(left), np.asarray(right)
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_deep(limit: int | None, resume: bool) -> None:
    started = time.perf_counter()
    rows = [
        json.loads(line) for line in ASR.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]
    completed: dict[str, dict] = {}
    if resume and DEEP.exists():
        completed = {
            row["id"]: row
            for row in (
                json.loads(line) for line in DEEP.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    mode = "a" if completed else "w"
    with DEEP.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            source_path = (row.get("comparison") or {}).get("source")
            generated = deep_features(Path(row["output"]))
            source = (
                deep_features(Path(source_path))
                if source_path and Path(source_path).exists()
                else {"enough_voice": False}
            )
            failures = list(row.get("failure_codes", []))
            f0_ratio = None
            spectral_ratio = None
            if (
                generated.get("f0_median_hz")
                and source.get("f0_median_hz")
            ):
                f0_ratio = (
                    generated["f0_median_hz"] / source["f0_median_hz"]
                )
                if not 0.72 <= f0_ratio <= 1.30:
                    failures.append("PITCH_IDENTITY")
            if (
                generated.get("spectral_centroid_median_hz")
                and source.get("spectral_centroid_median_hz")
            ):
                spectral_ratio = (
                    generated["spectral_centroid_median_hz"]
                    / source["spectral_centroid_median_hz"]
                )
                # Wide guard: catches severe "telephone/demon/needle" output,
                # not normal cross-language articulation differences.
                if not 0.45 <= spectral_ratio <= 2.20:
                    failures.append("SPECTRAL_DISTORTION")
            energy_corr = correlation(
                source.get("energy_contour"), generated.get("energy_contour"),
            )
            source.pop("energy_contour", None)
            generated.pop("energy_contour", None)
            row["deep_metrics"] = {
                "source": source, "generated": generated,
                "f0_median_ratio": f0_ratio,
                "spectral_centroid_ratio": spectral_ratio,
                "energy_contour_correlation": energy_corr,
            }
            row["failure_codes"] = sorted(set(failures))
            row["general_metric_ok"] = not any(
                code not in {"TAIL_RELEASE", "LAST_WORD"}
                for code in row["failure_codes"]
            )
            row["status"] = (
                "PASS"
                if row["general_metric_ok"] and row.get("oder_ok", False)
                else "FAIL"
            )
            completed[row["id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                handle.flush()
                print(
                    f"DEEP {index}/{len(rows)} "
                    f"fail={sum(r['status']=='FAIL' for r in completed.values())}",
                    flush=True,
                )
    final = list(completed.values())
    retry = [row["id"] for row in final if row["status"] == "FAIL"]
    write_retry_manifests(retry, DEEP_RETRY)
    write_summary(final, "deep", time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "technical", "release", "policy_reclassify",
            "asr", "asr_reclassify", "deep",
        ),
        required=True,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--asr-mode", choices=("sampled", "targeted", "all"),
        default="sampled",
    )
    args = parser.parse_args()
    if args.phase == "technical":
        run_technical(args.limit)
    elif args.phase == "release":
        run_release(args.limit)
    elif args.phase == "policy_reclassify":
        run_policy_reclassify()
    elif args.phase == "asr":
        run_asr(args.limit, args.resume, args.asr_mode)
    elif args.phase == "asr_reclassify":
        run_asr_reclassify()
    else:
        run_deep(args.limit, args.resume)


if __name__ == "__main__":
    main()
