"""Scene-level candidate matrix and localized FMV window diagnostics."""
from __future__ import annotations
from typing import Any, Mapping, Sequence


def audit_scene_windows(audio: Any, sample_rate: int, line_windows: Sequence[Mapping[str, Any]], *, activity_threshold: float = 1e-4, dialogue_channel: int = 0, source_audio: Any | None = None, source_channel: int | None = None, require_mounted_delta_line_ids: set[str] | None = None, mounted_delta_threshold: float = 1e-4) -> dict[str, Any]:
    """Attribute activity/clipping/tail evidence to each declared line window.

    ``dialogue_channel`` is an explicit contract.  A music bed on channel 0
    cannot satisfy a dialogue gate when the configured speech channel is 1.
    When a source stem is supplied, residual energy against that source is
    recorded and can be required only for newly generated line IDs.
    """
    import numpy as np
    raw = np.asarray(audio)
    if raw.ndim not in (1, 2):
        raise ValueError("scene audio must be one- or two-dimensional")
    channel_count = 1 if raw.ndim == 1 else int(raw.shape[1])
    if int(dialogue_channel) < 0 or int(dialogue_channel) >= channel_count:
        raise ValueError(f"dialogue_channel {dialogue_channel} outside audio channel range 0..{channel_count - 1}")
    value = raw if raw.ndim == 1 else raw[:, int(dialogue_channel)]
    source_value = None
    if source_audio is not None:
        source_raw = np.asarray(source_audio)
        if source_raw.ndim not in (1, 2):
            raise ValueError("source audio must be one- or two-dimensional")
        source_channel = int(dialogue_channel if source_channel is None else source_channel)
        source_count = 1 if source_raw.ndim == 1 else int(source_raw.shape[1])
        if source_channel < 0 or source_channel >= source_count:
            raise ValueError(f"source dialogue channel {source_channel} outside audio channel range 0..{source_count - 1}")
        source_value = source_raw if source_raw.ndim == 1 else source_raw[:, source_channel]
    required_delta = {str(item) for item in (require_mounted_delta_line_ids or set())}
    results: list[dict[str, Any]] = []
    for item in sorted(line_windows, key=lambda row: (int(row.get("start", 0)), int(row.get("end", 0)))):
        line_id = str(item.get("line_id", "")); start = max(0, int(item.get("start", 0))); end = min(len(value), int(item.get("end", 0)))
        clip = value[start:end] if end > start else value[:0]
        active = np.flatnonzero(np.abs(clip) > float(activity_threshold)) if len(clip) else np.asarray([], dtype=int)
        peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
        rms = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
        active_start = (start + int(active[0])) if len(active) else None
        active_end = (start + int(active[-1]) + 1) if len(active) else None
        clipping = int(np.count_nonzero(np.abs(clip) >= .999))
        # Activity outside the line is a context diagnostic, not an automatic
        # replacement trigger; attribution fails only for a silent/clipped line.
        before = value[max(0, start - max(1, int(.03 * sample_rate))):start]
        after = value[end:min(len(value), end + max(1, int(.03 * sample_rate)))]
        context_peak = max(float(np.max(np.abs(before))) if len(before) else 0.0, float(np.max(np.abs(after))) if len(after) else 0.0)
        source_clip = source_value[start:min(end, len(source_value))] if source_value is not None else None
        if source_clip is not None and len(source_clip) != len(clip):
            source_clip = np.pad(source_clip, (0, max(0, len(clip) - len(source_clip))))[:len(clip)]
        mounted_source_delta = None
        mounted_source_delta_db = None
        if source_clip is not None:
            residual = clip - source_clip
            mounted_source_delta = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else 0.0
            mounted_source_delta_db = -120.0 if mounted_source_delta <= 1e-12 else 20.0 * float(np.log10(mounted_source_delta))
        generated_activity = bool(len(active) and (source_clip is None or (mounted_source_delta or 0.0) >= float(mounted_delta_threshold)))
        failed = []
        if not len(active): failed.append("line_activity")
        if line_id in required_delta and not generated_activity: failed.append("mounted_source_delta")
        if clipping: failed.append("line_clipping")
        results.append({"line_id": line_id, "window_start": start, "window_end": end, "active_start": active_start, "active_end": active_end, "peak": peak, "rms_dbfs": -120.0 if rms <= 1e-12 else 20.0 * float(np.log10(rms)), "clipping_samples": clipping, "context_peak": context_peak, "tail_ms": ((end - active_end) / sample_rate * 1000.0) if active_end is not None else None, "mounted_source_delta": mounted_source_delta, "mounted_source_delta_db": mounted_source_delta_db, "generated_activity": generated_activity, "failed_gates": failed, "passed": not failed})
    failed_ids = [row["line_id"] for row in results if not row["passed"]]
    return {"dialogue_channel": int(dialogue_channel), "source_channel": source_channel, "line_gate_results": results, "failed_line_ids": failed_ids, "failed_line_count": len(failed_ids), "all_lines_active": not failed_ids}

def build_candidate_matrix(lines: Sequence[Any], options_by_line: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows=[]
    for line in lines:
        for option in options_by_line.get(line.id, ()):
            candidate=option.get("candidate")
            rows.append({"line_id":line.id,"candidate_id":getattr(candidate,"candidate_id",None),"stages":{"generated":True,"raw_qa":option.get("raw_audit") is not None,"processed":option.get("processed_audit") is not None,"mounted":option.get("mounted_audit") is not None,"serialized":option.get("serialized_audit") is not None},"eligible":bool(option.get("eligible")),"blocker":option.get("error") or option.get("alignment_status"),"recommended_action":None if option.get("eligible") else "REVIEW_OR_CAUSAL_REPAIR"})
    return rows

__all__=["build_candidate_matrix", "audit_scene_windows"]
