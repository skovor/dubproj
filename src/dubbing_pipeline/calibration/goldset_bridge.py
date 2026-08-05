"""Reproducible bridge from human gold-set labels to calibration rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from ..goldset import ClipRecord, GoldsetStore, HumanLabel, validate_goldset
from ..hashing import canonical_json, sha256_bytes
from .features import FeatureRow, final_anchor_features, target_features
from .lid_features import LIDFeatureRow, features as lid_features

TARGET_BAD = {"LEXICAL_ERROR", "PRONUNCIATION_BAD", "SOURCE_LANGUAGE_LEAK", "UNDECIDABLE"}
FINAL_BAD = {"FINAL_ANCHOR_MISSING", "TIMING_BAD", "MOUNT_BAD", "UNDECIDABLE"}


def _labels_by_clip(labels: list[HumanLabel]) -> dict[str, list[HumanLabel]]:
    result: dict[str, list[HumanLabel]] = {}
    for label in labels:
        result.setdefault(label.clip_id, []).append(label)
    return result


def _label_for(labels: list[HumanLabel], bad: set[str]) -> int:
    selected = {item for row in labels for item in row.labels}
    if not labels or "UNDECIDABLE" in selected:
        raise ValueError("cannot create calibration target from undecidable/missing human labels")
    return 0 if selected & bad else 1


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def extract_goldset_features(
    store: GoldsetStore,
    evidence_provider: Callable[[ClipRecord], Mapping[str, Any]],
    output_dir: str | Path,
    *,
    require_hidden_seal: bool = True,
    hidden_evaluation_receipt: Mapping[str, Any] | None = None,
    hidden_operator_id: str | None = None,
    hidden_run_id: str | None = None,
) -> dict[str, Any]:
    """Materialize target/final/LID JSONL datasets from frozen evidence.

    ``evidence_provider`` is deliberately injected: production callers must
    run the actual ASR/CTC/LID stack and return its content-addressed output;
    this bridge never invents scores or converts a pipeline verdict to a label.
    """
    clips = store.clips(); reviewer_labels = store.labels(); labels = store.effective_labels(); seal = store.hidden_seal()
    validation = validate_goldset(clips, reviewer_labels, require_double_review=True, hidden_sealed=bool(seal))
    if not validation["valid"]:
        raise ValueError("gold set is not ready: " + "; ".join(validation["errors"]))
    hidden_present = any(clip.split == "hidden_test" for clip in clips)
    if require_hidden_seal and hidden_present:
        if seal is None or not store.verify_hidden_seal():
            raise ValueError("hidden test seal is missing or invalid")
        if hidden_evaluation_receipt is None and hidden_operator_id and hidden_run_id:
            hidden_evaluation_receipt = store.open_hidden_evaluation(hidden_operator_id, hidden_run_id)
        if hidden_evaluation_receipt is None or not store.verify_hidden_evaluation_receipt(hidden_evaluation_receipt):
            raise ValueError("hidden evaluation receipt is required and must be issued by the sealed store")
    by_clip = _labels_by_clip(labels)
    target_rows: list[dict[str, Any]] = []; final_rows: list[dict[str, Any]] = []; lid_rows: list[dict[str, Any]] = []
    for clip in clips:
        clip_labels = by_clip.get(clip.clip_id, [])
        evidence = dict(evidence_provider(clip))
        if not evidence or not isinstance(evidence.get("target", evidence), Mapping):
            raise ValueError(f"missing frozen evidence for {clip.clip_id}")
        target_evidence = evidence.get("target", evidence)
        final_evidence = evidence.get("final", target_evidence)
        lid_evidence = evidence.get("lid")
        target = target_features(target_evidence, performance_mode=clip.performance_mode)
        final = final_anchor_features(final_evidence)
        base_meta = {"audio_sha256": clip.audio_sha256, "clip_id": clip.clip_id, "label_hash": sha256_bytes(canonical_json([row.to_dict() for row in clip_labels])), "evidence_hash": sha256_bytes(canonical_json(evidence)), "source": "human_goldset", "label_authority": "adjudicated_consensus" if any(row.adjudicated_by for row in clip_labels) else "double_review"}
        target_rows.append(FeatureRow(clip.clip_id, clip.split, clip.split_group, _label_for(clip_labels, TARGET_BAD), target, clip.performance_mode, base_meta).to_dict())
        final_rows.append(FeatureRow(clip.clip_id, clip.split, clip.split_group, _label_for(clip_labels, FINAL_BAD), final, clip.performance_mode, base_meta).to_dict())
        if lid_evidence is None:
            raise ValueError(f"missing independent LID evidence for {clip.clip_id}")
        lid = lid_features(lid_evidence, performance_mode=clip.performance_mode)
        lid_rows.append(LIDFeatureRow(clip.clip_id, clip.split, clip.split_group, 1 if "SOURCE_LANGUAGE_LEAK" in {item for row in clip_labels for item in row.labels} else 0, lid, clip.performance_mode, base_meta).to_dict())
    root = Path(output_dir)
    rows_by_role = {"target": target_rows, "final_anchor": final_rows, "lid": lid_rows}
    # Keep a complete diagnostic file, but make every training/evaluation
    # split independently addressable.  Callers that train a calibrator can
    # therefore pass only ``*_calibration.jsonl`` and cannot accidentally
    # include validation or hidden rows through a glob.
    paths = {"target": root / "target_features.jsonl", "final_anchor": root / "final_anchor_features.jsonl", "lid": root / "lid_features.jsonl"}
    digests = {role: _write_jsonl(paths[role], rows) for role, rows in rows_by_role.items()}
    paths_by_split: dict[str, dict[str, str]] = {}
    sha_by_split: dict[str, dict[str, str]] = {}
    for role, rows in rows_by_role.items():
        for split in ("calibration", "validation", "hidden_test"):
            path = root / f"{role}_{split}.jsonl"
            split_rows = [row for row in rows if row.get("split") == split]
            paths_by_split.setdefault(role, {})[split] = str(path)
            sha_by_split.setdefault(role, {})[split] = _write_jsonl(path, split_rows)
    return {"schema": "goldset-feature-bridge-v2", "paths": {key: str(path) for key, path in paths.items()}, "paths_by_split": paths_by_split, "sha256": digests, "sha256_by_split": sha_by_split, "counts": {"target": len(target_rows), "final_anchor": len(final_rows), "lid": len(lid_rows)}, "counts_by_split": {role: {split: sum(1 for row in rows if row.get("split") == split) for split in ("calibration", "validation", "hidden_test")} for role, rows in rows_by_role.items()}, "hidden_seal": seal, "hidden_evaluation_receipt": hidden_evaluation_receipt}


__all__ = ["extract_goldset_features"]
