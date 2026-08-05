"""Reproducible bridge from human gold-set labels to calibration rows."""
from __future__ import annotations

import hashlib
import json
import os, subprocess, uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ..goldset import ClipRecord, GoldsetStore, HumanLabel, TARGET_BAD_LABELS, FINAL_ANCHOR_BAD_LABELS, LID_BAD_LABELS, validate_goldset
from ..hashing import canonical_json, sha256_bytes
from .features import FeatureRow, final_anchor_features, target_features
from .lid_features import LIDFeatureRow, features as lid_features

TARGET_BAD = set(TARGET_BAD_LABELS)
FINAL_BAD = set(FINAL_ANCHOR_BAD_LABELS)


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
    extractor_id: str | None = None,
    extractor_version: str | None = None,
    code_commit: str | None = None,
    runtime_lock_sha256: str = "",
    models_lock_sha256: str = "",
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
    bridge_run_id = str((hidden_evaluation_receipt or {}).get("run_id") or f"bridge-{uuid.uuid4().hex}")
    if not code_commit:
        try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: code_commit = "unknown"
    extractor_id = extractor_id or f"{getattr(evidence_provider, '__module__', 'provider')}:{getattr(evidence_provider, '__qualname__', 'callable')}"
    extractor_version = extractor_version or str(getattr(evidence_provider, "__version__", "unknown"))
    by_clip = _labels_by_clip(labels)
    target_rows: list[dict[str, Any]] = []; final_rows: list[dict[str, Any]] = []; lid_rows: list[dict[str, Any]] = []
    bridge_receipts: list[dict[str, Any]] = []
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
        authoritative = store.authoritative_hidden_label_evidence(clip.clip_id)
        evidence_sha = sha256_bytes(canonical_json(evidence))
        base_meta = {"audio_sha256": clip.audio_sha256, "clip_id": clip.clip_id, "label_payload_sha256": authoritative["label_payload_sha256"], "evidence_sha256": evidence_sha, "label_hash": authoritative["label_payload_sha256"], "evidence_hash": evidence_sha, "source": "human_goldset", "label_authority": "adjudicated_consensus" if any(row.adjudicated_by for row in clip_labels) else "double_review", "extractor_id": extractor_id, "extractor_version": extractor_version, "code_commit": code_commit, "runtime_lock_sha256": runtime_lock_sha256, "models_lock_sha256": models_lock_sha256}
        role_values = (("target", target, TARGET_BAD_LABELS, authoritative["target_binary_label"], "char-alignment-v3"), ("final_anchor", final, FINAL_ANCHOR_BAD_LABELS, authoritative["final_anchor_binary_label"], "final-anchor-v1"))
        for role, features_value, _policy, binary, schema in role_values:
            receipt_payload = {"schema": "goldset-feature-bridge-receipt-v1", "clip_id": clip.clip_id, "role": role, "audio_sha256": clip.audio_sha256, "label_payload_sha256": authoritative["label_payload_sha256"], "binary_label": int(binary), "features": dict(features_value), "feature_schema_version": schema, "evidence": evidence, "evidence_sha256": evidence_sha, "extractor_id": extractor_id, "extractor_version": extractor_version, "code_commit": code_commit, "runtime_lock_sha256": runtime_lock_sha256, "models_lock_sha256": models_lock_sha256}
            bridge_receipts.append({**receipt_payload, "receipt_sha256": sha256_bytes(canonical_json(receipt_payload))})
        target_rows.append(FeatureRow(clip.clip_id, clip.split, clip.split_group, _label_for(clip_labels, TARGET_BAD), target, clip.performance_mode, base_meta).to_dict())
        final_rows.append(FeatureRow(clip.clip_id, clip.split, clip.split_group, _label_for(clip_labels, FINAL_BAD), final, clip.performance_mode, base_meta).to_dict())
        if lid_evidence is None:
            raise ValueError(f"missing independent LID evidence for {clip.clip_id}")
        lid = lid_features(lid_evidence, performance_mode=clip.performance_mode)
        lid_rows.append(LIDFeatureRow(clip.clip_id, clip.split, clip.split_group, authoritative["lid_binary_label"], lid, clip.performance_mode, base_meta).to_dict())
        lid_receipt_payload = {"schema": "goldset-feature-bridge-receipt-v1", "clip_id": clip.clip_id, "role": "lid", "audio_sha256": clip.audio_sha256, "label_payload_sha256": authoritative["label_payload_sha256"], "binary_label": int(authoritative["lid_binary_label"]), "features": dict(lid), "feature_schema_version": "lid-fusion-v3", "evidence": evidence, "evidence_sha256": evidence_sha, "extractor_id": extractor_id, "extractor_version": extractor_version, "code_commit": code_commit, "runtime_lock_sha256": runtime_lock_sha256, "models_lock_sha256": models_lock_sha256}
        bridge_receipts.append({**lid_receipt_payload, "receipt_sha256": sha256_bytes(canonical_json(lid_receipt_payload))})
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
    bridge_digests = store.record_bridge_receipts(bridge_run_id, bridge_receipts)
    return {"schema": "goldset-feature-bridge-v2", "paths": {key: str(path) for key, path in paths.items()}, "paths_by_split": paths_by_split, "sha256": digests, "sha256_by_split": sha_by_split, "counts": {"target": len(target_rows), "final_anchor": len(final_rows), "lid": len(lid_rows)}, "counts_by_split": {role: {split: sum(1 for row in rows if row.get("split") == split) for split in ("calibration", "validation", "hidden_test")} for role, rows in rows_by_role.items()}, "hidden_seal": seal, "hidden_evaluation_receipt": hidden_evaluation_receipt, "bridge_run_id": bridge_run_id, "bridge_receipts": bridge_digests}


__all__ = ["extract_goldset_features"]
