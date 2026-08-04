"""Auditable human gold-set contracts.

This module only creates review work and validates human labels.  It never
infers a label from ASR, CTC, LID, or a pipeline verdict.  Automatic evidence
may live beside a clip for debugging, but is intentionally omitted from the
review payload.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import atomic_json, canonical_json, sha256_bytes, sha256_file

LABELS = (
    "CORRECT_NEUTRAL", "CORRECT_EXPRESSIVE", "LEXICAL_ERROR",
    "FINAL_ANCHOR_MISSING", "SOURCE_LANGUAGE_LEAK", "PRONUNCIATION_BAD",
    "TIMING_BAD", "MOUNT_BAD", "UNDECIDABLE",
)
SPLITS = ("calibration", "validation", "hidden_test")
_REVIEW_FIELDS = {"clip_id", "scene_id", "line_id", "candidate_id", "speaker_id", "expected_text", "source_text", "performance_mode", "audio_path", "context_path", "source_reference_path"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def stable_split(split_group: str, *, seed: str = "goldset-v1") -> str:
    """Assign a group deterministically; one source line cannot straddle splits."""
    digest = hashlib.sha256(f"{seed}:{split_group}".encode("utf-8")).digest()[0] % 100
    return "calibration" if digest < 60 else ("validation" if digest < 80 else "hidden_test")


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    audio_sha256: str
    scene_id: str
    line_id: str
    candidate_id: str
    speaker_id: str
    expected_text: str
    source_text: str = ""
    performance_mode: str = "NEUTRAL"
    audio_path: str = ""
    context_path: str | None = None
    source_reference_path: str | None = None
    generation_provenance: dict[str, Any] = field(default_factory=dict)
    split_group: str = ""
    split: str = ""

    def __post_init__(self) -> None:
        for name in ("clip_id", "audio_sha256", "scene_id", "line_id", "candidate_id", "expected_text"):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        if len(self.audio_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.audio_sha256):
            raise ValueError("audio_sha256 must be a SHA-256 hex digest")
        group = self.split_group or f"{self.scene_id}:{self.line_id}"
        split = self.split or stable_split(group)
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        object.__setattr__(self, "split_group", group)
        object.__setattr__(self, "split", split)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def review_payload(self) -> dict[str, Any]:
        """Safe A/B view: no automatic scores, transcripts, or decisions."""
        return {key: self.to_dict().get(key) for key in _REVIEW_FIELDS}


@dataclass(frozen=True)
class HumanLabel:
    clip_id: str
    reviewer_id: str
    label: str
    severity: str = "unknown"
    region_start: float | None = None
    region_end: float | None = None
    affected_tokens: tuple[str, ...] = ()
    comment: str = ""
    confidence: str = "unknown"
    needs_context: bool = False
    adjudicated_by: str | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown human label: {self.label}")
        if not _text(self.clip_id) or not _text(self.reviewer_id):
            raise ValueError("clip_id and reviewer_id are required")
        if self.region_start is not None and self.region_end is not None and self.region_end < self.region_start:
            raise ValueError("label region is reversed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["affected_tokens"] = list(self.affected_tokens)
        return result


class GoldsetStore:
    """SQLite-backed queue with independent reviewer rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY, payload TEXT NOT NULL, split_group TEXT NOT NULL,
                split TEXT NOT NULL, audio_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS labels (
                clip_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (clip_id, reviewer_id), FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            );
            CREATE TABLE IF NOT EXISTS claims (
                clip_id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, claimed_at TEXT NOT NULL
            );
        """)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def add_clip(self, clip: ClipRecord) -> None:
        self._db.execute("INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?)", (clip.clip_id, canonical_json(clip.to_dict()), clip.split_group, clip.split, clip.audio_sha256))
        self._db.commit()

    def add_clips(self, clips: Iterable[ClipRecord]) -> None:
        for clip in clips:
            self.add_clip(clip)

    def claim(self, reviewer_id: str, *, split: str | None = None) -> ClipRecord | None:
        query = "SELECT c.* FROM clips c LEFT JOIN claims q ON q.clip_id=c.clip_id WHERE q.clip_id IS NULL"
        args: list[Any] = []
        if split:
            query += " AND c.split=?"; args.append(split)
        row = self._db.execute(query + " ORDER BY c.clip_id LIMIT 1", args).fetchone()
        if row is None:
            return None
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute("INSERT OR IGNORE INTO claims VALUES (?, ?, ?)", (row["clip_id"], reviewer_id, now)); self._db.commit()
        return ClipRecord(**json.loads(row["payload"]))

    def save_label(self, label: HumanLabel) -> None:
        if self._db.execute("SELECT 1 FROM clips WHERE clip_id=?", (label.clip_id,)).fetchone() is None:
            raise ValueError(f"unknown clip: {label.clip_id}")
        self._db.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)", (label.clip_id, label.reviewer_id, canonical_json(label.to_dict())))
        self._db.commit()

    def clips(self) -> list[ClipRecord]:
        return [ClipRecord(**json.loads(row["payload"])) for row in self._db.execute("SELECT payload FROM clips ORDER BY clip_id")]

    def labels(self) -> list[HumanLabel]:
        result = []
        for row in self._db.execute("SELECT payload FROM labels ORDER BY clip_id, reviewer_id"):
            value = json.loads(row["payload"]); value["affected_tokens"] = tuple(value.get("affected_tokens") or [])
            result.append(HumanLabel(**value))
        return result

    def export(self, directory: str | Path) -> dict[str, str]:
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        clips = self.clips(); labels = self.labels()
        paths = {"manifest": root / "manifest.jsonl", "labels": root / "labels.jsonl", "splits": root / "splits.json", "reviewers": root / "reviewers.json", "disagreements": root / "disagreements.jsonl"}
        paths["manifest"].write_text("".join(canonical_json(c.to_dict()) + "\n" for c in clips), encoding="utf-8")
        paths["labels"].write_text("".join(canonical_json(l.to_dict()) + "\n" for l in labels), encoding="utf-8")
        split_map = {c.clip_id: {"split": c.split, "split_group": c.split_group} for c in clips}
        atomic_json(paths["splits"], split_map)
        reviewers = sorted({l.reviewer_id for l in labels}); atomic_json(paths["reviewers"], {"reviewers": reviewers})
        by_clip: dict[str, list[HumanLabel]] = {}
        for label in labels: by_clip.setdefault(label.clip_id, []).append(label)
        disagreements = []
        for clip_id, rows in by_clip.items():
            if len({row.label for row in rows}) > 1 and not any(row.adjudicated_by for row in rows):
                disagreements.append({"clip_id": clip_id, "reviewers": [row.reviewer_id for row in rows], "labels": [row.label for row in rows]})
        paths["disagreements"].write_text("".join(canonical_json(row) + "\n" for row in disagreements), encoding="utf-8")
        return {key: str(value) for key, value in paths.items()}


def validate_goldset(clips: Iterable[ClipRecord], labels: Iterable[HumanLabel], *, require_double_review: bool = True) -> dict[str, Any]:
    clips = list(clips); labels = list(labels)
    errors: list[str] = []; ids = [c.clip_id for c in clips]
    if len(ids) != len(set(ids)): errors.append("duplicate clip_id")
    sha_splits: dict[str, set[str]] = {}; line_splits: dict[str, set[str]] = {}
    for clip in clips:
        sha_splits.setdefault(clip.audio_sha256, set()).add(clip.split)
        line_splits.setdefault(f"{clip.scene_id}:{clip.line_id}", set()).add(clip.split)
        if clip.split not in SPLITS: errors.append(f"invalid split: {clip.clip_id}")
        if clip.audio_path and not Path(clip.audio_path).is_file(): errors.append(f"missing audio: {clip.clip_id}")
    errors.extend(f"audio SHA crosses splits: {key}" for key, values in sha_splits.items() if len(values) > 1)
    errors.extend(f"line crosses splits: {key}" for key, values in line_splits.items() if len(values) > 1)
    labels_by_clip: dict[str, list[HumanLabel]] = {}
    for label in labels:
        if label.clip_id not in set(ids): errors.append(f"label references unknown clip: {label.clip_id}")
        labels_by_clip.setdefault(label.clip_id, []).append(label)
    if require_double_review:
        errors.extend(f"missing independent reviews: {clip_id}" for clip_id in ids if len({row.reviewer_id for row in labels_by_clip.get(clip_id, [])}) < 2)
    for clip_id, rows in labels_by_clip.items():
        if len({row.label for row in rows}) > 1 and not any(row.adjudicated_by for row in rows): errors.append(f"unadjudicated disagreement: {clip_id}")
    counts = {split: sum(1 for clip in clips if clip.split == split) for split in SPLITS}
    return {"valid": not errors, "errors": errors, "clip_count": len(clips), "label_count": len(labels), "split_counts": counts, "hidden_test_sealed": counts["hidden_test"] > 0}


def manifest_hash(path: str | Path) -> str:
    return sha256_file(path)


__all__ = ["LABELS", "SPLITS", "ClipRecord", "HumanLabel", "GoldsetStore", "stable_split", "validate_goldset", "manifest_hash"]
