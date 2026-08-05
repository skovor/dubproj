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
from typing import Any, Iterable, Mapping, Sequence

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
    label: str = ""
    labels: tuple[str, ...] = ()
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
        if not _text(self.clip_id) or not _text(self.reviewer_id):
            raise ValueError("clip_id and reviewer_id are required")
        selected = tuple(dict.fromkeys(str(item).strip() for item in self.labels if str(item).strip()))
        if self.label.strip() and self.label not in selected:
            selected = (self.label, *selected)
        if not selected:
            raise ValueError("at least one human label is required")
        unknown = [item for item in selected if item not in LABELS]
        if unknown:
            raise ValueError(f"unknown human label: {unknown[0]}")
        object.__setattr__(self, "labels", selected)
        object.__setattr__(self, "label", selected[0])
        if self.region_start is not None and self.region_end is not None and self.region_end < self.region_start:
            raise ValueError("label region is reversed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["affected_tokens"] = list(self.affected_tokens)
        result["labels"] = list(self.labels)
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
                clip_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, claimed_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL,
                PRIMARY KEY (clip_id, reviewer_id),
                FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            );
            CREATE TABLE IF NOT EXISTS adjudications (
                clip_id TEXT PRIMARY KEY, adjudicator_id TEXT NOT NULL,
                consensus_labels TEXT NOT NULL, comment TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            );
            CREATE TABLE IF NOT EXISTS hidden_seal (
                seal_id TEXT PRIMARY KEY, operator_id TEXT NOT NULL, digest TEXT NOT NULL,
                created_at TEXT NOT NULL, opened_at TEXT
            );
        """)
        self._migrate_claims()
        self._db.commit()

    def _migrate_claims(self) -> None:
        """Upgrade the pre-double-review queue without losing its audit trail."""
        columns = {str(row["name"]): int(row["pk"]) for row in self._db.execute("PRAGMA table_info(claims)")}
        if "lease_expires_at" in columns and columns.get("clip_id") != 0:
            return
        self._db.execute("ALTER TABLE claims RENAME TO claims_legacy")
        self._db.execute("""CREATE TABLE claims (
            clip_id TEXT NOT NULL, reviewer_id TEXT NOT NULL, claimed_at TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL, PRIMARY KEY (clip_id, reviewer_id),
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
        )""")
        from datetime import datetime, timezone
        expired = datetime.now(timezone.utc).isoformat()
        self._db.execute("INSERT INTO claims(clip_id, reviewer_id, claimed_at, lease_expires_at) SELECT clip_id, reviewer_id, claimed_at, ? FROM claims_legacy", (expired,))
        self._db.execute("DROP TABLE claims_legacy")

    def close(self) -> None:
        self._db.close()

    def add_clip(self, clip: ClipRecord) -> None:
        payload = canonical_json(clip.to_dict())
        try:
            self._db.execute("INSERT INTO clips VALUES (?, ?, ?, ?, ?)", (clip.clip_id, payload, clip.split_group, clip.split, clip.audio_sha256))
        except sqlite3.IntegrityError:
            existing = self._db.execute("SELECT payload FROM clips WHERE clip_id=?", (clip.clip_id,)).fetchone()
            if existing is None or str(existing["payload"]) != payload:
                raise ValueError(f"clip_id already exists with different immutable content: {clip.clip_id}")
        self._db.commit()

    def add_clips(self, clips: Iterable[ClipRecord]) -> None:
        for clip in clips:
            self.add_clip(clip)

    def claim(self, reviewer_id: str, *, split: str | None = None, lease_seconds: int = 900) -> ClipRecord | None:
        from datetime import datetime, timedelta, timezone
        reviewer_id = _text(reviewer_id)
        if not reviewer_id or lease_seconds <= 0:
            raise ValueError("reviewer_id and positive lease_seconds are required")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        expires_text = (now + timedelta(seconds=int(lease_seconds))).isoformat()
        self._db.execute("BEGIN IMMEDIATE")
        query = """SELECT c.* FROM clips c
            WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.clip_id=c.clip_id AND l.reviewer_id=?)
              AND NOT EXISTS (SELECT 1 FROM claims q WHERE q.clip_id=c.clip_id AND q.reviewer_id=? AND q.lease_expires_at>?)"""
        args: list[Any] = [reviewer_id, reviewer_id, now_text]
        if split:
            query += " AND c.split=?"; args.append(split)
        row = self._db.execute(query + " ORDER BY c.clip_id LIMIT 1", args).fetchone()
        if row is None:
            self._db.commit()
            return None
        self._db.execute("INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?)", (row["clip_id"], reviewer_id, now_text, expires_text)); self._db.commit()
        return ClipRecord(**json.loads(row["payload"]))

    def release_claim(self, clip_id: str, reviewer_id: str) -> None:
        self._db.execute("DELETE FROM claims WHERE clip_id=? AND reviewer_id=?", (clip_id, reviewer_id)); self._db.commit()

    def save_label(self, label: HumanLabel) -> None:
        if self._db.execute("SELECT 1 FROM clips WHERE clip_id=?", (label.clip_id,)).fetchone() is None:
            raise ValueError(f"unknown clip: {label.clip_id}")
        self._db.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)", (label.clip_id, label.reviewer_id, canonical_json(label.to_dict())))
        self._db.execute("DELETE FROM claims WHERE clip_id=? AND reviewer_id=?", (label.clip_id, label.reviewer_id))
        self._db.commit()

    def adjudicate(self, clip_id: str, adjudicator_id: str, consensus_labels: Sequence[str], *, comment: str = "") -> None:
        selected = tuple(dict.fromkeys(str(item).strip() for item in consensus_labels if str(item).strip()))
        if not selected or any(item not in LABELS for item in selected):
            raise ValueError("adjudication requires valid consensus labels")
        if self._db.execute("SELECT 1 FROM clips WHERE clip_id=?", (clip_id,)).fetchone() is None:
            raise ValueError(f"unknown clip: {clip_id}")
        from datetime import datetime, timezone
        self._db.execute("INSERT OR REPLACE INTO adjudications VALUES (?, ?, ?, ?, ?)", (clip_id, adjudicator_id, canonical_json(list(selected)), comment, datetime.now(timezone.utc).isoformat()))
        self._db.commit()

    def seal_hidden_test(self, operator_id: str) -> dict[str, Any]:
        """Seal hidden membership once; the seal is content-addressed."""
        from datetime import datetime, timezone
        operator_id = _text(operator_id)
        if not operator_id:
            raise ValueError("operator_id is required")
        if self._db.execute("SELECT 1 FROM hidden_seal LIMIT 1").fetchone() is not None:
            raise ValueError("hidden test is already sealed")
        rows = [{"clip_id": row["clip_id"], "payload": json.loads(row["payload"]), "audio_sha256": row["audio_sha256"]} for row in self._db.execute("SELECT clip_id, payload, audio_sha256 FROM clips WHERE split='hidden_test' ORDER BY clip_id")]
        if not rows:
            raise ValueError("cannot seal an empty hidden test")
        digest = sha256_bytes(canonical_json(rows))
        created_at = datetime.now(timezone.utc).isoformat()
        self._db.execute("INSERT INTO hidden_seal VALUES (?, ?, ?, ?, NULL)", (f"hidden-{digest[:16]}", operator_id, digest, created_at)); self._db.commit()
        return self.hidden_seal() or {}

    def hidden_seal(self) -> dict[str, Any] | None:
        row = self._db.execute("SELECT seal_id, operator_id, digest, created_at, opened_at FROM hidden_seal LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    def mark_hidden_opened(self, operator_id: str) -> dict[str, Any]:
        """Record the one-shot hidden-set access used for final evaluation."""
        seal = self.hidden_seal()
        if seal is None:
            raise ValueError("hidden test is not sealed")
        if seal.get("opened_at"):
            raise ValueError("hidden test has already been opened")
        from datetime import datetime, timezone
        self._db.execute("UPDATE hidden_seal SET opened_at=? WHERE seal_id=?", (datetime.now(timezone.utc).isoformat(), seal["seal_id"])); self._db.commit()
        return self.hidden_seal() or {}

    def clips(self) -> list[ClipRecord]:
        return [ClipRecord(**json.loads(row["payload"])) for row in self._db.execute("SELECT payload FROM clips ORDER BY clip_id")]

    def labels(self) -> list[HumanLabel]:
        result = []
        adjudicated = {row["clip_id"]: row["adjudicator_id"] for row in self._db.execute("SELECT clip_id, adjudicator_id FROM adjudications")}
        for row in self._db.execute("SELECT payload FROM labels ORDER BY clip_id, reviewer_id"):
            value = json.loads(row["payload"]); value["affected_tokens"] = tuple(value.get("affected_tokens") or []); value["labels"] = tuple(value.get("labels") or ([value.get("label")] if value.get("label") else []))
            if value.get("clip_id") in adjudicated: value["adjudicated_by"] = adjudicated[value["clip_id"]]
            result.append(HumanLabel(**value))
        return result

    def export(self, directory: str | Path, *, include_hidden: bool = False) -> dict[str, str]:
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        clips = self.clips(); labels = self.labels()
        hidden = [clip for clip in clips if clip.split == "hidden_test"]
        seal = self.hidden_seal()
        if hidden and seal is None:
            raise ValueError("hidden test must be sealed before export")
        if not include_hidden:
            labels = [label for label in labels if next((clip.split for clip in clips if clip.clip_id == label.clip_id), "") != "hidden_test"]
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
            if len({tuple(row.labels) for row in rows}) > 1 and not any(row.adjudicated_by for row in rows):
                disagreements.append({"clip_id": clip_id, "reviewers": [row.reviewer_id for row in rows], "labels": [list(row.labels) for row in rows]})
        paths["disagreements"].write_text("".join(canonical_json(row) + "\n" for row in disagreements), encoding="utf-8")
        if seal:
            seal_path = root / "hidden_seal.json"; atomic_json(seal_path, {**seal, "labels_exported": bool(include_hidden)}); paths["hidden_seal"] = seal_path
        return {key: str(value) for key, value in paths.items()}


def validate_goldset(clips: Iterable[ClipRecord], labels: Iterable[HumanLabel], *, require_double_review: bool = True, hidden_sealed: bool = False) -> dict[str, Any]:
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
        if len({tuple(row.labels) for row in rows}) > 1 and not any(row.adjudicated_by for row in rows): errors.append(f"unadjudicated disagreement: {clip_id}")
    counts = {split: sum(1 for clip in clips if clip.split == split) for split in SPLITS}
    if counts["hidden_test"] and not hidden_sealed:
        errors.append("hidden test membership is not sealed")
    return {"valid": not errors, "errors": errors, "clip_count": len(clips), "label_count": len(labels), "split_counts": counts, "hidden_test_sealed": bool(counts["hidden_test"] and hidden_sealed)}


def manifest_hash(path: str | Path) -> str:
    return sha256_file(path)


__all__ = ["LABELS", "SPLITS", "ClipRecord", "HumanLabel", "GoldsetStore", "stable_split", "validate_goldset", "manifest_hash"]
