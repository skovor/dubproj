"""Auditable human gold-set contracts.

This module only creates review work and validates human labels.  It never
infers a label from ASR, CTC, LID, or a pipeline verdict.  Automatic evidence
may live beside a clip for debugging, but is intentionally omitted from the
review payload.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from functools import wraps
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
TARGET_BAD_LABELS = frozenset({"LEXICAL_ERROR", "PRONUNCIATION_BAD", "SOURCE_LANGUAGE_LEAK", "UNDECIDABLE"})
FINAL_ANCHOR_BAD_LABELS = frozenset({"FINAL_ANCHOR_MISSING", "TIMING_BAD", "MOUNT_BAD", "UNDECIDABLE"})
LID_BAD_LABELS = frozenset({"SOURCE_LANGUAGE_LEAK"})
_REVIEW_FIELDS = {"clip_id", "scene_id", "line_id", "candidate_id", "speaker_id", "expected_text", "source_text", "performance_mode", "audio_path", "context_path", "source_reference_path"}


def _db_locked(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return guarded


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
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
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
            CREATE TABLE IF NOT EXISTS hidden_evaluations (
                receipt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL, receipt_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hidden_finalizations (
                finalization_id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE, profile_id TEXT NOT NULL,
                code_commit TEXT NOT NULL, payload TEXT NOT NULL,
                finalization_sha256 TEXT NOT NULL, consumed_at TEXT,
                consumed_by_profile_id TEXT, consumed_by_code_commit TEXT,
                FOREIGN KEY (receipt_id) REFERENCES hidden_evaluations(receipt_id)
            );
            CREATE TABLE IF NOT EXISTS bridge_receipts (
                receipt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                clip_id TEXT NOT NULL, role TEXT NOT NULL, payload TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                UNIQUE(run_id, clip_id, role),
                FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
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

    @_db_locked
    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False

    @_db_locked
    def add_clip(self, clip: ClipRecord) -> None:
        if clip.split == "hidden_test" and self.hidden_seal() is not None:
            raise ValueError("hidden test membership is sealed")
        payload = canonical_json(clip.to_dict())
        try:
            self._db.execute("INSERT INTO clips VALUES (?, ?, ?, ?, ?)", (clip.clip_id, payload, clip.split_group, clip.split, clip.audio_sha256))
        except sqlite3.IntegrityError:
            existing = self._db.execute("SELECT payload FROM clips WHERE clip_id=?", (clip.clip_id,)).fetchone()
            if existing is None or str(existing["payload"]) != payload:
                raise ValueError(f"clip_id already exists with different immutable content: {clip.clip_id}")
        self._db.commit()

    @_db_locked
    def add_clips(self, clips: Iterable[ClipRecord]) -> None:
        for clip in clips:
            self.add_clip(clip)

    @_db_locked
    def claim(self, reviewer_id: str, *, split: str | None = None, lease_seconds: int = 900, allow_hidden: bool = False) -> ClipRecord | None:
        from datetime import datetime, timedelta, timezone
        reviewer_id = _text(reviewer_id)
        if not reviewer_id or lease_seconds <= 0:
            raise ValueError("reviewer_id and positive lease_seconds are required")
        if split == "hidden_test" and not allow_hidden:
            raise PermissionError("hidden_test claims require the isolated hidden evaluation flow")
        now = datetime.now(timezone.utc)
        now_text = now.isoformat()
        expires_text = (now + timedelta(seconds=int(lease_seconds))).isoformat()
        self._db.execute("BEGIN IMMEDIATE")
        query = """SELECT c.* FROM clips c
            WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.clip_id=c.clip_id AND l.reviewer_id=?)
              AND NOT EXISTS (SELECT 1 FROM claims q WHERE q.clip_id=c.clip_id AND q.reviewer_id=? AND q.lease_expires_at>?)"""
        args: list[Any] = [reviewer_id, reviewer_id, now_text]
        if not allow_hidden:
            query += " AND c.split != 'hidden_test'"
        if split:
            query += " AND c.split=?"; args.append(split)
        row = self._db.execute(query + " ORDER BY c.clip_id LIMIT 1", args).fetchone()
        if row is None:
            self._db.commit()
            return None
        self._db.execute("INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?)", (row["clip_id"], reviewer_id, now_text, expires_text)); self._db.commit()
        return ClipRecord(**json.loads(row["payload"]))

    @_db_locked
    def release_claim(self, clip_id: str, reviewer_id: str) -> None:
        self._db.execute("DELETE FROM claims WHERE clip_id=? AND reviewer_id=?", (clip_id, reviewer_id)); self._db.commit()

    @_db_locked
    def save_label(self, label: HumanLabel) -> None:
        clip_row = self._db.execute("SELECT split FROM clips WHERE clip_id=?", (label.clip_id,)).fetchone()
        if clip_row is None:
            raise ValueError(f"unknown clip: {label.clip_id}")
        if clip_row["split"] == "hidden_test" and self.hidden_seal() is not None:
            raise ValueError("hidden test labels are sealed")
        self._db.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)", (label.clip_id, label.reviewer_id, canonical_json(label.to_dict())))
        self._db.execute("DELETE FROM claims WHERE clip_id=? AND reviewer_id=?", (label.clip_id, label.reviewer_id))
        self._db.commit()

    @_db_locked
    def adjudicate(self, clip_id: str, adjudicator_id: str, consensus_labels: Sequence[str], *, comment: str = "") -> None:
        selected = tuple(dict.fromkeys(str(item).strip() for item in consensus_labels if str(item).strip()))
        if not selected or any(item not in LABELS for item in selected):
            raise ValueError("adjudication requires valid consensus labels")
        clip_row = self._db.execute("SELECT split FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if clip_row is None:
            raise ValueError(f"unknown clip: {clip_id}")
        if clip_row["split"] == "hidden_test" and self.hidden_seal() is not None:
            raise ValueError("hidden test adjudication is sealed")
        from datetime import datetime, timezone
        self._db.execute("INSERT OR REPLACE INTO adjudications VALUES (?, ?, ?, ?, ?)", (clip_id, adjudicator_id, canonical_json(list(selected)), comment, datetime.now(timezone.utc).isoformat()))
        self._db.commit()

    @_db_locked
    def seal_hidden_test(self, operator_id: str) -> dict[str, Any]:
        """Seal hidden membership once; the seal is content-addressed."""
        from datetime import datetime, timezone
        operator_id = _text(operator_id)
        if not operator_id:
            raise ValueError("operator_id is required")
        if self._db.execute("SELECT 1 FROM hidden_seal LIMIT 1").fetchone() is not None:
            raise ValueError("hidden test is already sealed")
        rows = self._hidden_seal_payload()
        if not rows:
            raise ValueError("cannot seal an empty hidden test")
        digest = sha256_bytes(canonical_json(rows))
        created_at = datetime.now(timezone.utc).isoformat()
        self._db.execute("INSERT INTO hidden_seal VALUES (?, ?, ?, ?, NULL)", (f"hidden-{digest[:16]}", operator_id, digest, created_at)); self._db.commit()
        return self.hidden_seal() or {}

    @_db_locked
    def hidden_seal(self) -> dict[str, Any] | None:
        row = self._db.execute("SELECT seal_id, operator_id, digest, created_at, opened_at FROM hidden_seal LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    @_db_locked
    def mark_hidden_opened(self, operator_id: str) -> dict[str, Any]:
        """Record the one-shot hidden-set access used for final evaluation."""
        seal = self.hidden_seal()
        if seal is None:
            raise ValueError("hidden test is not sealed")
        if seal.get("opened_at"):
            raise ValueError("hidden test has already been opened")
        if _text(operator_id) != str(seal.get("operator_id", "")):
            raise ValueError("only the sealing operator may open the hidden test")
        from datetime import datetime, timezone
        self._db.execute("UPDATE hidden_seal SET opened_at=? WHERE seal_id=?", (datetime.now(timezone.utc).isoformat(), seal["seal_id"])); self._db.commit()
        return self.hidden_seal() or {}

    @_db_locked
    def open_hidden_evaluation(self, operator_id: str, run_id: str) -> dict[str, Any]:
        """Atomically consume the sealed hidden set and issue one receipt."""
        from datetime import datetime, timezone
        operator_id = _text(operator_id); run_id = _text(run_id)
        if not operator_id or not run_id:
            raise ValueError("operator_id and run_id are required")
        self._db.execute("BEGIN IMMEDIATE")
        seal_row = self._db.execute("SELECT seal_id, operator_id, digest, created_at, opened_at FROM hidden_seal LIMIT 1").fetchone()
        if seal_row is None:
            self._db.rollback(); raise ValueError("hidden test is not sealed")
        seal = dict(seal_row)
        if seal.get("opened_at"):
            self._db.rollback(); raise ValueError("hidden test has already been opened")
        if operator_id != str(seal.get("operator_id", "")):
            self._db.rollback(); raise PermissionError("only the sealing operator may open the hidden test")
        if sha256_bytes(canonical_json(self._hidden_seal_payload())) != str(seal["digest"]).casefold():
            self._db.rollback(); raise ValueError("hidden seal digest mismatch")
        opened_at = datetime.now(timezone.utc).isoformat()
        rows = [dict(row) for row in self._db.execute("SELECT clip_id, audio_sha256, split, split_group FROM clips WHERE split='hidden_test' ORDER BY clip_id")]
        payload = {"schema": "hidden-evaluation-receipt-v1", "run_id": run_id, "operator_id": operator_id, "seal_id": seal["seal_id"], "seal_digest": seal["digest"], "opened_at": opened_at, "clips": rows}
        receipt_sha = sha256_bytes(canonical_json(payload))
        receipt = {**payload, "receipt_id": f"hidden-eval-{receipt_sha[:16]}", "receipt_sha256": receipt_sha}
        self._db.execute("INSERT INTO hidden_evaluations VALUES (?, ?, ?, ?)", (receipt["receipt_id"], run_id, canonical_json(receipt), receipt_sha))
        self._db.execute("UPDATE hidden_seal SET opened_at=? WHERE seal_id=?", (opened_at, seal["seal_id"]))
        self._db.commit()
        return receipt

    @_db_locked
    def verify_hidden_evaluation_receipt(self, receipt: Mapping[str, Any]) -> bool:
        if not verify_hidden_evaluation_receipt_payload(receipt):
            return False
        seal = self.hidden_seal()
        if seal is None or str(receipt.get("seal_id")) != str(seal.get("seal_id")) or str(receipt.get("seal_digest")) != str(seal.get("digest")):
            return False
        if not seal.get("opened_at") or str(receipt.get("opened_at")) != str(seal.get("opened_at")):
            return False
        row = self._db.execute("SELECT payload FROM hidden_evaluations WHERE receipt_id=? AND run_id=?", (receipt.get("receipt_id"), receipt.get("run_id"))).fetchone()
        if row is None or json.loads(row["payload"]) != dict(receipt):
            return False
        actual = [dict(item) for item in self._db.execute("SELECT clip_id, audio_sha256, split, split_group FROM clips WHERE split='hidden_test' ORDER BY clip_id")]
        return actual == list(receipt.get("clips") or []) and self.verify_hidden_seal()

    @_db_locked
    def authoritative_hidden_label_evidence(self, clip_id: str) -> dict[str, Any]:
        """Rebuild immutable calibration labels from the SQLite human record."""
        clip_row = self._db.execute("SELECT split, audio_sha256 FROM clips WHERE clip_id=?", (_text(clip_id),)).fetchone()
        if clip_row is None:
            raise ValueError(f"unknown clip: {clip_id}")
        labels = [label for label in self.effective_labels() if label.clip_id == _text(clip_id)]
        if not labels:
            raise ValueError(f"clip has no effective human label: {clip_id}")
        canonical_labels = sorted((label.to_dict() for label in labels), key=lambda value: (str(value.get("reviewer_id", "")), tuple(value.get("labels", []))))
        selected = {item for label in labels for item in label.labels}
        if "UNDECIDABLE" in selected:
            raise ValueError(f"clip has undecidable human label: {clip_id}")
        return {
            "clip_id": _text(clip_id), "audio_sha256": str(clip_row["audio_sha256"]),
            "split": str(clip_row["split"]), "canonical_labels": canonical_labels,
            "label_payload_sha256": sha256_bytes(canonical_json(canonical_labels)),
            "target_binary_label": 0 if selected & TARGET_BAD_LABELS else 1,
            "final_anchor_binary_label": 0 if selected & FINAL_ANCHOR_BAD_LABELS else 1,
            "lid_binary_label": 0 if selected & LID_BAD_LABELS else 1,
        }

    @_db_locked
    def record_bridge_receipts(self, run_id: str, receipts: Iterable[Mapping[str, Any]]) -> dict[str, str]:
        run_id = _text(run_id)
        if not run_id:
            raise ValueError("bridge run_id is required")
        digests: dict[str, str] = {}
        for value in receipts:
            receipt = dict(value)
            if receipt.get("schema") != "goldset-feature-bridge-receipt-v1":
                raise ValueError("invalid bridge receipt schema")
            receipt_sha = str(receipt.get("receipt_sha256", "")).casefold()
            payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
            if receipt_sha != sha256_bytes(canonical_json(payload)):
                raise ValueError("bridge receipt digest mismatch")
            clip_id, role = _text(receipt.get("clip_id")), _text(receipt.get("role"))
            if not clip_id or role not in {"target", "final_anchor", "lid"}:
                raise ValueError("bridge receipt clip and role are required")
            receipt_id = f"bridge-{receipt_sha[:16]}"
            try:
                self._db.execute("INSERT INTO bridge_receipts VALUES (?, ?, ?, ?, ?, ?)", (receipt_id, run_id, clip_id, role, canonical_json(receipt), receipt_sha))
            except sqlite3.IntegrityError:
                existing = self._db.execute("SELECT payload FROM bridge_receipts WHERE run_id=? AND clip_id=? AND role=?", (run_id, clip_id, role)).fetchone()
                if existing is None or json.loads(existing["payload"]) != receipt:
                    raise ValueError("bridge receipt already exists with different content")
            digests[f"{clip_id}:{role}"] = receipt_sha
        self._db.commit()
        return digests

    @_db_locked
    def get_bridge_receipt(self, run_id: str, clip_id: str, role: str) -> dict[str, Any]:
        row = self._db.execute("SELECT payload FROM bridge_receipts WHERE run_id=? AND clip_id=? AND role=?", (_text(run_id), _text(clip_id), _text(role))).fetchone()
        if row is None:
            raise ValueError("authoritative bridge receipt is missing")
        receipt = json.loads(row["payload"])
        payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != sha256_bytes(canonical_json(payload)):
            raise ValueError("authoritative bridge receipt is corrupted")
        return receipt

    @_db_locked
    def get_hidden_evaluation_receipt(self, receipt_id: str, run_id: str | None = None) -> dict[str, Any]:
        """Return the receipt stored in SQLite, never an operator supplied JSON claim."""
        query = "SELECT payload FROM hidden_evaluations WHERE receipt_id=?"
        args: list[Any] = [_text(receipt_id)]
        if run_id is not None:
            query += " AND run_id=?"; args.append(_text(run_id))
        row = self._db.execute(query, args).fetchone()
        if row is None:
            raise ValueError("hidden evaluation receipt is not present in the authoritative store")
        receipt = json.loads(row["payload"])
        if not self.verify_hidden_evaluation_receipt(receipt):
            raise ValueError("authoritative hidden evaluation receipt failed verification")
        return receipt

    @_db_locked
    def finalize_hidden_evaluation(
        self, *, receipt_id: str, run_id: str, profile_id: str, code_commit: str,
        role_hidden_rows: Mapping[str, Iterable[Any]],
        role_hidden_reports: Mapping[str, Mapping[str, Any]],
        hidden_jsonl_hashes: Mapping[str, str],
        hidden_report_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        """Finalize all hidden roles against the sealed DB before promotion.

        This is deliberately a database operation: a JSON receipt with a valid
        self-hash is not evidence.  Every role must contain exactly the sealed
        clip set, the immutable audio digest and frozen human/evidence hashes.
        """
        receipt = self.get_hidden_evaluation_receipt(receipt_id, run_id)
        hidden_receipt = receipt
        roles = ("target", "final_anchor", "lid")
        if set(role_hidden_rows) != set(roles) or set(role_hidden_reports) != set(roles):
            raise ValueError("all three independent hidden roles are required")
        sealed = {str(item["clip_id"]): dict(item) for item in receipt.get("clips", [])}
        if not sealed:
            raise ValueError("hidden evaluation receipt has no clips")

        def row_dict(value: Any) -> dict[str, Any]:
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            if not isinstance(value, Mapping):
                raise ValueError("hidden evidence row is not a mapping")
            return dict(value)

        rows_by_role: dict[str, list[dict[str, Any]]] = {}
        row_digests: dict[str, str] = {}
        role_labels: dict[str, dict[str, Any]] = {}
        bridge_digests: dict[str, str] = {}
        expected_binary = {"target": "target_binary_label", "final_anchor": "final_anchor_binary_label", "lid": "lid_binary_label"}
        expected_schema = {"target": "char-alignment-v3", "final_anchor": "final-anchor-v1", "lid": "lid-fusion-v3"}
        for role in roles:
            rows = [row_dict(row) for row in role_hidden_rows[role]]
            ids = [str(row.get("clip_id", "")) for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(sealed):
                raise ValueError(f"{role} hidden rows do not exactly match the sealed clip set")
            for row in rows:
                clip_id = str(row["clip_id"]); metadata = row.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise ValueError(f"{role}/{clip_id} is missing immutable metadata")
                if str(metadata.get("audio_sha256", "")).casefold() != str(sealed[clip_id]["audio_sha256"]).casefold():
                    raise ValueError(f"{role}/{clip_id} audio hash is not the sealed hash")
                authoritative = self.authoritative_hidden_label_evidence(clip_id)
                receipt = self.get_bridge_receipt(run_id, clip_id, role)
                if str(receipt.get("role")) != role or str(receipt.get("clip_id")) != clip_id:
                    raise ValueError(f"{role}/{clip_id} bridge receipt identity mismatch")
                if str(receipt.get("audio_sha256", "")).casefold() != str(authoritative["audio_sha256"]).casefold():
                    raise ValueError(f"{role}/{clip_id} bridge audio hash mismatch")
                if str(receipt.get("label_payload_sha256", "")).casefold() != str(authoritative["label_payload_sha256"]).casefold():
                    raise ValueError(f"{role}/{clip_id} bridge label evidence mismatch")
                if int(row.get("label", -1)) != int(authoritative[expected_binary[role]]) or int(receipt.get("binary_label", -1)) != int(authoritative[expected_binary[role]]):
                    raise ValueError(f"{role}/{clip_id} binary label does not match human consensus")
                if dict(row.get("features") or {}) != dict(receipt.get("features") or {}):
                    raise ValueError(f"{role}/{clip_id} features differ from authoritative bridge receipt")
                if str(receipt.get("feature_schema_version", "")) != expected_schema[role]:
                    raise ValueError(f"{role}/{clip_id} feature schema mismatch")
                evidence_payload = receipt.get("evidence")
                if not isinstance(evidence_payload, Mapping) or str(receipt.get("evidence_sha256", "")).casefold() != sha256_bytes(canonical_json(evidence_payload)):
                    raise ValueError(f"{role}/{clip_id} evidence digest cannot be recomputed")
                if str(metadata.get("label_payload_sha256", metadata.get("label_hash", ""))).casefold() != str(authoritative["label_payload_sha256"]).casefold() or str(metadata.get("evidence_sha256", metadata.get("evidence_hash", ""))).casefold() != str(receipt["evidence_sha256"]).casefold():
                    raise ValueError(f"{role}/{clip_id} row metadata is not authoritative")
                role_labels[f"{role}:{clip_id}"] = {"label_payload_sha256": authoritative["label_payload_sha256"], "binary_label": authoritative[expected_binary[role]]}
                bridge_digests[f"{role}:{clip_id}"] = str(receipt["receipt_sha256"])
            rows.sort(key=lambda item: str(item["clip_id"]))
            rows_by_role[role] = rows
            row_digests[role] = sha256_bytes(canonical_json(rows))
            report = role_hidden_reports[role]
            if not isinstance(report, Mapping) or not str(report.get("run_id", "")).strip():
                raise ValueError(f"{role} hidden report has no one-shot run_id")
        for name, hashes in (("hidden_jsonl", hidden_jsonl_hashes), ("hidden_report", hidden_report_hashes)):
            if set(hashes) != set(roles) or any(not re.fullmatch(r"[0-9a-f]{64}", str(value).casefold()) for value in hashes.values()):
                raise ValueError(f"{name} hashes must cover all roles with SHA-256 values")
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", _text(code_commit)) or not _text(profile_id):
            raise ValueError("profile_id and a real code commit are required")
        payload = {
            "schema": "hidden-evaluation-finalization-v1", "receipt_id": hidden_receipt["receipt_id"],
            "receipt_sha256": hidden_receipt["receipt_sha256"], "run_id": _text(run_id),
            "profile_id": _text(profile_id), "code_commit": _text(code_commit).lower(),
            "sealed_clip_ids": sorted(sealed), "sealed_audio_sha256": {key: sealed[key]["audio_sha256"] for key in sorted(sealed)},
            "role_row_sha256": row_digests, "hidden_jsonl_sha256": dict(hidden_jsonl_hashes),
            "hidden_report_sha256": dict(hidden_report_hashes),
            "authoritative_labels": role_labels, "bridge_receipts": bridge_digests,
            "role_report_run_ids": {role: str(role_hidden_reports[role]["run_id"]) for role in roles},
        }
        finalization_sha = sha256_bytes(canonical_json(payload))
        finalization = {**payload, "finalization_id": f"hidden-final-{finalization_sha[:16]}", "finalization_sha256": finalization_sha}
        try:
            self._db.execute("INSERT INTO hidden_finalizations VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)", (finalization["finalization_id"], hidden_receipt["receipt_id"], _text(run_id), _text(profile_id), _text(code_commit).lower(), canonical_json(finalization), finalization_sha))
        except sqlite3.IntegrityError as exc:
            raise ValueError("this hidden evaluation run has already been finalized") from exc
        self._db.commit()
        return finalization

    @_db_locked
    def verify_hidden_evaluation_finalization(self, finalization_id: str, *, profile_id: str | None = None, code_commit: str | None = None) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM hidden_finalizations WHERE finalization_id=?", (_text(finalization_id),)).fetchone()
        if row is None:
            raise ValueError("hidden finalization is not present in the authoritative store")
        finalization = json.loads(row["payload"])
        payload = {key: finalization[key] for key in finalization if key not in {"finalization_id", "finalization_sha256"}}
        if finalization.get("finalization_sha256") != sha256_bytes(canonical_json(payload)):
            raise ValueError("hidden finalization digest mismatch")
        receipt = self.get_hidden_evaluation_receipt(str(row["receipt_id"]), str(row["run_id"]))
        if finalization.get("receipt_sha256") != receipt.get("receipt_sha256") or finalization.get("sealed_clip_ids") != sorted(str(item["clip_id"]) for item in receipt.get("clips", [])):
            raise ValueError("hidden finalization is not bound to the sealed receipt")
        if profile_id is not None and str(row["profile_id"]) != _text(profile_id):
            raise ValueError("hidden finalization profile identity mismatch")
        if code_commit is not None and str(row["code_commit"]).lower() != _text(code_commit).lower():
            raise ValueError("hidden finalization code identity mismatch")
        return {**finalization, "consumed_at": row["consumed_at"], "consumed_by_profile_id": row["consumed_by_profile_id"], "consumed_by_code_commit": row["consumed_by_code_commit"]}

    @_db_locked
    def consume_hidden_finalization(self, finalization_id: str, *, profile_id: str, code_commit: str) -> dict[str, Any]:
        """Atomically reserve one authoritative finalization for one profile."""
        finalization = self.verify_hidden_evaluation_finalization(finalization_id, profile_id=profile_id, code_commit=code_commit)
        if finalization.get("consumed_at"):
            raise ValueError("hidden finalization has already been consumed")
        from datetime import datetime, timezone
        consumed_at = datetime.now(timezone.utc).isoformat()
        updated = self._db.execute("UPDATE hidden_finalizations SET consumed_at=?, consumed_by_profile_id=?, consumed_by_code_commit=? WHERE finalization_id=? AND consumed_at IS NULL", (consumed_at, _text(profile_id), _text(code_commit).lower(), _text(finalization_id)))
        if updated.rowcount != 1:
            self._db.rollback(); raise ValueError("hidden finalization was consumed concurrently")
        self._db.commit()
        return self.verify_hidden_evaluation_finalization(finalization_id, profile_id=profile_id, code_commit=code_commit)

    @_db_locked
    def clips(self) -> list[ClipRecord]:
        return [ClipRecord(**json.loads(row["payload"])) for row in self._db.execute("SELECT payload FROM clips ORDER BY clip_id")]

    @_db_locked
    def labels(self) -> list[HumanLabel]:
        result = []
        adjudicated = {row["clip_id"]: row["adjudicator_id"] for row in self._db.execute("SELECT clip_id, adjudicator_id FROM adjudications")}
        for row in self._db.execute("SELECT payload FROM labels ORDER BY clip_id, reviewer_id"):
            value = json.loads(row["payload"]); value["affected_tokens"] = tuple(value.get("affected_tokens") or []); value["labels"] = tuple(value.get("labels") or ([value.get("label")] if value.get("label") else []))
            if value.get("clip_id") in adjudicated: value["adjudicated_by"] = adjudicated[value["clip_id"]]
            result.append(HumanLabel(**value))
        return result

    @_db_locked
    def effective_labels(self) -> list[HumanLabel]:
        """Return one authoritative label per clip for calibration.

        Reviewer rows remain available through :meth:`labels` as the audit
        trail.  Once an adjudication exists it is the only label consumed by
        feature extraction; silently unioning conflicting reviewer labels
        would manufacture a negative from a disagreement that a lead already
        resolved.
        """
        reviewer_labels = self.labels()
        adjudications = {
            row["clip_id"]: (str(row["adjudicator_id"]), tuple(json.loads(row["consensus_labels"])))
            for row in self._db.execute("SELECT clip_id, adjudicator_id, consensus_labels FROM adjudications")
        }
        by_clip: dict[str, list[HumanLabel]] = {}
        for label in reviewer_labels:
            by_clip.setdefault(label.clip_id, []).append(label)
        result: list[HumanLabel] = []
        for clip_id in sorted(by_clip):
            if clip_id in adjudications:
                adjudicator, consensus = adjudications[clip_id]
                result.append(HumanLabel(clip_id, f"adjudicator:{adjudicator}", labels=consensus, adjudicated_by=adjudicator))
            else:
                result.extend(by_clip[clip_id])
        return result

    def _hidden_seal_payload(self) -> list[dict[str, Any]]:
        """Return hidden membership plus current effective label evidence."""
        hidden_ids = {str(row["clip_id"]) for row in self._db.execute("SELECT clip_id FROM clips WHERE split='hidden_test'")}
        labels = [label.to_dict() for label in self.effective_labels() if label.clip_id in hidden_ids]
        rows = [{"clip_id": row["clip_id"], "payload": json.loads(row["payload"]), "audio_sha256": row["audio_sha256"], "split": row["split"], "split_group": row["split_group"]} for row in self._db.execute("SELECT clip_id, payload, audio_sha256, split, split_group FROM clips WHERE split='hidden_test' ORDER BY clip_id")]
        return [{"clip": row, "effective_labels": [label for label in labels if label.get("clip_id") == row["clip_id"]]} for row in rows]

    @_db_locked
    def verify_hidden_seal(self) -> bool:
        seal = self.hidden_seal()
        return bool(seal and sha256_bytes(canonical_json(self._hidden_seal_payload())) == str(seal["digest"]).casefold())

    @_db_locked
    def export(self, directory: str | Path, *, include_hidden: bool = False) -> dict[str, str]:
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        clips = self.clips(); labels = self.labels(); effective = self.effective_labels()
        hidden = [clip for clip in clips if clip.split == "hidden_test"]
        seal = self.hidden_seal()
        if hidden and seal is None:
            raise ValueError("hidden test must be sealed before export")
        if not include_hidden:
            labels = [label for label in labels if next((clip.split for clip in clips if clip.clip_id == label.clip_id), "") != "hidden_test"]
        paths = {"manifest": root / "manifest.jsonl", "labels": root / "labels.jsonl", "effective_labels": root / "effective_labels.jsonl", "splits": root / "splits.json", "reviewers": root / "reviewers.json", "disagreements": root / "disagreements.jsonl"}
        paths["manifest"].write_text("".join(canonical_json(c.to_dict()) + "\n" for c in clips), encoding="utf-8")
        paths["labels"].write_text("".join(canonical_json(l.to_dict()) + "\n" for l in labels), encoding="utf-8")
        paths["effective_labels"].write_text("".join(canonical_json(l.to_dict()) + "\n" for l in effective if include_hidden or next((clip.split for clip in clips if clip.clip_id == l.clip_id), "") != "hidden_test"), encoding="utf-8")
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


def verify_hidden_evaluation_receipt_payload(receipt: Mapping[str, Any]) -> bool:
    """Verify the receipt's canonical self-digest without trusting its claims."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "hidden-evaluation-receipt-v1":
        return False
    try:
        expected = str(receipt["receipt_sha256"])
        payload = {key: receipt[key] for key in ("schema", "run_id", "operator_id", "seal_id", "seal_digest", "opened_at", "clips")}
    except (KeyError, TypeError):
        return False
    return expected == sha256_bytes(canonical_json(payload)) and str(receipt.get("receipt_id", "")) == f"hidden-eval-{expected[:16]}"


__all__ = ["LABELS", "SPLITS", "ClipRecord", "HumanLabel", "GoldsetStore", "stable_split", "validate_goldset", "manifest_hash", "verify_hidden_evaluation_receipt_payload"]
