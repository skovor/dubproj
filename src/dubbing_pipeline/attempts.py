"""Content-addressed repair attempt journal."""
from __future__ import annotations
import json, sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from .hashing import canonical_json, contract_hash

@dataclass(frozen=True)
class AttemptSignature:
    line_id: str
    input_audio_sha256: str
    strategy: str
    parameters: dict[str, Any]
    reference_sha256: str | None = None
    def digest(self) -> str: return contract_hash("repair-attempt-v1", self.__dict__)

class AttemptStore:
    def __init__(self, path: str | Path):
        self.db=sqlite3.connect(str(path)); self.db.execute("CREATE TABLE IF NOT EXISTS attempts (signature TEXT PRIMARY KEY, signature_payload TEXT NOT NULL DEFAULT '{}', payload TEXT NOT NULL, status TEXT NOT NULL)")
        columns={row[1] for row in self.db.execute("PRAGMA table_info(attempts)")}
        if "signature_payload" not in columns:
            self.db.execute("ALTER TABLE attempts ADD COLUMN signature_payload TEXT NOT NULL DEFAULT '{}'")
        self.db.commit()
    def seen(self, signature: AttemptSignature) -> bool: return self.db.execute("SELECT 1 FROM attempts WHERE signature=?",(signature.digest(),)).fetchone() is not None
    def count_causal(self, signature: AttemptSignature) -> int:
        """Count attempts for one causal input/strategy, regardless of params."""
        total = 0
        for payload, in self.db.execute("SELECT signature_payload FROM attempts"):
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if value.get("line_id") == signature.line_id and value.get("input_audio_sha256") == signature.input_audio_sha256 and value.get("strategy") == signature.strategy and value.get("reference_sha256") == signature.reference_sha256:
                total += 1
        return total
    def record(self, signature: AttemptSignature, payload: Mapping[str, Any], status: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO attempts(signature,signature_payload,payload,status) VALUES (?, ?, ?, ?)",(signature.digest(),canonical_json(signature.__dict__),canonical_json(dict(payload)),status)); self.db.commit()
    def close(self): self.db.close()

__all__=["AttemptSignature","AttemptStore"]
