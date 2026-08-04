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
        self.db=sqlite3.connect(str(path)); self.db.execute("CREATE TABLE IF NOT EXISTS attempts (signature TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL)"); self.db.commit()
    def seen(self, signature: AttemptSignature) -> bool: return self.db.execute("SELECT 1 FROM attempts WHERE signature=?",(signature.digest(),)).fetchone() is not None
    def record(self, signature: AttemptSignature, payload: Mapping[str, Any], status: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO attempts VALUES (?, ?, ?)",(signature.digest(),canonical_json(dict(payload)),status)); self.db.commit()
    def close(self): self.db.close()

__all__=["AttemptSignature","AttemptStore"]
