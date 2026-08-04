"""Atomic SQLite state transitions for resumable cost routing."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class StateStore:
    def __init__(self,path: str|Path):
        self.db=sqlite3.connect(str(path)); self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("CREATE TABLE IF NOT EXISTS line_state (line_id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL)"); self.db.commit()
    def transition(self,line_id: str,state: str,payload: str="{}")->None:
        now=datetime.now(timezone.utc).isoformat(); self.db.execute("INSERT INTO line_state VALUES (?,?,?,?) ON CONFLICT(line_id) DO UPDATE SET state=excluded.state,payload=excluded.payload,updated_at=excluded.updated_at",(line_id,state,payload,now)); self.db.commit()
    def get(self,line_id: str): return self.db.execute("SELECT line_id,state,payload,updated_at FROM line_state WHERE line_id=?",(line_id,)).fetchone()
    def close(self): self.db.close()

__all__=["StateStore"]
