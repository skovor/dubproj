"""Run-scoped resident model pool; one load per identity and explicit close."""
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

@dataclass(frozen=True)
class ModelIdentity:
    role: str
    model_id: str
    revision: str
    device: str = "cpu"
    dtype: str = "float32"

class ModelPool:
    def __init__(self): self._models: dict[ModelIdentity, Any] = {}; self._loads: dict[ModelIdentity, int] = {}; self._lock=RLock(); self.closed=False
    def get(self, identity: ModelIdentity, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if self.closed: raise RuntimeError("model pool is closed")
            if identity not in self._models: self._models[identity]=loader(); self._loads[identity]=self._loads.get(identity,0)+1
            return self._models[identity]
    def loaded(self) -> tuple[ModelIdentity, ...]: return tuple(self._models)
    def load_counts(self) -> dict[ModelIdentity,int]: return dict(self._loads)
    def close(self, closer: Callable[[Any],None] | None = None) -> None:
        with self._lock:
            if closer:
                for model in self._models.values(): closer(model)
            self._models.clear(); self.closed=True
    def __enter__(self): return self
    def __exit__(self,*_): self.close()

__all__=["ModelIdentity","ModelPool"]
