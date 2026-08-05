from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import hashlib

@dataclass(frozen=True)
class SecondGameAdapter:
    game_id: str
    independent_adapter: bool = True
    adapter_version: str = "1"
    def validate(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if not self.game_id.strip(): raise ValueError("game_id required")
        errors=[]; scenes=manifest.get("scenes")
        if not self.independent_adapter: errors.append("adapter_not_independent")
        if not isinstance(scenes,list) or not scenes: errors.append("scenes_required")
        verified=[]
        for index, scene in enumerate(scenes or []):
            if not isinstance(scene,dict): errors.append(f"scene_{index}_must_be_object"); continue
            if str(scene.get("game_id",self.game_id)) != self.game_id: errors.append(f"scene_{index}_game_id_mismatch")
            audio=Path(str(scene.get("audio_path", ""))); reference=Path(str(scene.get("reference_path", "")))
            if not audio.is_file(): errors.append(f"scene_{index}_audio_missing")
            if not reference.is_file(): errors.append(f"scene_{index}_reference_missing")
            timing=scene.get("timing")
            if not isinstance(timing,dict) or not float(timing.get("end",0)) > float(timing.get("start",0)):
                errors.append(f"scene_{index}_timing_missing")
            if audio.is_file() and scene.get("audio_sha256") != hashlib.sha256(audio.read_bytes()).hexdigest(): errors.append(f"scene_{index}_audio_hash_mismatch")
            if reference.is_file() and scene.get("reference_sha256") != hashlib.sha256(reference.read_bytes()).hexdigest(): errors.append(f"scene_{index}_reference_hash_mismatch")
            if scene.get("extraction_status") != "VERIFIED": errors.append(f"scene_{index}_extraction_unverified")
            verified.append({"scene_id":scene.get("scene_id"),"audio_path":str(audio),"reference_path":str(reference)})
        return {"valid":not errors,"content_verified":not errors and bool(verified),"game_id":self.game_id,"adapter_version":self.adapter_version,"independent_adapter":self.independent_adapter,"verified_scenes":verified,"errors":errors}

