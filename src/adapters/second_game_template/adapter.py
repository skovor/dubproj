from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from pathlib import Path
import hashlib, inspect, re
from dubbing_pipeline.audio import read
from dubbing_pipeline.hashing import canonical_json, sha256_bytes
from dubbing_pipeline.attestation import verify_trusted_attestation

@dataclass(frozen=True)
class SecondGameAdapter:
    game_id: str
    adapter_version: str = "1"
    adapter_role: str = "second_game_extractor"
    def validate(self, manifest: dict[str, Any], *, expected_commit: str | None = None, attestation_trust_store: Mapping[str, Any] | None = None, attestation_key_id: str | None = None) -> dict[str, Any]:
        if not self.game_id.strip(): raise ValueError("game_id required")
        errors=[]; scenes=manifest.get("scenes")
        if not isinstance(scenes,list) or not scenes: errors.append("scenes_required")
        verified=[]
        for index, scene in enumerate(scenes or []):
            if not isinstance(scene,dict): errors.append(f"scene_{index}_must_be_object"); continue
            audio_duration = reference_duration = 0.0
            if "game_id" not in scene: errors.append(f"scene_{index}_game_id_missing")
            if str(scene.get("game_id", "")) != self.game_id: errors.append(f"scene_{index}_game_id_mismatch")
            if not str(scene.get("scene_id", "")).strip(): errors.append(f"scene_{index}_scene_id_missing")
            audio=Path(str(scene.get("audio_path", ""))); reference=Path(str(scene.get("reference_path", "")))
            if not audio.is_file(): errors.append(f"scene_{index}_audio_missing")
            if not reference.is_file(): errors.append(f"scene_{index}_reference_missing")
            timing=scene.get("timing")
            if not isinstance(timing,dict) or float(timing.get("start", -1)) < 0 or not float(timing.get("end",0)) > float(timing.get("start",0)):
                errors.append(f"scene_{index}_timing_missing")
            audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest() if audio.is_file() else ""
            reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest() if reference.is_file() else ""
            if audio.is_file() and scene.get("audio_sha256") != audio_hash: errors.append(f"scene_{index}_audio_hash_mismatch")
            if reference.is_file() and scene.get("reference_sha256") != reference_hash: errors.append(f"scene_{index}_reference_hash_mismatch")
            source_container = Path(str(scene.get("source_container_path", ""))); source_hash = hashlib.sha256(source_container.read_bytes()).hexdigest() if source_container.is_file() else ""
            if not source_container.is_file(): errors.append(f"scene_{index}_source_container_missing")
            if scene.get("source_container_sha256") != source_hash: errors.append(f"scene_{index}_source_container_hash_mismatch")
            adapter_source = Path(inspect.getsourcefile(type(self)) or ""); adapter_source_hash = hashlib.sha256(adapter_source.read_bytes()).hexdigest() if adapter_source.is_file() else ""
            if scene.get("extractor_source_sha256") != adapter_source_hash: errors.append(f"scene_{index}_extractor_source_hash_mismatch")
            for label, path in (("audio", audio), ("reference", reference)):
                if path.is_file():
                    try:
                        samples, rate = read(path, always_2d=True)
                        import numpy as np
                        if samples.ndim != 2 or samples.shape[1] <= 0 or len(samples) <= 0 or int(rate) <= 0 or not bool(np.isfinite(samples).all()): raise ValueError("invalid decoded frames/rate/channels/finiteness")
                        if label == "audio": audio_duration = len(samples) / int(rate)
                        else: reference_duration = len(samples) / int(rate)
                    except Exception as exc:
                        errors.append(f"scene_{index}_{label}_decode_invalid:{exc}")
            if scene.get("extraction_status") != "VERIFIED": errors.append(f"scene_{index}_extraction_unverified")
            if isinstance(timing, dict) and "audio_duration" in locals() and "reference_duration" in locals():
                tolerance = 0.01
                if float(timing.get("end", 0)) > audio_duration + tolerance or float(timing.get("end", 0)) > reference_duration + tolerance: errors.append(f"scene_{index}_timing_exceeds_audio")
            receipt = scene.get("extraction_receipt")
            if not isinstance(receipt, dict) or receipt.get("schema") != "second-game-extraction-receipt-v1":
                errors.append(f"scene_{index}_extraction_receipt_missing")
            else:
                receipt_payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
                if str(receipt.get("receipt_sha256", "")).casefold() != sha256_bytes(canonical_json(receipt_payload)):
                    errors.append(f"scene_{index}_extraction_receipt_hash_mismatch")
                for key, expected in (("scene_id", scene.get("scene_id")), ("game_id", self.game_id), ("audio_sha256", audio_hash), ("reference_sha256", reference_hash)):
                    if str(receipt.get(key, "")).casefold() != str(expected or "").casefold(): errors.append(f"scene_{index}_receipt_{key}_mismatch")
                for key in ("extractor_id", "extractor_version", "extractor_source_sha256", "source_container_sha256", "code_commit"):
                    if not str(receipt.get(key, "")).strip(): errors.append(f"scene_{index}_receipt_{key}_missing")
                if not re.fullmatch(r"[0-9a-fA-F]{64}", str(receipt.get("source_container_sha256", ""))): errors.append(f"scene_{index}_source_container_hash_invalid")
                for key, expected in (("source_container_sha256", source_hash), ("extractor_source_sha256", adapter_source_hash)):
                    if str(receipt.get(key, "")).lower() != str(expected).lower(): errors.append(f"scene_{index}_receipt_{key}_mismatch")
                if expected_commit is not None and str(receipt.get("code_commit", "")).lower() != str(expected_commit).lower(): errors.append(f"scene_{index}_receipt_commit_mismatch")
                if not re.fullmatch(r"[0-9a-fA-F]{40}", str(receipt.get("code_commit", ""))): errors.append(f"scene_{index}_receipt_commit_invalid")
                subject = {"schema": "second-game-extraction-subject-v1", "repository": "skovor/dubproj", "workflow": ".github/workflows/ci.yml", "game_id": self.game_id, "scene_id": scene.get("scene_id"), "source_container_sha256": source_hash, "audio_sha256": audio_hash, "reference_sha256": reference_hash, "timing": timing, "extractor_id": receipt.get("extractor_id"), "extractor_version": receipt.get("extractor_version"), "extractor_source_sha256": adapter_source_hash, "code_commit": receipt.get("code_commit")}
                if attestation_trust_store is None or not attestation_key_id or not verify_trusted_attestation(receipt.get("attestation") or {}, attestation_trust_store, key_id=attestation_key_id, expected_subject=subject, expected_commit=str(receipt.get("code_commit", "")), repository="skovor/dubproj", workflow=".github/workflows/ci.yml"): errors.append(f"scene_{index}_extraction_attestation_invalid")
            verified.append({"scene_id":scene.get("scene_id"),"audio_path":str(audio),"reference_path":str(reference)})
        return {"valid":not errors,"content_verified":not errors and bool(verified),"game_id":self.game_id,"adapter_version":self.adapter_version,"adapter_role":self.adapter_role,"independent_adapter":not errors and bool(verified),"verified_scenes":verified,"errors":errors}

