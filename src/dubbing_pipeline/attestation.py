"""Ed25519 attestations for benchmark evidence.

The benchmark runner may emit an unsigned diagnostic result, but promotion can
only accept a signature over the recomputed evidence subject.
"""
from __future__ import annotations

import base64
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json, sha256_bytes


def subject_digest(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(payload)))


def sign_attestation(payload: Mapping[str, Any], private_key_b64: str, *, key_id: str) -> dict[str, Any]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    subject = dict(payload)
    message = canonical_json(subject)
    signature = key.sign(message)
    return {"schema": "benchmark-attestation-v1", "key_id": str(key_id), "subject": subject, "subject_sha256": sha256_bytes(message), "signature": base64.b64encode(signature).decode("ascii"), "verified": True}


def verify_attestation(attestation: Mapping[str, Any], public_key_b64: str, *, expected_subject: Mapping[str, Any], expected_commit: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    import re
    if attestation.get("schema") != "benchmark-attestation-v1" or not attestation.get("key_id") or attestation.get("subject") != dict(expected_subject):
        return False
    subject = dict(attestation.get("subject") or {})
    if str(subject.get("code_commit", "")).lower() != str(expected_commit).lower() or not re.fullmatch(r"[0-9a-fA-F]{40}", str(expected_commit)):
        return False
    message = canonical_json(subject)
    if str(attestation.get("subject_sha256", "")).lower() != sha256_bytes(message):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64)).verify(base64.b64decode(str(attestation.get("signature", ""))), message)
    except Exception:
        return False
    return True


def validate_trust_store(store: Mapping[str, Any]) -> dict[str, Any]:
    if store.get("schema") != "attestation-trust-store-v1" or not isinstance(store.get("keys"), Mapping) or not store["keys"]:
        raise ValueError("invalid attestation trust store schema")
    validated: dict[str, Any] = {}
    for key_id, value in store["keys"].items():
        if not isinstance(value, Mapping) or value.get("algorithm") != "ed25519":
            raise ValueError(f"invalid trust anchor algorithm: {key_id}")
        public = str(value.get("public_key", ""))
        try: raw = base64.b64decode(public, validate=True)
        except Exception as exc: raise ValueError(f"invalid trust anchor key: {key_id}") from exc
        fingerprint = hashlib.sha256(raw).hexdigest()
        if len(raw) != 32 or str(value.get("public_key_sha256", "")).lower() != fingerprint:
            raise ValueError(f"trust anchor fingerprint mismatch: {key_id}")
        if not str(value.get("allowed_repository", "")).strip() or not str(value.get("allowed_workflow", "")).strip():
            raise ValueError(f"trust anchor scope is incomplete: {key_id}")
        validated[str(key_id)] = {**dict(value), "public_key_sha256": fingerprint}
    return validated


def load_trust_store(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_trusted_attestation(attestation: Mapping[str, Any], store: Mapping[str, Any], *, key_id: str, expected_subject: Mapping[str, Any], expected_commit: str, repository: str, workflow: str) -> bool:
    anchors = validate_trust_store(store)
    anchor = anchors.get(str(key_id))
    if anchor is None or str(attestation.get("key_id")) != str(key_id): return False
    subject = dict(attestation.get("subject") or {})
    if subject != dict(expected_subject): return False
    if str(anchor["allowed_repository"]) != str(repository) or str(anchor["allowed_workflow"]) != str(workflow): return False
    if str(subject.get("repository")) != str(repository) or str(subject.get("workflow")) != str(workflow): return False
    return verify_attestation(attestation, str(anchor["public_key"]), expected_subject=expected_subject, expected_commit=expected_commit)


__all__ = ["load_trust_store", "sign_attestation", "subject_digest", "validate_trust_store", "verify_attestation", "verify_trusted_attestation"]
