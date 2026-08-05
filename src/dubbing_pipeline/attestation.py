"""Ed25519 attestations for benchmark evidence.

The benchmark runner may emit an unsigned diagnostic result, but promotion can
only accept a signature over the recomputed evidence subject.
"""
from __future__ import annotations

import base64
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


__all__ = ["sign_attestation", "subject_digest", "verify_attestation"]
