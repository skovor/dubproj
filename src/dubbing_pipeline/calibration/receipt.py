"""Canonical promotion payloads shared by promotion and runtime QA.

The receipt is intentionally content-addressed.  It excludes only values that
would create a self-referential cycle (creation time and the receipt's own
path/hash); every value that can alter a hard decision remains in the payload.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..hashing import canonical_json, sha256_bytes


def promotion_profile_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical decision-bearing subset of a promoted profile."""
    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    payload = {
        "schema": profile.get("schema"),
        "profile_id": profile.get("profile_id"),
        "identity": profile.get("identity"),
        "thresholds": profile.get("thresholds"),
        "calibrators": profile.get("calibrators"),
        "dataset": profile.get("dataset"),
        "metrics": profile.get("metrics"),
        "provenance": {
            key: provenance.get(key)
            for key in (
                "code_commit",
                "runtime_lock_sha256",
                "models_lock_sha256",
                "hidden_test_run_id",
            )
            if key in provenance
        },
    }
    return payload


def promotion_profile_payload_sha256(profile: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(promotion_profile_payload(profile)))


__all__ = ["promotion_profile_payload", "promotion_profile_payload_sha256"]
