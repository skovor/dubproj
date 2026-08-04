#!/usr/bin/env python3
"""Validate the portable skill/rule bundle and promotion provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not match:
        raise ValueError(f"missing frontmatter: {path}")
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1); values[key.strip()] = value.strip()
    return values


def validate(root: Path, write_manifest: bool = False) -> dict:
    skills = []
    missing_references = []
    for skill_path in sorted((root / "skills-src").glob("*/SKILL.md")):
        meta_path = skill_path.with_name("meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")); fm = frontmatter(skill_path)
        if meta.get("name") != skill_path.parent.name or fm.get("name") != meta.get("name"):
            raise ValueError(f"skill name mismatch: {skill_path}")
        if "description" not in fm:
            raise ValueError(f"skill description missing: {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        for reference in re.findall(r"`?((?:docs|scripts)/[^`\s)]+)", text):
            candidate = root.parent.parent / reference if reference.startswith("scripts/") else root.parent.parent / reference
            if not candidate.exists() and reference not in missing_references:
                missing_references.append(reference)
        skills.append({"name": meta["name"], "path": str(skill_path.relative_to(root)), "sha256": digest(skill_path), "meta_sha256": digest(meta_path), "phases": meta.get("phases", []), "triggers": meta.get("triggers", [])})
    promotion = json.loads((root / "PROMOTION_MANIFEST.json").read_text(encoding="utf-8"))
    rules = []
    for rule_path in sorted((root / "rules").glob("AC-*.md")):
        rule_id = rule_path.stem
        if not any(item.get("rule_id") == rule_id for item in promotion.get("rules", [])):
            raise ValueError(f"rule absent from promotion manifest: {rule_id}")
        rules.append({"rule_id": rule_id, "path": str(rule_path.relative_to(root)), "sha256": digest(rule_path)})
    report = {"schema": "instruction-bundle-validation-v2", "validated_utc": datetime.now(timezone.utc).isoformat(), "skills": skills, "rules": rules, "external_references": sorted(missing_references), "self_contained_code": True, "kiro_included": False}
    if write_manifest:
        (root / "BUNDLE_MANIFEST_V2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "instructions")); parser.add_argument("--write-manifest", action="store_true"); args = parser.parse_args()
    report = validate(Path(args.root), args.write_manifest); print(json.dumps({"status": "PASS", "skills": len(report["skills"]), "rules": len(report["rules"]), "external_references": len(report["external_references"]), "manifest": str(Path(args.root) / "BUNDLE_MANIFEST_V2.json") if args.write_manifest else None}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
