#!/usr/bin/env python3
"""Despliega únicamente rutas exactas probadas por [vp] y el ACB.

La fuente de verdad es ``exact_physical_routes_final.json``. Nunca se deriva
el cue desde el número de línea ni desde el número de stream físico.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from deploy_shared_full_to_ryo import encode


ROOT = Path(__file__).resolve().parent
ROUTES = ROOT / "event_physical_routes_final.json"
PRODUCTION = ROOT / "produccion"
STAGING = ROOT / "exact_ryo_staging"
REPORT = ROOT / "exact_ryo_deploy_report.json"
MOD = Path(r"C:\Users\juand\Desktop\Reloaded-II\Mods\p3r.doblaje.aleman")
RYO = MOD / "ryo" / "P3R"


def externally_managed_routes() -> set[tuple[str, int]]:
    """Rutas que no debe borrar el despliegue exacto al quedar fuera del plan."""
    result = set()
    for filename in (
        "shared_ryo_deploy_report.json",
        "shared_delivery_ryo_deploy_report.json",
    ):
        payload = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        result.update(
            (row["bank"], int(row["cue"]))
            for row in payload.get("details", [])
        )
    composite = json.loads(
        (ROOT / "composite_event_generation_deploy_report.json").read_text(
            encoding="utf-8"
        )
    )
    result.update(
        (f"Voice_Event_{row['event']}", int(row["cue"]))
        for row in composite.get("assets", [])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    previous = (
        json.loads(REPORT.read_text(encoding="utf-8"))
        if REPORT.is_file()
        else {"details": []}
    )
    route_payload = json.loads(ROUTES.read_text(encoding="utf-8"))
    selected = [
        row
        for row in route_payload["routes"]
        if row["disposition"] == "READY_EVENT_ROUTE"
    ]

    routes = {}
    for row in selected:
        destination = (row["bank"], int(row["cue"]))
        if destination in routes:
            raise RuntimeError(f"Colisión exacta: {destination}")
        stem = row["unit_id"].removeprefix("VU_")
        source = PRODUCTION / f"{stem}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        routes[destination] = {"row": row, "source": source}
    if len(routes) != route_payload["ready_routes"]:
        raise RuntimeError("El contrato de rutas finales no coincide")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    details = []
    for number, ((bank, cue), item) in enumerate(sorted(routes.items()), 1):
        target = STAGING / f"{bank}.acb" / f"{cue}.hca"
        contract = encode(item["source"], target)
        details.append(
            {
                "unit_id": item["row"]["unit_id"],
                "bank": bank,
                "cue": cue,
                "physical_stream": item["row"].get("physical_stream"),
                "reference_wav": item["row"]["reference_wav"],
                "source": str(item["source"]),
                "contract": contract,
            }
        )
        if number % 50 == 0:
            print(f"[{number}/{len(routes)}] rutas codificadas", flush=True)
    for bank_dir in STAGING.glob("*.acb"):
        (bank_dir / "config.yaml").write_text(
            "volume: 1.0\nuse_player_volume: false\n",
            encoding="utf-8",
        )

    old_keys = {
        (row["bank"], int(row["cue"])) for row in previous.get("details", [])
    }
    new_keys = set(routes)
    protected_keys = externally_managed_routes()
    stale_keys = sorted(old_keys - new_keys - protected_keys)
    deployed = False
    backup = None
    removed_stale = []
    if args.deploy:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MOD / "_backup_exact_ryo" / stamp
        touched = old_keys | new_keys
        for bank, cue in sorted(touched):
            old = RYO / f"{bank}.acb" / f"{cue}.hca"
            if old.is_file():
                old_backup = backup / f"{bank}.acb"
                old_backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, old_backup / old.name)
        for bank, cue in stale_keys:
            old = RYO / f"{bank}.acb" / f"{cue}.hca"
            if old.is_file():
                old.unlink()
                removed_stale.append({"bank": bank, "cue": cue})
        for bank_dir in sorted(STAGING.glob("*.acb")):
            destination = RYO / bank_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for item in bank_dir.iterdir():
                shutil.copy2(item, destination / item.name)
        backup.mkdir(parents=True, exist_ok=True)
        (backup / "deployment_manifest.json").write_text(
            json.dumps(
                {
                    "old_routes": len(old_keys),
                    "new_routes": len(new_keys),
                    "stale_routes": [
                        {"bank": bank, "cue": cue} for bank, cue in stale_keys
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        deployed = True

    payload = {
        "selected": len(selected),
        "deployed_routes": len(routes),
        "banks": len({bank for bank, _ in routes}),
        "excluded": [
            row
            for row in route_payload["routes"]
            if row["disposition"] != "READY_EVENT_ROUTE"
        ],
        "old_routes": len(old_keys),
        "stale_routes": len(stale_keys),
        "protected_external_routes": len(protected_keys),
        "removed_stale": removed_stale,
        "deployed": deployed,
        "backup": str(backup) if backup else None,
        "details": details,
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "details"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
