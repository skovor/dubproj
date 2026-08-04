---
name: packaging-runtime
description: Empaqueta solo unidades RELEASE_READY con staging limpio, y no despliega sin backup, hash y smoke real dentro del juego. Úsala en fases J (empaquetado) y K (deploy y smoke).
---

# packaging-runtime

Empaqueta solo lo que el plan vigente marca `RELEASE_READY`, con staging
limpio, y no despliega sin backup, hash y smoke real dentro del juego.

## Triggers

- Empaquetar salidas generadas hacia el formato de release del juego.
- Desplegar al juego real (`deploy`) o probar en runtime (`runtime_probe`).

## Inputs esperados

- El plan/manifiesto vigente con la acción actual de cada unidad.
- `GATE_STATUS.json` con evidencia real (no la plantilla vacía).

## Límites operativos

- Empaquetar SOLO `RELEASE_READY`; el payload original se deja intacto para
  `KEEP_ORIGINAL`/holds. No convertir "existe físicamente en produccion/"
  en autorización de release.
- Staging nuevo o limpio por corrida: no mezclar contenedores válidos con
  residuos de una corrida fallida.
- El backup de una sonda de runtime y el backup de release son gates
  distintos, con evidencia propia cada uno.
- `scripts/operation_guard.py` bloquea `generate`/`package`/`runtime_probe`/
  `deploy` sin autorización y evidencia — no eludir el bloqueo modificando
  el estado sin evidencia reproducible.

## Procedimiento

1. Resolver la acción ACTUAL de cada unidad desde el manifiesto antes de
   tocar ningún WAV/contenedor.
2. Empaquetar en staging nuevo, verificar hash/roundtrip del paquete final.
3. `python scripts/operation_guard.py --operation deploy --status GATE_STATUS.json`
   antes de copiar nada al juego.
4. Backup del estado previo, deploy, smoke real dentro del juego en tres
   capas (carga, reproducción, ausencia de regresión), triaje por síntoma
   si algo falla.

## Artefactos de salida

- Paquete de release con manifiesto de lo incluido/excluido y por qué.
- Reporte de smoke con resultado por capa.

## Gates afectados

`package_roundtrip_pass`, `staging_reconciled`, `runtime_probe_backup_ready`,
`release_backup_created`, `runtime_paths_verified`, `smoke_plan_ready`,
`smoke_passed`.

## Referencias canónicas

- `docs/canonical/60_PACKAGE_DEPLOY_AND_SMOKE.md`
- `docs/phases/J_PACKAGING.md`, `docs/phases/K_DEPLOY_AND_SMOKE.md`
- `scripts/operation_guard.py`, `docs/GATE_OWNERSHIP.md`

## Reglas promovidas

- `AC-64`: persistir el estado de corridas largas de forma atomica y
  reanudable, tolerando locks transitorios.
- `AC-65`: reauditar los archivos finales despues de serializar y remuxear,
  no confiar unicamente en el manifiesto.
- `AC-66`: un diagnostico no codificable no puede bloquear la validacion del
  contenedor; conservar el error como evidencia y continuar el gate tecnico.

## Validación

```bash
python scripts/operation_guard.py --operation deploy --status GATE_STATUS.json
python scripts/test_operation_guard.py
```
