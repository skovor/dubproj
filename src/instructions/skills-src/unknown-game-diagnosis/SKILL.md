---
name: unknown-game-diagnosis
description: Resuelve la topología de un motor, contenedor o formato de juego desconocido con un procedimiento de 8 pasos y máximo tres sondas reversibles, sin generar audio ni desplegar nada. Úsala cuando una fase A-K está bloqueada por algo sin clasificar, o al entrar a un juego nuevo.
---

# unknown-game-diagnosis

Resuelve la topología de un activo desconocido (motor, contenedor, formato)
sin generar ni desplegar, con un presupuesto fijo de sondas.

## Triggers

- Motor, middleware o formato de contenedor nunca visto en este proyecto.
- Una fase A–K está bloqueada por algo que no se sabe clasificar todavía.

## Inputs esperados

- Rutas de los archivos/paquetes del juego en cuestión.
- Cualquier hallazgo previo (aunque sea parcial) de sesiones anteriores.

## Límites operativos

- Solo lectura: sin TTS, sin deploy, sin empaquetado de release.
- Máximo tres sondas reversibles.
- No usar generación para "probar" un mapping todavía no demostrado.
- No recorrer el árbol de assets completo si una sonda focalizada alcanza.

## Procedimiento

Ver `docs/canonical/X_DIAGNOSIS.md` — los 8 pasos completos (inventario de
solo lectura, hechos vs. inferencias, identificación de motor/middleware,
clasificación del bloqueo, sondas, control positivo, condición de parada,
adapter/playbook resultante) y las 8 clases de bloqueo P1–P8.

## Artefactos de salida

- Hechos verificados e inferencias, separados.
- Clase de bloqueo identificada (P1–P8).
- Un adapter/playbook en `docs/adapters/<juego-o-motor>/`, o el bloqueo
  exacto si no se resolvió.

## Gates afectados

Ninguno de A–K se cierra desde aquí. El resultado habilita (o no) volver a
esas fases con la topología ya resuelta.

## Referencias canónicas

- `docs/canonical/X_DIAGNOSIS.md`
- `docs/adapters/_template/`

## Validación

```bash
python scripts/build_session_packet.py --phase X_DIAGNOSIS --agent <claude|codex>
```
