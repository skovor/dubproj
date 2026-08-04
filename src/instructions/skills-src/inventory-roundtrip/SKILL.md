---
name: inventory-roundtrip
description: Construye el inventario físico completo de un juego (audio, video, texto, control de eventos) y prueba round-trip de sus contenedores antes de que ninguna otra fase dependa de esos datos. Úsala en fase A (inventario) y B (round-trip).
---

# inventory-roundtrip

Inventario físico completo del activo y prueba de round-trip de sus
contenedores, antes de que ninguna otra fase dependa de esos datos.

## Triggers

- Arranque de un proyecto nuevo (fase A).
- Necesidad de reconciliar el inventario físico contra mapas/bancos/texto
  antes de declarar cobertura (fase B).

## Inputs esperados

- Ruta al paquete/instalación del juego.
- Herramientas de extracción disponibles y su versión.

## Límites operativos

- Solo lectura en A; el round-trip de B debe ser reversible (extraer →
  reempaquetar → comparar, sin dejar el activo modificado).
- No declarar cobertura completa sin reconciliar TODOS los inventarios
  (audio, video, texto, control de eventos).

## Procedimiento

1. Enumerar y localizar físicamente contenedores de audio, video,
   localización y scripts/control de eventos.
2. Confirmar el middleware por evidencia binaria o de metadatos, no por
   parecido de nombre.
3. Probar round-trip: extraer, reempaquetar, comparar bit a bit o por hash
   donde el formato lo permita.
4. Registrar cualquier discrepancia como `UNPROVEN_HOLD`, nunca omitirla en
   silencio.

## Artefactos de salida

- Inventario físico con conteos exactos por tipo de contenedor.
- Reporte de round-trip (éxito/fallo por contenedor).

## Gates afectados

`inventory_complete`, `roundtrip_pass` (ver `docs/GATE_OWNERSHIP.md`).

## Referencias canónicas

- `docs/canonical/10_PREFLIGHT_AND_MAPPING.md`
- `docs/phases/A_INVENTORY.md`, `docs/phases/B_ROUNDTRIP.md`
- `docs/GATE_OWNERSHIP.md`

## Validación

Sin script genérico (depende del formato del juego); el criterio de éxito
es el gate `roundtrip_pass` con evidencia reproducible en `GATE_STATUS.json`.
