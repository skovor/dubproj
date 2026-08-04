---
name: text-discovery
description: Localiza texto oficial de diálogo fuera del contenedor obvio (tablas paralelas por idioma, texto ausente pero con audio real) antes de declarar que no existe o de recurrir a ASR como fuente principal. Úsala en fases B y C cuando un VoiceId/cue no tiene texto asociado.
---

# text-discovery

Encuentra texto oficial que no está en el lugar obvio, antes de declarar
que falta o de recurrir a ASR como fuente principal.

## Triggers

- El contenedor de texto esperado (LOCRES, tabla principal) no existe o
  está incompleto para el idioma de destino.
- Un cue tiene `VoiceId` válido pero el campo de texto viene vacío.

## Inputs esperados

- El inventario de contenedores de texto ya localizados (de
  `inventory-roundtrip`).
- El/los `VoiceId` o `CueId` en cuestión.

## Límites operativos

- ASR es recuperación de evidencia, no fuente primaria: solo se usa cuando
  se demuestra que no hay texto oficial en ninguna tabla paralela.
- Nunca inventar texto objetivo silenciosamente; marcar para revisión humana.

## Procedimiento

1. Buscar familias de tablas localizadas PARALELAS por idioma (no solo el
   contenedor de texto principal) y verificar paridad de nombres entre
   idiomas antes de declarar ausencia.
2. Separar tablas de contenido, control de diálogo y registro de voz, aunque
   compartan zona o identificador.
3. Si de verdad no hay texto oficial pero SÍ hay audio físico, extraer solo
   esas excepciones y transcribir con ASR fuerte, dejando constancia de que
   es transcripción, no texto oficial.
4. Cruzar `VoiceId`/cue/duración antes de excluir una fila por texto vacío.

## Artefactos de salida

- Mapa `VoiceId`/cue → texto oficial (o transcripción marcada como tal).
- Lista de excepciones que siguen sin texto, para revisión humana.

## Gates afectados

`mapping_closed_or_explicit_holds`, `external_text_audit_pass`.

## Referencias canónicas

- `docs/canonical/20_TOPOLOGIES_AND_TEXT_POLICY.md`
- `docs/phases/C_MAPPING.md`

## Validación

Sin script genérico; el criterio es que cada `VoiceId` termine con texto
oficial, transcripción marcada, o `UNPROVEN_HOLD` explícito — nunca vacío
en silencio.
