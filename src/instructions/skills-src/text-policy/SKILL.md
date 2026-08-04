---
name: text-policy
description: Clasifica cada unidad como TTS, SHORT_TTS_QA o KEEP_ORIGINAL, separa texto oficial de síntesis y aplica políticas distintas a líneas/cues independientes frente a FMV con ventanas y Empalme B. Úsala en fase D o al auditar texto antes de gastar GPU.
---

# text-policy

Decide qué unidades se sintetizan, cuáles se conservan originales, y separa
siempre el texto oficial del texto normalizado para síntesis. Declara también
`topology_class`: una línea independiente no necesita ser recortada de un
vídeo, mientras que una FMV sí necesita ventana, timebase y montaje.

## Triggers

- Clasificar el corpus mapeado en TTS / SHORT_TTS_QA / KEEP_ORIGINAL.
- Auditar texto ya clasificado antes de gastar GPU.
- Detectar convenciones tipográficas de subtítulo que dañan al sintetizador.

## Inputs esperados

- El mapping cerrado de la fase C.
- El texto oficial en el idioma origen y destino.

## Límites operativos

- **Regla universal `SUBTITLE_AUTHORIZED_ONLY`:** solo se sintetiza una unidad
  cuando existe una superficie textual autoritativa y verificable para ese
  juego (tarjeta de subtítulo quemada, cuadro de diálogo de la UI, VN/text box
  o equivalente confirmado). La actividad de audio, ASR/Whisper, el orden de
  un inventario o una coincidencia léxica no crean por sí solos un subtítulo.
  Si no hay esa evidencia, la unidad se clasifica `KEEP_ORIGINAL` y no bloquea
  la escena ni entra en la cola TTS. Esta regla vale para VN, in-engine, anime,
  FMV, cutscenes 3D y cualquier otro juego o motor.

- El texto oficial (`text_*`) nunca se edita; las correcciones de entrega
  van en `review_text_*`/`tts_text_*` aparte, para que un diff futuro siga
  significando algo.
- No auditar una línea aislada: si cita, corrige o responde a otra línea,
  la referencia válida es la versión LOCALIZADA de esa otra línea.
- No normalizar por forma superficial: discriminar por gramática real del
  idioma destino (un guion de elipsis legítima no es un corte de maqueta).
- Multitud/ambiente sin hablante identificable se conserva original; turnos
  de personajes con nombre se segmentan, no se degradan a KEEP_ORIGINAL.
- Una tarjeta lexical visible y autoritativa no se degrada a `KEEP_ORIGINAL`
  solo porque falten dos anclas ASR, haya una guardia `short_nonvad` o el
  borde AC29 no sea estable. Esas condiciones generan revisión temporal y
  recuperación de `speech_start/end`; la ausencia de tarjeta, lo no verbal o
  una mezcla inseparable sí justifican conservar.
- Regla operativa de este proyecto para llamadas aisladas: un nombre propio,
  Persona/skill call o grito nominal que constituye toda la unidad y es
  idéntico en origen y destino (`Palladion!`, `Aigis!`, `Koromaru!`) se marca
  `KEEP_ORIGINAL` para conservar la actuación original. Un nombre dentro de
  una oración completa sí se dobla con la oración; una llamada traducida o
  con delivery distinto requiere una decisión explícita, no una heurística.

## Procedimiento

1. Clasificar cada unidad con `line_policy` (o equivalente del proyecto):
   TTS, SHORT_TTS_QA o KEEP_ORIGINAL. Antes de permitir TTS, exigir
   `subtitle_authorized=true` (o evidencia equivalente registrada en el
   mapping); en caso contrario usar `KEEP_ORIGINAL` con causa
   `NO_VISIBLE_SUBTITLE_CARD`.
   - En `LINE_SEPARATED`, una fila/cue es una entrega completa y conserva su
     archivo independiente; no crear una máscara ni un stem global.
   - En `IN_ENGINE_TIMELINE`, conservar la ventana del evento y ajustar solo
     el cuerpo si hace falta.
   - En `EMBEDDED_FMV`, registrar `preserved_source_component`, `montage_hint`,
     `speech_start/end` e `internal_delivery_gaps_s`.
2. Auditar el texto de entrega contra artefactos reales de convención
   tipográfica (guiones de maqueta, apóstrofos sueltos, prefijos de
   hablante pegados) con un barrido corpus-completo, no una muestra.
3. Para condicionales gramaticales (género, número) que un audio estático
   no puede obedecer, exigir una redacción neutral válida para todos los
   estados o bloquear la unidad.
4. Mantener tres textos separados: `source_text` (inglés para referencia y
   detección de fugas), `target_text`/texto oficial alemán y `tts_text`
   normalizado para hablar. Si se prueba el sufijo `...`, añadirlo solo a
   `tts_text`; nunca editar el texto oficial ni hacer que `ref_text` deje de
   describir exactamente el `ref_audio`.

## Artefactos de salida

- Corpus clasificado con política por unidad.
- Reporte de barrido de artefactos de entrega.

## Gates afectados

`text_policy_reconciled`, `external_text_audit_pass`.

## Referencias canónicas

- `docs/canonical/20_TOPOLOGIES_AND_TEXT_POLICY.md`
- `docs/phases/D_TEXT_POLICY.md`
- `docs/lessons/pending/D.jsonl`

## Reglas promovidas

- `AC-59`: clasificar por entrega y separar idioma, contenido y fallback; no
  convertir una metrica de transcripcion en un bloqueo universal.
- `AC-60`: una linea solo puede liberarse con su identidad, mapa y frontera
  acustica demostrados.

## Validación

Sin script genérico (depende del corpus); el criterio es que el barrido de
artefactos corra sobre el 100% de las filas, no una muestra.
