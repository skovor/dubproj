---
name: dialogue-mapping
description: Resuelve la cadena texto→trigger→cue→stream físico→ruta runtime con control positivo, distingue líneas/archivos independientes de ventanas FMV y clasifica cada unidad como generable, conservable o hold antes de gastar GPU. Úsala en fase C o ante un cue ID/ventana que no corresponde de forma obvia a un stream físico.
---

# dialogue-mapping

Resuelve la cadena completa texto → trigger → cue → stream físico → path
runtime, con control positivo, antes de que ninguna línea entre a generación.
El resultado debe declarar la topología: `LINE_SEPARATED`,
`IN_ENGINE_TIMELINE` o `EMBEDDED_FMV`.

## Triggers

- Empezar el mapping de una fase/zona nueva (fase C).
- Un cue ID no corresponde de forma obvia a un índice físico del banco.
- Líneas que parecen huérfanas al inspeccionar solo un script/controlador.

## Inputs esperados

- El texto localizado ya descubierto (`text-discovery`).
- El inventario físico de streams (`inventory-roundtrip`).

## Límites operativos

- Una unidad solo es candidata a doblaje si su texto aparece en una superficie
  textual autoritativa y verificable del juego. Audio activo, ASR/Whisper, OCR
  aislado, una tabla maestra o una coincidencia léxica no sustituyen esa
  prueba. Sin subtítulo/cuadro de diálogo confirmado, registrar
  `NO_VISIBLE_SUBTITLE_CARD` + `KEEP_ORIGINAL`; nunca crear un bloqueo TTS.

- Nunca tratar `CueId`, `CueIndex`, `StreamAwbId` o número de subsong como
  intercambiables porque coincidan en las primeras filas.
- No declarar una línea alcanzable solo porque existe en una tabla maestra:
  exige una ruta demostrable desde controlador, script, o LevelSequence.
- Control positivo obligatorio: antes de concluir ausencia, demostrar que
  el método de búsqueda SÍ encuentra los casos conocidos.
- En FMV con mezcla embebida, correlación de audio identifica la película,
  no una línea: nunca deriva `line_id`, texto ni hablante desde energía, ASR,
  orden de tablas o coincidencia léxica aislada.
- En `LINE_SEPARATED`, el archivo/cue físico ya es la unidad de entrega: no
  inventar una ventana global de película ni exigir dos anclas ASR para una
  línea que ya tiene `VoiceId` y texto oficial.
- En FMV, una tarjeta visual humana con identidad/timebase verificados es
  autoridad suficiente para crear la unidad. Dos anclas ASR son evidencia de
  apoyo, no un hard gate: si faltan, recalcular bordes con la tarjeta y VAD,
  marcar revisión temporal o usar `speech_start/end`; no convertir una línea
  lexical visible en `KEEP_ORIGINAL` automáticamente.
- No mezclar evidencia histórica con la decisión vigente: si
  `mapping_validation_reason` dice “visible subtitle” pero
  `force_keep_reason` dice `not_in_visual_subtitle_authority`, marcar una
  contradicción de contrato y resolverla en el mapa antes de generar o
  presentar `PASS`. Un HTML debe distinguir `PASS_TTS` de
  `PASS_KEEP_ORIGINAL`.

## Procedimiento

1. Resolver ID lógico → tabla de sonido → CueId → CueTable/ReferenceIndex →
   WaveformTable/StreamAwbId → posición física, exigiendo cobertura
   biyectiva o alias documentados.
2. Si un script llama una secuencia por índice sin nombrar la línea,
   escanear los assets de LevelSequence/CutScene antes de declarar la línea
   huérfana.
3. Para páginas no citadas literalmente, comparar el ID normalizado contra
   funciones de evento y confirmar que algún actor/mapa/helper de UI las
   consume.
4. Clasificar cada unidad: `CONTROLLER_REACHABLE`, `REGISTRY_ONLY` o
   `BANK_ONLY`. Solo la primera entra al corpus generable por defecto.
5. Para una tarjeta de FMV, exigir además la terna `movie_identity_verified`,
   `card_identity_verified` y `card_timebase_verified`. La prueba visual debe
   mostrar la misma película y el subtítulo oficial; VTT/captions, OCR aislado
   o gameplay con UI son locators. Sin esa terna, `UNPROVEN_HOLD`.
6. Clasificar las excepciones después de la evidencia, no antes: ausencia de
   tarjeta → `NO_VISIBLE_SUBTITLE_CARD`/`KEEP_ORIGINAL`; no verbal →
   `KEEP_ORIGINAL`; superposición inseparable → hold humano; tarjeta lexical
   visible con anchors ASR insuficientes → generable con timing revisable.

## Artefactos de salida

- Mapa completo texto↔evento↔audio con estado de alcanzabilidad por unidad.
- Lista de colisiones de ownership (`ownership_collisions_zero`).

## Gates afectados

`mapping_closed_or_explicit_holds`, `ownership_collisions_zero`,
`mapping_verified_against_audio`.

## Referencias canónicas

- `docs/canonical/10_PREFLIGHT_AND_MAPPING.md`
- `docs/phases/C_MAPPING.md`
- `docs/lessons/pending/C.jsonl` (hallazgos pendientes específicos de mapping)

## Reglas promovidas

- `AC-60`: la frontera de una entrega se valida contra mapa, evidencia
  acustica y contenido; una ventana de subtitulo por si sola no basta.
- `AC-63`: para frases ultracortas, usar contexto del mismo actor sin cambiar
  la asociacion linea-evento.

## Validación

Sin script genérico (depende del motor); el criterio es cobertura
biyectiva demostrada, no un conteo de filas.
