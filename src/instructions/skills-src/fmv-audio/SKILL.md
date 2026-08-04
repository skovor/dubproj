---
name: fmv-audio
description: Dobla diálogo dentro de video prerenderizado (FMV) con audio multiplexado, respetando ventanas y timebase físicos, Empalme B, detección de fugas y conservación del bed/ambiente central. Úsala cuando el audio vive dentro de un .usm/.webm/.bik o equivalente, en fases E, G o H; no para líneas/cues independientes.
---

# fmv-audio

Dobla diálogo dentro de video prerenderizado sin destruir el bed
central/ambiente, respetando una ventana de tiempo físicamente fija. Esta ruta
solo aplica cuando el audio está multiplexado en el vídeo; si cada personaje
ya tiene un WAV/OGG/cue separado, usar `generation-qa` con `LINE_SEPARATED` o
`IN_ENGINE_TIMELINE`, no esta skill.

## Triggers

- El activo es un `.usm`/`.webm`/`.bik` (o equivalente) con audio
  multiplexado, no un stream de evento independiente.
- Hay que decidir timing/atempo para una línea dentro de una FMV.

## Inputs esperados

- El plan de entrega FMV: texto oficial, ventanas, `internal_delivery_gaps_s`
  y allowlist visual con identidad de película/tarjeta/timebase verificados.
- La fuente de video en alta calidad (no un recomprimido de longplay, si hay
  alternativa).
- La terna por unidad `movie_identity_verified`, `card_identity_verified` y
  `card_timebase_verified`, más `source_start/end`, `speech_start/end` cuando
  existan y la política de `subtitle_authorized`.

## Límites operativos

- **Política universal de contenido:** en FMV/prerenderizado solo se dobla una
  línea con tarjeta de subtítulo visible y autoritativa, identidad de tarjeta y
  timebase verificadas. Voces de fondo, multitud, anuncios o ASR sin tarjeta se
  conservan originales (`KEEP_ORIGINAL`) y no bloquean el release. La misma
  decisión debe aplicarse a cualquier juego y topología, no solo a P3R.

- Nunca borrar el bed central/ambiente: se estima y se conserva, la voz se
  monta encima con crossfade corto.
- El rescate de duración se aplica SOLO al cuerpo TTS activo, nunca al
  esfuerzo original, el fondo, ni el contenedor completo.
- `start/end` describe la ventana de montaje; `speech_start/speech_end`
  describe el habla activa usada para alinear y auditar. No usar toda la tarjeta
  como intervalo de idioma si después empieza una fila `KEEP_ORIGINAL`.
- Un video mudo (sin stream de audio) no es FMV doblable solo por tener
  diálogo en el nombre del archivo.
- Multihablante en una misma unidad se segmenta por frontera de turno con
  referencia de voz propia por mitad; no se resuelve como `KEEP_ORIGINAL`
  si ya existe la maquinaria de segmentación.
- `force_clone` nunca salta el allowlist visual. Correlación/fingerprint de
  audio sólo sitúa una escena; VTT, captions automáticos, OCR aislado o UI de
  gameplay no prueban texto, `line_id` ni ventana de producción.
- ASR/VAD sólo mide actividad dentro de una ventana ya aceptada. Una
  contradicción ASR de alta confianza bloquea; reconocimiento parcial o ruido
  no crea ni invalida por sí solo un mapping.
- Una tarjeta visual `HUMAN_CONFIRMED` con timebase verificable autoriza la
  unidad aunque falten dos anclas ASR coincidentes. `fewer_than_two_agreeing_asr_anchors`,
  `short_nonvad_hallucination_guard` o un borde AC29 inestable son señales para
  recalcular onset/offset o revisar el audio, no una razón automática para
  dejar una línea lexical subtitulada en inglés. Solo la ausencia de tarjeta,
  una vocalización no lingüística o una superposición inseparable justifican
  `KEEP_ORIGINAL` sin intentar un cuerpo alemán.
- El fallo de una entrega no autoriza seleccionar la "menos mala" ni sustituir
  el contenedor completo por inglés. Tras agotar intentos, una unidad sin
  subtítulo queda `KEEP_ORIGINAL`; una unidad subtitulada sin candidato seguro
  queda en revisión/máscara explícita y la película no se declara liberable.
  Las demás entregas PASS sí pueden montarse y conservar sus hashes.

## Procedimiento

1. Confirmar que el contenedor de video SÍ tiene stream de audio antes de
   clasificarlo como doblable.
2. Verificar que la fuente externa muestra la misma película, sin UI ajena,
   y confirmar cada tarjeta desde píxeles contra el texto oficial. Alinear por
   audio sólo después para convertir esa tarjeta a la timebase del activo.
   Deduplicar sus observaciones por continuidad temporal, no por texto igual.
   Sin tarjeta verificable, conservar original/`UNPROVEN_HOLD`.
   No generar ni reintentar una línea que no tenga esa tarjeta: marcarla
   `KEEP_ORIGINAL` con `NO_VISIBLE_SUBTITLE_CARD`.
3. Estimar el bed central desde los canales laterales, antes de saber las
   ventanas de diálogo (para no aprender la voz original como si fuera
   ambiente).
4. Para Empalme B, registrar `preserved_source_intervals`, `source_resume` y
   `effort_end`. Conservar `Ugh/Geez/ahh` bit-exacto y sintetizar solo el
   cuerpo alemán; nunca pasar la onomatopeya preservada como `ref_audio` con
   un `ref_text` alemán que no describe el mismo contenido.
5. Sintetizar con el texto alemán (y, si el perfil lo activa, añadir `...`
   únicamente al texto de síntesis), aplicar el contrato de longitud solo al
   cuerpo activo, montar con crossfade corto y devolver el bed en un valle
   seguro. Nunca cortar una palabra activa para hacer caber la toma.
6. Auditar el stem montado en ventanas que se enmascaren a los intervalos
   autorizados. Usar cribado de idioma + confirmación alemana; mantener un
   rechazo fuerte para palabras fuente inglesas como `created`, `machine`,
   `nearby`, `suitable` o `vessel`.
7. Remuxear y verificar muestras, vídeo, layout, canales no-dialogue y prueba
   runtime. Si una línea subtitulada no tiene candidato seguro, mantener la
   escena no liberable y registrar revisión; no reintroducir inglés
   silenciosamente.

## Artefactos de salida

- Reporte de release por película (`release_ok`, unidades PASS/FAIL,
  hashes de fuente y salida), con evidencia de identidad y timebase por
  unidad.
- WebM/USM remuxeado, solo si el 100% de las unidades tiene una disposición
  segura: `PASS` alemán o `KEEP_ORIGINAL` justificado. Ninguna toma TTS que
  falle hard gates llega al remux final.

## Gates afectados

`timebase_verified`, `qa_hard_gates_pass`, `package_roundtrip_pass`.

## Referencias canónicas

- `docs/canonical/40_AUDIO_TIMING_AND_FMV.md`
- `docs/phases/E_MASTER_PLAN.md`, `docs/phases/G_GENERATION_AND_MONTAGE.md`
- `docs/lessons/pending/G.jsonl`

## Reglas promovidas

- `AC-58`: las rondas FMV son globales y una escucha aislada no altera la
  produccion sin evidencia reproducible.
- `AC-59`: separar idioma, contenido y fallback por entrega; una fuga inglesa
  sigue siendo un fallo aunque la forma alemana pase.
- `AC-60`: fijar cada frontera con evidencia acustica, linguistica y de mapa;
  no aceptar recortes basados solo en el texto.
- `AC-62`: comprobar headroom, clipping y serializacion como controles
  distintos antes de declarar un stem liberable.
- `AC-65`: reauditar los artefactos reales despues de reparar, serializar y
  remuxear.

## Validación

Sin script genérico (depende del proyecto); el criterio es release_ok=true
solo cuando el 100% de las unidades declaradas pasó QA con el contrato
vigente.
