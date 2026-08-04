---
name: generation-qa
description: Genera y valida TTS con rutas separadas para líneas de voz independientes (VN/in-engine/barks) y para FMV/anime con ventanas físicas; aplica hard gates reales, reintentos dirigidos y el contrato correcto de ASR, duración y montaje. Úsala en fases F, G, H o I.
---

# generation-qa

Genera una toma, la somete a compuertas duras independientes y solo reintenta
el fallo real. Selecciona la ruta por topología: una línea aislada no se trata
como un fragmento de película. Un score nunca compensa un hard gate fallido.

## Triggers

- Cargar el modelo y empezar a generar TTS para un lote.
- Una toma falla QA y hay que decidir si reintentar.
- Revisión perceptual de casos en zona gris (falsos positivos de MOS, etc.).

## Inputs esperados

- Corpus clasificado (`text-policy`) con `ref_audio`/`ref_text` coherentes.
- Preflight ejecutable ya en verde (entorno, GPU, referencias resueltas).

## Límites operativos

- El preflight debe rechazar cualquier unidad sin
  `subtitle_authorized=true` (o evidencia textual autoritativa equivalente).
  Esas unidades se conservan como `KEEP_ORIGINAL` y no consumen GPU, no se
  auditan como fugas de idioma y no pueden bloquear una escena. Esto es una
  regla transversal para cualquier juego/topología.

- Una toma inicial por unidad en `LINE_SEPARATED`/`IN_ENGINE_TIMELINE`; reintentos
  dirigidos solo para fallos reales, máximo cuatro intentos totales. Para
  `EMBEDDED_FMV`/anime, declarar el perfil `4 iniciales + 4 de retry` y
  ejecutarlo por rondas globales: generar las iniciales de todas las líneas,
  hacer QA y luego repetir solo los IDs sin PASS. Nunca usar best-of-8 ni
  completar cuatro tomas preventivas en un juego normal.
- Los gates duros son independientes: no vacío, idioma fuente, cola, contenido
  de entrega/palabra final, costura y límites del empalme, timing de habla de
  la costura, loudness, clipping y contrato físico. Ninguna puntuación alta
  compensa uno que falla.
- En `LINE_SEPARATED`, validar cada archivo/cue de forma independiente:
  referencia y texto de la misma línea, formato/sample rate, no vacío, cola,
  clipping, nivel activo y duración natural (VN) o ventana del evento
  (in-engine). No crear un WAV global ni cortar una película inexistente.
- En `EMBEDDED_FMV`, aplicar `duration=`, ajuste localizado del cuerpo TTS,
  alineación de onset y ganancia sobre habla activa antes de QA. `text/WER`,
  onset general, span EN↔DE, rate, pausa, F0/prosodia y LUFS residual son
  diagnósticos/ranking, no hard gates; `splice_speech_timing` sí es hard gate.
- No gastar GPU antes de que el preflight ejecutable esté en verde.
- Corregir localmente: nunca un procesado global del corpus para arreglar
  un caso de borde.
- En ventana fija, energía cerca del último frame es sólo diagnóstico. Declarar
  cola/corte abrupto únicamente con doble evidencia: llegada activa al borde y
  alineación confiable que muestre la palabra/sílaba final abreviada. Un eco,
  respiración o final plosivo completo no justifica descartar una toma.
- ASR no autoriza mapping ni convierte una coincidencia parcial en contenido
  correcto. Su WER alemán es diagnóstico; sólo una fuga inglesa clara es gate.
  Para esfuerzos/no verbales, no aplicar una compuerta léxica.
- Ejecutar ASR postgeneración de forma sistemática solo para FMV/anime
  prerenderizado. En VN, in-engine, cues/archivos independientes y battle
  barks ya mapeados, no meter Whisper en el bucle ni usarlo como score o
  desempate: el texto oficial y los gates acústicos baratos son la autoridad.
  Si una línea aislada presenta específicamente el bug conocido de última
  palabra de OmniVoice, permitir un chequeo target-only dirigido, sin usarlo
  para mapping ni para inventar texto.
- En FMV, usar cribado ASR + confirmación beam-5 solo dentro de intervalos de
  subtítulo autorizados. Marcadores fuente fuertes (`created`, `machine`,
  `nearby`, `suitable`, etc.) siguen rechazando aunque el idioma general sea
  alemán; nombres propios o ventanas cortas requieren confirmación alemana.

## Procedimiento

1. Confirmar preflight (entorno, hashes de modelo, referencias) antes de
   cargar pesos.
2. Separar las entradas por `topology_class` y conservar un registro por
   `line_id`; no mezclar el contrato de un archivo independiente con el de
   una ventana FMV.
3. Generar con `duration=` nativo cuando el motor lo soporte; evitar
   estirar/comprimir salvo que la toma caiga fuera de la ventana aceptable.
4. Aplicar primero correcciones locales permitidas de timing/onset/nivel y
   evaluar después los gates duros; guardar métricas blandas para ranking.
5. Si falla, reintentar SOLO esa unidad, variando lo que de verdad pueda
   cambiar el resultado — repetir la misma configuración contra un fallo
   determinista es puro gasto.
   La única excepción es un perfil FMV que declare selección `4 + 4`: ejecutar
   cuatro iniciales del lote completo, QA global y cuatro adicionales sólo para
   IDs sin PASS; no se extiende a VN, in-engine o barks.
6. Para frases alemanas TTS, si el perfil activa el experimento de continuidad,
   añadir `...` únicamente al `tts_text` de síntesis. No alterar el texto
   oficial, el texto inglés de referencia ni el `ref_text` de la referencia.
7. Zona gris de naturalidad → revisión perceptual con control de fuente
   calibrado, nunca un MOS absoluto sin referencia.

## Artefactos de salida

- Reporte por unidad: estado, intentos, compuertas fallidas, score.
- Casos en `_review_failed/` para revisión humana, nunca insertados en
  silencio.

## Gates afectados

`preflight_executable_pass`, `qa_hard_gates_pass`, `perceptual_review_done`.

## Referencias canónicas

- `docs/canonical/50_QA_AND_REGENERATION.md`
- `docs/phases/F_CALIBRATION.md`, `docs/phases/H_AUTOMATIC_QA.md`,
  `docs/phases/I_PERCEPTUAL_REVIEW.md`
- `docs/lessons/pending/G.jsonl`, `docs/lessons/pending/H.jsonl`

## Reglas promovidas

- `AC-57`: los reintentos dependen del defecto y de la topologia; no repetir
  una configuracion ante un fallo determinista ni conservar el peor candidato.
- `AC-59`: separar los gates de idioma, contenido y fallback por entrega.
- `AC-61`: separar sintesis GPU y QA CPU por cohortes y consolidar la cohorte
  antes de reauditarla.
- `AC-63`: las frases ultracortas requieren contexto de referencia del mismo
  actor sin cambiar el texto oficial.
- `AC-65`: el cierre reaudita archivos reales, no solo reportes previos.

## Validación

Sin script genérico (depende del motor); el criterio es que cada unidad
PASS tenga las compuertas duras satisfechas de forma independiente, no un
score agregado.
