---
name: dubbing-router
description: Decide la ruta de entrada al pipeline y la topología física (líneas/archivos independientes, timeline in-engine o FMV con audio embebido) cuando no está claro en qué fase A-K/REPORT cae la tarea, cuando cruza varias fases conocidas, o cuando hay incertidumbre técnica sobre el juego/motor. Úsala antes de invocar build_session_packet.py, no como sustituto de él.
---

# dubbing-router

Decide por dónde entrar al pipeline y clasifica primero la topología física:
una fase A–K/REPORT conocida, el conjunto mínimo si la tarea cruza fases,
`X_DIAGNOSIS` ante incertidumbre técnica, o `ALL` solo para una auditoría
transversal excepcional.

## Triggers

- Empezar una tarea nueva y no saber en qué fase cae.
- Una tarea que menciona dos o más fases conocidas.
- Un juego, motor o formato nunca visto en este proyecto.
- No está claro si cada voz ya vive en un archivo/cue independiente o si está
  embebida en una película.

## Inputs esperados

- El objetivo de la tarea, en una frase.
- `TASK_BRIEF.json` si existe (campos `phase`, `engine`, `middleware`,
  `problem_classes`); nunca se infiere de texto libre.
- Inventario físico: archivos de voz/cues separados, timeline in-engine o
  contenedor de vídeo con audio multiplexado.

## Límites operativos

- No generar, empaquetar ni desplegar desde esta skill: solo decide la ruta.
- No usar `ALL` como respuesta por defecto a la duda.
- No elegir `fmv-audio` por el nombre “cutscene” o por encontrar un MP4 de
  referencia: inspeccionar dónde vive el audio que usa el runtime.

## Procedimiento

1. Clasificar `topology_class` antes de cargar el modelo:
   - `LINE_SEPARATED`: un WAV/OGG/banco/cue por línea o unidad; no hay que
     cortar una película ni reconstruir una mezcla global.
   - `IN_ENGINE_TIMELINE`: archivos/cues separados, pero una timeline fija
     controla onset/final; usar generación/QA normal con contrato de ventana.
   - `EMBEDDED_FMV`: el vídeo (USM/WebM/BIK o equivalente) contiene el audio;
     cada línea requiere tarjeta, timebase y montaje dentro del stem.
   - `UNKNOWN`: enviar a `X_DIAGNOSIS`; no generar para adivinar.
2. ¿La tarea nombra una fase A–K/REPORT conocida? → usar esa fase.
3. ¿Cruza dos o más fases conocidas? → construir el conjunto mínimo de esas
   fases (varias invocaciones de `build_session_packet.py`), o un plan
   secuencial. No usar `ALL`.
4. ¿Hay incertidumbre técnica real (motor/formato/topología desconocidos)?
   → `--phase X_DIAGNOSIS`.
5. ¿Es una auditoría transversal excepcional, un cambio de canon que afecta
   a muchas fases, o una revisión final de contradicciones? → `ALL`.

## Artefactos de salida

Ninguno propio: el resultado es la elección de `topology_class` y fase para el
siguiente comando (`build_session_packet.py --phase <FASE> --agent <A>`).

## Gates afectados

Ninguno directamente; una elección de fase equivocada retrasa el cierre de
gates de esa fase, no los bloquea.

## Referencias canónicas

- `docs/canonical/X_DIAGNOSIS.md`
- `AGENTS.md` / `CLAUDE.md` (sección de arranque obligatorio)
- `scripts/context_router.py`

## Reglas promovidas

- `AC-57`: enrutar los reintentos segun el defecto y la topologia, con limite
  y politica fail-closed.
- `AC-58`: FMV se procesa por rondas globales; no cerrar una escena aislada
  mientras falten unidades de la pelicula.
- `AC-61`: declarar por adelantado el recurso y la cohorte para no mezclar
  sintesis GPU con QA CPU ni duplicar trabajo.

## Validación

```bash
python scripts/context_router.py --phase <FASE> --agent <claude|codex> --json
```
