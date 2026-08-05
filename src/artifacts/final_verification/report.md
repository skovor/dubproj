# Verificación final de Dubproj

**Estado:** `IMPLEMENTED_BUT_REAL_VALIDATION_BLOCKED`

Código auditado: `0cca1f257cd115c24b4f84eeb1fe2002f1a23485` en
`refactor/p3r-pipeline-v2`. La matriz completa de hallazgos está en
[`docs/AUDIT_STATUS.md`](../../docs/AUDIT_STATUS.md) y la versión máquina está en
`report.json`.

## Resultado reproducible

- `.venv\Scripts\python.exe -m pytest -q` → código `0`, `112 passed in 4.45s`.
- `scripts/release_check.py --out artifacts/release_check_local.json` → código `0`,
  `PASS`; incluyó `compileall`, pytest por descubrimiento completo, smoke, V2,
  portabilidad e instrucciones.
- `pip check` → `No broken requirements found`.
- OmniVoice `0.2.1`, revisión `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`:
  carga real en `cuda:0`/`float16` en `2.217 s`.
- FFmpeg `9.0-essentials_build-www.gyan.dev`: smoke de seno, código `0`.

El workflow `.github/workflows/ci.yml` quedó versionado y publicado en
`dc5cdeb727290426e1f1f180710b4d4478a0372b`. Ejecuta la suite en GitHub-hosted
Ubuntu con Python 3.12, pero todavía no se cuenta una ejecución observada de
Actions; por eso este informe no convierte las pruebas locales en evidencia CI.

## Cambios integrados

Se hicieron commits atómicos para calibración target/final/LID, parsing real de
`segments[].chars`, gold set sellado y puente de features, SpeechBrain, TextGrid,
MFA, routing por línea, pool/estado, reparaciones bounded, selección FMV
atribuible, Scene QA acústica, manifests con hashes, promoción no falsificable,
full discovery y seguridad del adapter P3R. Todos los SHA y mensajes están en
`report.json`.

`run_scene_v2` invoca y registra `route_qa`, `ModelPool`, `StateStore`,
`classify_performance`/`policy_for`, LID/fusión, MFA diagnóstico y
`plan_repairs`/`apply_repair`. `ASR_UNCERTAIN` queda en `HOLD_NO_TTS`; la ausencia
de un ejecutor de audio queda en `BLOCKED_NO_EXECUTOR`, nunca en un falso plan ni
en una regeneración arbitraria.

## Por qué no se declara validación de juego

Se intentó primero Persona 3 Reload y después Dragon Quest III HD-2D Remake. Las
instalaciones existen, pero su contenido está empaquetado en `.pak/.ucas/.utoc`
(P3R) o `.pak` (DQ3). No hay clips de audio con subtítulos/timing extraídos ni
`UnrealPak`, `umodel` o `vgmstream` disponible. Por ello no es posible demostrar
20 líneas variadas, FMV multilínea, montaje, reapertura, Scene QA y benchmark
real sin inventar evidencia. Los archivos bajo `.gemini` y el mod de Reloaded se
trataron como artefactos previos no verificables, no como corpus real.

El benchmark queda `BLOCKED` por diseño: `scripts/run_real_benchmark.py` exige
un runner `module:function` que invoque `run_scene_v2` y un manifiesto
content-addressed; `scripts/promote_branch.py` vuelve a calcular esas evidencias.

## Limitaciones restantes

1. Hace falta un extractor autorizado para los contenedores del juego y un
   adapter que produzca clips, subtítulos/timing y hashes.
2. Hace falta ejecutar ese adapter sobre al menos 20 líneas P3R o DQ3 y una
   escena FMV, incluyendo los casos adversos indicados por la auditoría.
3. El ejecutor técnico de reparaciones es una dependencia explícita (`repair_executor`)
   y queda bloqueado si no se proporciona.

No se marca `VERIFIED_ON_P3R` ni `VERIFIED_ON_DQ3_HD2D` porque esa evidencia
externa no está disponible en este entorno.
