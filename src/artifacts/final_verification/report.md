# Verificación final de Dubproj

**Estado:** `IMPLEMENTED_BUT_REAL_VALIDATION_BLOCKED`

Código auditado: `0cca1f257cd115c24b4f84eeb1fe2002f1a23485` en
`refactor/p3r-pipeline-v2`. La matriz completa está en
[`src/docs/AUDIT_STATUS.md`](../../docs/AUDIT_STATUS.md) y la versión máquina está
en `report.json`.

## Resultado reproducible

- `.venv\Scripts\python.exe -m pytest -q` → código `0`, `112 passed in 4.45s`.
- `scripts/release_check.py --out artifacts/release_check_local.json` → código `0`,
  `PASS`; ejecuta compileall, pytest por descubrimiento completo, smoke, V2,
  portabilidad e instrucciones.
- `pip check` → `No broken requirements found`.
- OmniVoice `0.2.1`, revisión `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`:
  carga real en `cuda:0`/`float16` en `2.217 s`.
- FFmpeg `9.0-essentials_build-www.gyan.dev`: smoke de seno, código `0`.

## Evidencia CI inmutable

El run independiente de GitHub Actions que respalda el checkout anterior es:

| Campo | Valor |
|---|---|
| Workflow | `.github/workflows/ci.yml` |
| Workflow commit observado | `dc5cdeb727290426e1f1f180710b4d4478a0372b` |
| Checkout probado | `9b8822478c21337478c49533ef3a2963f0d518a3` |
| Run / job | `31039137909` / `92418905970` |
| Estado | `completed / success` |
| Runner / Python | `ubuntu-24.04` / `3.12.13` |
| Pytest | `112 passed` |
| Artefacto / digest | `8943859065` / `sha256:127b3559a68fd19318dcf344ee42e8979f0dc9cba81520e98aae766fe0e33d49` |
| Observado | `2026-08-05T19:24:17Z` |

Las identidades tienen funciones distintas: `validated_code_commit` es el último
commit que modificó el runtime auditado; `workflow_commit` identifica la definición
CI usada por la ejecución registrada; `evidence_head_sha` es el checkout exacto que
Actions probó. No se utiliza un booleano mutable como `observed_github_run`: una
ejecución futura siempre debe registrarse como una nueva evidencia.

El workflow endurecido en el commit actual todavía necesita su propia ejecución.
El run anterior no se reutiliza retroactivamente para afirmar que probó esos cambios.

## Alcance real del CI

- `CI_CORE_CONTRACTS = VERIFIED`: instalación core, `pip check`, descubrimiento
  completo, release gates y smoke.
- `CI_OPTIONAL_ML_BACKENDS = NOT_RUN`: WhisperX, SpeechBrain y faster-whisper se
  prueban mediante un job CPU manual (`workflow_dispatch`), separado del CI normal.
- `LOCAL_GPU_MODEL_LOAD = VERIFIED`: carga local de OmniVoice en la GPU documentada.
- `REAL_GAME_AUDIO = BLOCKED`: no se inventa benchmark de juego.

El workflow actual fija las Actions por SHA, usa `src/constraints-ci.txt`, captura
`pip freeze` y exige los artefactos con `if-no-files-found: error`. La matriz de
smoke cubre Ubuntu/Windows y Python 3.10/3.12. `release_check.py --skip-pytest`
evita ejecutar pytest dos veces en el job completo; el pytest completo permanece
como paso explícito del job.

## Cambios integrados

Se hicieron commits atómicos para calibración target/final/LID, parsing real de
`segments[].chars`, gold set sellado y puente de features, SpeechBrain, TextGrid,
MFA, routing por línea, pool/estado, reparaciones bounded, selección FMV atribuible,
Scene QA acústica, manifests con hashes, promoción no falsificable, full discovery y
seguridad del adapter P3R. Todos los SHA y mensajes están en `report.json`.

`run_scene_v2` invoca y registra `route_qa`, `ModelPool`, `StateStore`,
`classify_performance`/`policy_for`, LID/fusión, MFA diagnóstico y
`plan_repairs`/`apply_repair`. `ASR_UNCERTAIN` queda en `HOLD_NO_TTS`; la ausencia
de un ejecutor de audio queda en `BLOCKED_NO_EXECUTOR`, nunca en una regeneración
arbitraria.

## Por qué no se declara validación de juego

Se intentó primero Persona 3 Reload y después Dragon Quest III HD-2D Remake. Las
instalaciones existen, pero su contenido está empaquetado en `.pak/.ucas/.utoc`
(P3R) o `.pak` (DQ3). No hay clips de audio con subtítulos/timing extraídos ni
`UnrealPak`, `umodel` o `vgmstream` disponible. Por ello no es posible demostrar
20 líneas variadas, FMV multilínea, montaje, reapertura, Scene QA y benchmark real
sin inventar evidencia.

El benchmark queda `BLOCKED` por diseño: `scripts/run_real_benchmark.py` exige un
runner `module:function` que invoque `run_scene_v2` y un manifiesto content-addressed;
`scripts/promote_branch.py` vuelve a calcular esas evidencias.

## Limitaciones restantes

1. Hace falta un extractor autorizado para los contenedores del juego y un adapter
   que produzca clips, subtítulos/timing y hashes.
2. Hace falta ejecutar ese adapter sobre al menos 20 líneas P3R o DQ3 y una escena
   FMV, incluyendo los casos adversos indicados por la auditoría.
3. El ejecutor técnico de reparaciones es una dependencia explícita
   (`repair_executor`) y queda bloqueado si no se proporciona.

No se marca `VERIFIED_ON_P3R` ni `VERIFIED_ON_DQ3_HD2D` porque esa evidencia externa
no está disponible en este entorno.
