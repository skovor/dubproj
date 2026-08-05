# Estado reproducible de `AUDITORIA_PROFUNDA_DUBPROJ_27ca23e`

Esta matriz se actualizó después de integrar la auditoría en `run_scene_v2`.
`VERIFIED` significa que hay una prueba reproducible en el checkout; `IMPLEMENTED`
significa que el contrato/camino está integrado pero requiere un ejecutable o
asset externo para elevar la evidencia; `BLOCKED` identifica una dependencia
externa concreta. No se usa ningún estado basado únicamente en un JSON declarado.

Evidencia común de software: `\.venv\Scripts\python.exe -m pytest -q` devuelve
`112 passed`; `scripts/release_check.py --out artifacts/release_check_local.json`
devuelve `PASS` y ejecuta pytest por descubrimiento completo, smoke, V2,
portabilidad e instrucciones. HEAD local y remoto coinciden en
`3b10f065fc22867882e4d71044ac72d1a85b4775`.

| # | Hallazgo | Estado | Evidencia específica |
|---:|---|---|---|
| 1 | Paridad matemática entrenamiento/runtime | VERIFIED | `test_v2.py::test_matching_calibration_profile_can_confirm`, `qa_v2.predict_probability` aplica `normalization` |
| 2 | Calibradores target/final separados | VERIFIED | `qa_v2.py`, `test_matching_calibration_profile_can_confirm`, artefactos separados en promoción |
| 3 | WhisperX `segments[*].chars` | VERIFIED | `test_whisperx_official_segment_chars_are_used_before_word_fallback` |
| 4 | Aislamiento calibration/validation/hidden | VERIFIED | `train_calibrator` rechaza splits no-calibration; tests de calibración |
| 5 | Promoción recalculada | VERIFIED | commit `5607925`, `tests/test_calibration_promotion.py`, filas selladas requeridas |
| 6 | Identidad real en promoción | VERIFIED | `promote_calibration_profile.py` obtiene Git/locks y no usa defaults `unknown` |
| 7 | Doble revisión gold set | VERIFIED | `goldset.py`, UI POST y tests de revisión independiente |
| 8 | Gold set → features → calibradores | VERIFIED | commit `bbca89d`, `calibration/goldset_bridge.py` y tests |
| 9 | Contrato de calibración LID | VERIFIED | `LIDFeatureRow`, `lid-fusion-v1`, trainer y tests |
| 10 | LID calibrado en runtime | VERIFIED | commit `e6cca2f`, `independent_lid`/fusión en `run_scene_v2`, tests |
| 11 | Scores SpeechBrain | VERIFIED | commit `8ebf144`, log-score se convierte a probabilidad y test |
| 12 | MFA CLI/implementación única | IMPLEMENTED | `alignment.MFAAlignerAdapter` delega a `mfa_adapter.align_diagnostic`; no hay binario MFA local para verificación real |
| 13 | Cobertura TextGrid por contenido | VERIFIED | commit `3dbd444`, pruebas de contenido distinto con misma longitud |
| 14 | Reparación FMV atribuible | VERIFIED | commit `115327b`, sustitución localizada y test de fallo atribuido |
| 15 | Benchmark real | BLOCKED | juegos solo contienen `.pak/.ucas/.utoc`; falta extracción de audio/timing y runner `run_scene_v2` real |
| 16 | Promoción de rama no falsificable | VERIFIED | `promote_branch.py --manifest` recalcula hashes, filas, commit y segundo juego |
| 17 | Adapter real de segundo juego | BLOCKED | `SecondGameAdapter` exige archivos/timing/hash/extraction `VERIFIED`; DQ3 no tiene assets extraídos ni extractor disponible |
| 18 | Integración en `run_scene_v2` | VERIFIED | reporte integra route/model pool/state/performance/LID/MFA/repairs; tests de scheduler, FMV y MFA |
| 19 | Performance por línea | VERIFIED | `classify_performance` + `policy_for` por `Line.metadata`; reporte `performance_by_line` y test |
| 20 | Executor de reparaciones/budgets | IMPLEMENTED | `apply_repair` aplica presupuesto causal, `HOLD_NO_TTS` y `BLOCKED_NO_EXECUTOR`; hook `repair_executor` integrado |
| 21 | Cost router/model pool/state | VERIFIED | `route_qa`, `ModelPool`, `StateStore` se ejecutan y quedan en el reporte |
| 22 | Scene QA auditiva/contextual | VERIFIED | gates de clipping, pico, loudness, topología y fuga contextual opcional; tests |
| 23 | Manifest content-addressed | VERIFIED | commit `80b0e26`, hashes de audios/referencias/locks y test de mutación |
| 24 | Gold set multi-label | VERIFIED | `HumanLabel.labels` y tests |
| 25 | Inmutabilidad `add_clip` | VERIFIED | constraint y tests de `GoldsetStore` |
| 26 | Hidden sellado/one-shot | VERIFIED | commit `709fbdc`, `hidden_seal` y consumo one-shot con tests |
| 27 | Faltantes de features no enmascarados | VERIFIED | `calibration/features.py` falla ante faltantes; bridge estricto y tests |
| 28 | Composición mínima hidden/validation | VERIFIED | promoción exige positivos/negativos mínimos y recalcula |
| 29 | Release/full discovery | VERIFIED | `release_check.py` ejecutó `pytest -q` (`112 passed`) además de gates existentes |
| 30 | Fixtures equivalentes a APIs reales | IMPLEMENTED | WhisperX `segments[].chars`, SpeechBrain/MFA/selector cubiertos; falta sesión de modelos/assets reales |
| 31 | Adapter P3R más allá de inventario | BLOCKED | `runtime_destinations` es seguro y `runtime_smoke` fail-closed, pero falta invocación del juego y extracción de sus contenedores |

## Bloqueo de validación de juego

Se verificaron las instalaciones de Steam en:

- `C:\Program Files (x86)\Steam\steamapps\common\P3R`
- `C:\Program Files (x86)\Steam\steamapps\common\DRAGON QUEST III HD-2D Remake`

Ambas instalaciones exponen contenedores empaquetados, no clips WAV/MP3 y
subtítulos/timing listos para el adaptador. `Get-Command UnrealPak,umodel,vgmstream`
no encontró un extractor disponible. Los WAV/MP3 encontrados bajo `.gemini` o el
mod de Reloaded son artefactos de trabajos anteriores y no tienen una cadena de
extracción/timing verificable; no se usaron como “audio real” para falsear el
benchmark.

La carga real del checkpoint descargado sí fue verificada aparte: OmniVoice local
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4` cargó en `cuda:0`/`float16` en 2.217 s;
FFmpeg local pasó el smoke de seno con código 0. Eso valida el entorno, no una
escena del juego.

## CI provenance update

The immutable CI evidence recorded in `src/artifacts/final_verification/report.json`
is GitHub Actions run `31041358093` (job `92426223541`) at checkout
`878c140a05d0184b2f86bde745556725698209c0`, with conclusion `success` and artifact
digest `sha256:4345e124c090361ee7637099db6aa88a6ffb2917c4f756493b4d97f0a7e4655e`.
That run validates the hardened core/platform workflow. The optional ML import job
is manual and remains explicitly unverified until dispatched.
