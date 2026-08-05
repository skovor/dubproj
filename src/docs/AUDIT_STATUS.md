# Auditoría `AUDITORIA_PROFUNDA_DUBPROJ_27ca23e`

Estado inicial de la rama `refactor/p3r-pipeline-v2` antes de las correcciones de esta ejecución. `IMPLEMENTED` significa que existe código relacionado; `VERIFIED` se reservará para una prueba que cubra el camino real; `PENDING` requiere corrección o integración; `BLOCKED` requiere un asset/ejecutable externo no disponible.

| # | Hallazgo | Estado inicial | Evidencia |
|---:|---|---|---|
| 1 | Paridad matemática train/runtime | IMPLEMENTED | `dubbing_pipeline/calibration/train.py`, `qa_v2.py` |
| 2 | Calibradores target/final separados | IMPLEMENTED | `qa_v2.py`, `config` schemas |
| 3 | `segments[*].chars` de WhisperX | IMPLEMENTED | `alignment.py`, cobertura de tests parcial |
| 4 | Separación calibration/validation/hidden | IMPLEMENTED | `calibration/train.py` |
| 5 | Promoción recalculada y ligada a evidencias | PENDING | `calibration/promote.py` |
| 6 | Identidad real en promoción | PENDING | `scripts/promote_calibration_profile.py` |
| 7 | Doble revisión gold set/UI | IMPLEMENTED | `goldset.py`, `serve_goldset_review.py` |
| 8 | Gold set → features → calibradores | PENDING | no existe extractor E2E |
| 9 | Contrato de calibración LID | PENDING | `calibration/lid_features.py`, trainer |
| 10 | LID calibrado en runtime | PENDING | `lid.py`, `orchestration_v2.py` |
| 11 | Probabilidades SpeechBrain | PENDING | `alignment.py` |
| 12 | MFA CLI/implementación única | PENDING | `mfa_adapter.py`, `alignment.py` |
| 13 | Cobertura TextGrid por contenido | PENDING | `textgrid.py` |
| 14 | Reparación FMV atribuible | PENDING | `fmv_selector.py`, `orchestration_v2.py` |
| 15 | Benchmark real | PENDING | `scripts/run_real_benchmark.py` |
| 16 | Promoción de rama no falsificable | PENDING | `scripts/promote_branch.py` |
| 17 | Adapter real de segundo juego | PENDING/BLOCKED | no hay assets DQ3/P3R locales |
| 18 | Integración módulos en `run_scene_v2` | PENDING | `orchestration_v2.py` |
| 19 | Performance por línea | PENDING | `performance.py`, `performance_policy.py` |
| 20 | Executor de reparaciones y budgets | PENDING | `repair.py` |
| 21 | Cost router/model pool/state | PENDING | módulos aislados |
| 22 | Scene QA auditiva/contextual | PENDING | `scene_qa.py` |
| 23 | Manifest content-addressed | PENDING | `benchmark.py` |
| 24 | Multi-label gold set | IMPLEMENTED | `HumanLabel.labels` |
| 25 | Inmutabilidad de `add_clip` | IMPLEMENTED | `GoldsetStore.add_clip` |
| 26 | Hidden test sellado/one-shot | PENDING | `goldset.py` |
| 27 | Faltantes de features no enmascarados | PENDING | `calibration/features.py` |
| 28 | Composición mínima hidden/validation | PENDING | `calibration/promote.py` |
| 29 | Release check/full discovery | PENDING | `scripts/release_check.py` |
| 30 | Fixtures equivalentes a APIs reales | PENDING | `tests/` |
| 31 | Adapter P3R beyond inventory | PENDING/BLOCKED | runtime/game assets absent |

La tabla se actualizará solo con evidencia reproducible: comando, archivo y código de salida.
