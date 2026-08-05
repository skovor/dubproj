# AuditorÃ­a profunda de cÃ³digo y lÃ³gica â€” `a5c44b9`

## Alcance

- Repositorio: `skovor/dubproj`
- Rama: `refactor/p3r-pipeline-v2`
- HEAD auditado: `a5c44b9391005a83aac84e897a06eb3f65b5258d`
- Base comparada: `27ca23eb6758846ad6f1ce026ec60c497190c01d`
- Cambios acumulados revisados: 32 commits y aproximadamente 50 archivos.
- Ãreas: calibraciÃ³n, gold set, hidden test, WhisperX, LID, MFA, QA, orquestaciÃ³n, reparaciones, performance, FMV, benchmark, segundo juego, promociÃ³n y tests.

## Veredicto

El CI verde es vÃ¡lido dentro de su alcance CPU/core, pero el cÃ³digo conserva defectos crÃ­ticos de lÃ³gica. El estado honesto es:

```text
IMPLEMENTED_WITH_CRITICAL_LOGIC_DEFECTS
AND_REAL_VALIDATION_BLOCKED
```

Los tests existentes cubren muchos contratos aislados, pero no varias rutas end-to-end donde aparecen los defectos.

---

# Hallazgos P0

## P0.1 â€” JSON Schema y runtime son incompatibles

**Archivos:**

- `src/config/calibration-profile.schema.json`
- `src/dubbing_pipeline/qa_v2.py`

El runtime exige `calibrators.target`, `calibrators.final_anchor` y `calibrators.lid`. El schema permite `language_id`, no `lid`, y prohÃ­be propiedades adicionales. Un perfil puede funcionar en runtime y fallar schema, o viceversa.

AdemÃ¡s, `finalAnchorCalibrator` hereda por `allOf` un `const: char-alignment-v2` y simultÃ¡neamente exige `const: final-anchor-v1`. El objeto es imposible de validar.

**CorrecciÃ³n:** usar exactamente los mismos roles y crear definiciones separadas para target, final anchor y LID. AÃ±adir una prueba que valide con JSON Schema un perfil real generado por el promotor.

## P0.2 â€” Solo se valida el calibrador target

**Archivos:**

- `src/dubbing_pipeline/calibration/promote.py`
- `src/scripts/promote_calibration_profile.py`
- `src/tests/test_calibration_promotion.py`

La promociÃ³n recalcula validation y hidden Ãºnicamente para target. Los artefactos de ancla final y LID entran como autorizados sin evaluar sus predicciones.

**CorrecciÃ³n:** recomputar seis reportes independientes:

```text
target_validation
target_hidden
final_anchor_validation
final_anchor_hidden
lid_validation
lid_hidden
```

Cada rol debe verificar composiciÃ³n de clases, dataset, prediction digest, Brier, ECE y falsos PASS.

## P0.3 â€” Autoridad autocertificada

**Archivos:**

- `src/dubbing_pipeline/qa_v2.py`
- `src/dubbing_pipeline/calibration/promote.py`
- `src/config/calibration-profile.schema.json`

El runtime acepta campos declarativos, artefactos con hash y locks coincidentes, pero no liga suficientemente el perfil al commit activo, los datasets reales, lÃ­mites de mÃ©tricas o una receipt verificable.

**CorrecciÃ³n mÃ­nima:**

- SHA completo de commit y coincidencia con commit autorizado;
- reabrir y rehashar manifest, labels, splits y feature datasets;
- mÃ©tricas dentro de `[0,1]`;
- lÃ­mites de promociÃ³n;
- cero falsos PASS crÃ­ticos;
- `promotion_receipt` content-addressed;
- no declarar el sistema â€œinfalsificableâ€ sin raÃ­z de confianza.

## P0.4 â€” La normalizaciÃ³n destruye contrastes del alemÃ¡n

**Archivos:**

- `src/dubbing_pipeline/qa_v2.py`
- `src/dubbing_pipeline/alignment.py`

`fold()` usa NFKD, casefold y elimina diacrÃ­ticos. Puede equiparar palabras distintas:

```text
schÃ¶n / schon
wÃ¼rde / wurde
mÃ¼sste / musste
fÃ¼r / fur
MaÃŸe / Masse
```

**CorrecciÃ³n:** NFC, preservar `Ã¤ Ã¶ Ã¼ ÃŸ`, separar normalizaciÃ³n lÃ©xica de grafemas acÃºsticos, versionar la polÃ­tica e invalidar calibradores anteriores si cambia el feature schema.

## P0.5 â€” La adjudicaciÃ³n no sustituye las etiquetas originales

**Archivos:**

- `src/dubbing_pipeline/goldset.py`
- `src/dubbing_pipeline/calibration/goldset_bridge.py`

El store guarda `consensus_labels`, pero el bridge sigue agregando las etiquetas originales. Un desacuerdo adjudicado puede seguir contaminando target, final anchor y LID.

**CorrecciÃ³n:** usar exclusivamente el consenso para clips adjudicados y conservar labels originales solo como audit trail.

## P0.6 â€” Hidden test no sellado de forma efectiva

**Archivos:**

- `src/dubbing_pipeline/goldset.py`
- `src/scripts/validate_goldset.py`
- `src/dubbing_pipeline/calibration/goldset_bridge.py`

DespuÃ©s del sello todavÃ­a pueden cambiarse o aÃ±adirse datos hidden, extraerse features repetidamente y validarse un `hidden_seal.json` solo por existir.

**CorrecciÃ³n:** sellar membresÃ­a, hashes de audio, labels efectivos y split map; bloquear mutaciones; recomputar el digest; consumir una evaluaciÃ³n one-shot; impedir una segunda apertura/promociÃ³n.

## P0.7 â€” Benchmark y promociÃ³n falsificables

**Archivos:**

- `src/dubbing_pipeline/benchmark.py`
- `src/scripts/run_real_benchmark.py`
- `src/scripts/promote_branch.py`
- `src/adapters/second_game_template/adapter.py`

`--runner` puede ser cualquier callable que devuelva `PASS`. `real_audio` y la evidencia del segundo juego son en gran parte declarativos. `promote_branch.py` confÃ­a en un JSON externo sin ejecutar de nuevo el adapter.

**CorrecciÃ³n:** runner aprobado e identificable; reportes y outputs reabiertos y rehasheados; `real_audio` derivado de provenance; ejecutar el adapter real durante promociÃ³n; bloquear mocks y JSON fabricado.

---

# Hallazgos P1

## P1.1 â€” Bridge y entrenamiento incompatibles con tres splits

El bridge escribe calibration, validation y hidden en cada JSONL; los scripts cargan todo; `train_calibrator()` rechaza filas que no sean calibration.

**CorrecciÃ³n:** archivos separados por rol/split o filtrado explÃ­cito y auditable. Validation y hidden nunca deben entrar al entrenamiento.

## P1.2 â€” SemÃ¡ntica incorrecta del vector LID

`whisper_source_probability` recibe la confianza de Whisper aunque el idioma detectado sea target. TambiÃ©n se usa un score CTC crudo como si fuera probabilidad.

**CorrecciÃ³n:** condicionar probabilidades al idioma, separar raw score de calibrated probability y congelar un contrato compartido por bridge, trainer y runtime.

## P1.3 â€” Routing no escala sospechas de fuga

`route_qa()` no envÃ­a a CTC+LID estados como:

```text
LANGUAGE_LEAK_SUSPECTED
LANGUAGE_LEAK_STRONG_SUSPICION
EVIDENCE_CONFLICT
```

**CorrecciÃ³n:** tabla explÃ­cita de escalamiento. `CTC_LID_SECOND` debe gobernar una rama real.

## P1.4 â€” Sospecha convertida en leak confirmado

El orquestador convierte sospechas/conflictos en `FailureCause.LANGUAGE_LEAK_CONFIRMED`, habilitando reparaciones TTS.

**CorrecciÃ³n:** sospecha y conflicto deben quedar HOLD sin TTS. Solo fuga confirmada puede regenerar.

## P1.5 â€” Reparaciones no reingresan al pipeline

La salida del repair executor se registra, pero no se persiste/reabre/re-audita ni compite en selecciÃ³n.

**CorrecciÃ³n:** ciclo acotado `execute â†’ persist â†’ reopen â†’ QA â†’ mounted/serialized QA â†’ selection`.

## P1.6 â€” Servidor gold set inseguro con threads

`ThreadingHTTPServer` comparte una conexiÃ³n SQLite creada fuera de los handlers. TambiÃ©n faltan autenticaciÃ³n y separaciÃ³n de hidden.

**CorrecciÃ³n:** conexiÃ³n/store por request o arquitectura segura equivalente; claims atÃ³micos; autorizaciÃ³n de reviewers/adjudicators; hidden no visible en GET normal.

## P1.7 â€” Performance modes conectados superficialmente

El default NEUTRAL se inyecta como metadata explÃ­cita; no se pasan RMS/pitch/speech ratio; `max_duration_error_ms` y `require_pitch_identity` no se aplican; EFFORT elimina gates lÃ©xicos pero el QA global sigue exigiendo PASS lingÃ¼Ã­stico.

**CorrecciÃ³n:** modo explÃ­cito vs no resuelto, features acÃºsticas reales, polÃ­ticas ejecutables, encoding categÃ³rico y estado final vÃ¡lido para esfuerzo no verbal.

## P1.8 â€” FMV sin atribuciÃ³n real de fallos

El selector entiende `failed_line_ids`, pero Scene QA no los produce. Sin culpable, cambia una lÃ­nea arbitraria. Los cursores monotÃ³nicos pueden perder combinaciones vÃ¡lidas.

**CorrecciÃ³n:** gates por ventana, `failed_line_ids`, bÃºsqueda local acotada con backtracking/beam y caso adversarial `A1+B1`, `A1+B2` fallan, `A2+B1` pasa.

## P1.9 â€” Scene QA global y dÃ©bil

Mide RMS/clipping global del stem. No detecta por lÃ­nea voz ausente, corte final, seam, loudness o contextual leak. MÃºsica de fondo puede ocultar una lÃ­nea silenciosa.

**CorrecciÃ³n:** gates por ventana y agregaciÃ³n de escena.

## P1.10 â€” API legacy permite hard PASS con strings del caller

La ruta `transcript/language/probability` sin evidence records puede devolver PASS.

**CorrecciÃ³n:** diagnostic-only; hard PASS solo con evidencia estructurada en modo estricto.

---

# Hallazgos P2

## P2.1 â€” Desempate sesgado en sequence alignment

SUBSTITUTE cuesta lo mismo que DELETE+INSERT, pero el desempate prioriza gaps y distorsiona ratios.

## P2.2 â€” ApÃ³strofe como carÃ¡cter acÃºstico obligatorio

WhisperX puede no asignarle timing, generando falsos DELETE o anchors incompletos.

## P2.3 â€” ExpansiÃ³n multicaracter de `ÃŸ`

La normalizaciÃ³n puede romper la correspondencia entre grafemas esperados y chars observados.

---

# Aspectos correctamente mejorados

- Calibradores JSON seguros ejecutables.
- SHA de artefactos.
- SeparaciÃ³n target/final/LID.
- ExtracciÃ³n de chars nativos WhisperX.
- ConversiÃ³n conservadora de score SpeechBrain.
- Estados inciertos fail-closed.
- Varias incertidumbres sin retry inicial.
- SelecciÃ³n FMV local sin producto cartesiano completo.
- CI core multiplataforma real.
- Contratos post-transform y serializaciÃ³n mejorados.

---

# Plan correctivo recomendado

1. `Align calibration schema and runtime roles`
2. `Validate target anchor and LID calibrators independently`
3. `Bind calibration authority to reproducible promotion evidence`
4. `Preserve German contrasts in lexical and character QA`
5. `Make adjudicated goldset labels authoritative`
6. `Seal hidden evaluation and isolate review transactions`
7. `Make goldset calibration training split safe`
8. `Unify LID feature semantics and escalation routing`
9. `Hold uncertain repairs and re-audit repaired outputs`
10. `Execute performance policies end to end`
11. `Attribute FMV scene failures and strengthen window QA`
12. `Bind benchmark promotion to trusted execution`
13. `Add adversarial end to end regression coverage`

---

# Estado por Ã¡rea

| Ãrea | Estado |
|---|---|
| CI core | Verde |
| Schema/perfil runtime | Roto |
| Calibrador target | Parcialmente validado |
| Calibrador ancla final | No validado |
| Calibrador LID | No validado |
| NormalizaciÃ³n alemana | No segura |
| AdjudicaciÃ³n gold set | Incorrecta para entrenamiento |
| Hidden test | No sellado efectivamente |
| Bridge â†’ entrenamiento | Incompatible con tres splits |
| Routing LID | Incompleto |
| Reparaciones | No funcionales end-to-end |
| Performance modes | Parciales/inconsistentes |
| FMV atribuible | Solo con auditor artificial |
| Scene QA | Parcial |
| Benchmark no falsificable | No |
| Segundo juego real | No validado |
| Audio real P3R/DQ3 | Bloqueado |
| ProducciÃ³n | No apta |

## ConclusiÃ³n

Antes de afirmar que el Ãºnico bloqueo restante es extraer P3R o DQ3, deben corregirse estos defectos de cÃ³digo y lÃ³gica. La ausencia de audio real impide validar calidad acÃºstica, pero no explica ni justifica las incompatibilidades internas encontradas.

