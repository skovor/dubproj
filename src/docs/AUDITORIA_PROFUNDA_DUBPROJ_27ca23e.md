# Auditoría profunda de `refactor/p3r-pipeline-v2`

**Repositorio:** `skovor/dubproj`  
**Commit auditado:** `27ca23eb6758846ad6f1ce026ec60c497190c01d`  
**Base comparada:** `f9da27c5e18d76f71a38fc0863efa487f7b255b1`  
**Resultado general:** la rama es mayormente *fail-closed*, pero no está funcionalmente completa ni lista para calibración real, benchmark real o promoción.

## Dictamen ejecutivo

- La cadena contiene 11 commits desde la base, incluido `2c1c52b` antes de los diez commits enumerados.
- Los módulos nuevos existen y muchas pruebas unitarias pasan, pero buena parte de ellos no está conectada al camino real de `run_scene_v2`.
- Hay errores confirmados en calibración, LID, gold set, MFA, selector FMV, benchmark y promoción.
- `release_check` no ejecuta la suite completa: solo `test_v2.py`, smoke, compilación, portabilidad e instrucciones.
- No debe hacerse merge ni habilitar `calibration_authority` hasta corregir los P0 y completar una validación end-to-end con audio real.

# P0 — Bloqueadores críticos

## 1. Train-serving skew en el calibrador

El entrenamiento normaliza las features usando medias y desviaciones y aprende coeficientes sobre esas features normalizadas. El runtime carga el JSON pero descarta `normalization` y aplica los coeficientes directamente a features crudas. Las probabilidades de producción no corresponden al modelo entrenado.

**Corrección:** el loader debe validar y conservar la normalización, y `predict_probability` debe aplicar exactamente `(x - mean) / scale` antes del logit. Añadir prueba golden entre entrenamiento y runtime.

## 2. Un calibrador se usa para dos tareas incompatibles

El entrenamiento genera dos artefactos: target y final-anchor. El perfil y el runtime solo admiten un `calibrator`. El runtime ejecuta ese mismo artefacto sobre el vector target y sobre un vector de ancla final reformateado con nombres target. El calibrador final creado por el script nunca se promueve ni se usa.

**Corrección:** perfil con `calibrators.target` y `calibrators.final_anchor`; loaders, hashes, features y thresholds separados.

## 3. Extracción real de caracteres de WhisperX incompatible

WhisperX devuelve los caracteres en `result["segments"][i]["chars"]`. El repositorio solo busca `char_segments` en la raíz o `chars` dentro de `word_segments`. Las pruebas usan un fake con `chars` dentro de las palabras, forma que no representa la salida oficial. Con WhisperX real, la evidencia de caracteres puede quedar vacía y bloquear toda calibración.

**Corrección:** aplanar `segments[*].chars`, conservar índices/timestamps/scores y añadir una prueba contractual basada en una muestra real serializada de WhisperX.

## 4. Validación contaminada durante entrenamiento

`train_calibrator` solo excluye `hidden_test`; acepta filas `validation`. El script de entrenamiento pasa todas las filas recibidas sin filtrar. Esto contamina validación y produce métricas optimistas.

**Corrección:** entrenar únicamente con split `calibration`; usar `validation` solo para elegir hiperparámetros/thresholds; hidden test una sola vez.

## 5. Promoción del perfil no está ligada criptográficamente a la evaluación

El script de promoción acepta reportes JSON ya preparados. No recalcula las predicciones desde el artefacto, el dataset y las filas hidden. Basta que un JSON diga `false_pass_count=0` y tenga `run_id` no vacío. Tampoco exige ambas clases en validación/hidden; la propia prueba promociona un hidden set con un único positivo.

**Corrección:** la promoción debe recibir filas selladas y artefactos, recomputar internamente todos los resultados, exigir negativos suficientes y ligar hashes de artefacto, filas, predicciones y reporte.

## 6. El script normal de promoción produce identidad inutilizable

`promote_calibration_profile.py` fija backend/model/revisión en `unknown`, commit en `unknown` y hashes de locks en cero. El perfil resultante no puede coincidir con el runtime real.

**Corrección:** todos esos datos deben provenir de locks y del Git HEAD real, sin defaults ficticios.

## 7. El flujo gold set no puede completar doble revisión

La tabla `claims` tiene `clip_id` como clave primaria única. Una vez reclamado, el clip queda fuera de la cola y no existe liberación/reasignación para un segundo revisor. Sin embargo, el validador exige dos revisores independientes.

Además, la UI solo expone un GET JSON; no implementa reclamar, guardar etiquetas, adjudicar ni reproducir A/B.

**Corrección:** claims por `(clip_id, reviewer_id)` o tareas de revisión explícitas por ronda, expiración/lease y UI transaccional con POST seguro.

## 8. No existe puente gold set → features → calibradores

El repositorio puede guardar clips y labels y puede entrenar sobre JSONL de `FeatureRow`, pero no contiene un pipeline que abra los clips, ejecute la evidencia congelada, convierta las etiquetas humanas en targets separados y produzca filas feature con hashes/procedencia.

**Corrección:** crear extractor reproducible que produzca datasets target, final-anchor y LID separados.

## 9. Entrenamiento LID roto

`FeatureRow` rechaza cualquier feature fuera de TARGET/FINAL. `train_lid_calibrator.py` intenta crear `FeatureRow` con `LID_FEATURES`, por lo que lanza `ValueError` antes de entrenar.

**Corrección:** contrato `LIDFeatureRow` separado y esquema propio.

## 10. LID calibrado no está conectado al runtime

`lid.py` usa thresholds fijos y no carga el calibrador LID. `run_scene_v2` tampoco usa `independent_lid` ni `fuse_language_evidence`; sigue llamando al adapter antiguo bajo una condición basada en scores CTC crudos.

**Corrección:** unificar un solo flujo LID y conectarlo al router/orquestador con artefacto calibrado.

## 11. SpeechBrain interpreta mal el score

SpeechBrain devuelve el mejor score como log-likelihood/log-posterior; para obtener escala lineal se debe exponentiar. El adapter toma `output[1]`, lo convierte directamente a float y lo recorta a `[0,1]`. Un score válido negativo termina normalmente en `0.0`.

**Corrección:** `probability = exp(log_score)`, validar distribución y conservar el vector de clases cuando sea posible.

## 12. MFA nuevo usa una CLI inválida

El módulo nuevo ejecuta `mfa align_one[_hf] --clean --single_speaker AUDIO OUT`, omitiendo archivo de texto y modelos/diccionario o model ID. Esa firma no coincide con MFA oficial.

Existe además un adapter legado separado en `alignment.py` que sí construye la firma legacy completa. Hay dos implementaciones divergentes.

**Corrección:** eliminar duplicación, detectar capacidades mediante `mfa <command> --help`, construir transcript temporal y argumentos exactos para legacy/HF.

## 13. Cobertura TextGrid no compara contenido

`TextGrid.coverage()` concatena etiquetas y compara solo longitud: `min(len(heard), len(want))/len(want)`. Una cadena incorrecta del mismo tamaño obtiene 1.0. También puede mezclar tiers de palabras y teléfonos.

**Corrección:** seleccionar tier correcto, normalizar tokens/fonemas y hacer alineación de secuencia real.

## 14. Selector FMV no realiza reparación local tras fallo de escena

El selector elige el primer candidato montable por línea y audita. Si la auditoría de escena falla, la siguiente iteración vuelve a intentar los mismos candidatos primero; como no cambia nada, termina. No atribuye el fallo ni sustituye la línea responsable.

**Corrección:** la auditoría debe devolver blockers por línea/frontera y el selector debe avanzar el índice solo para los responsables, guardando tabu/visited states.

## 15. Benchmark real no existe

`run_real_benchmark.py` usa un runner fijo que devuelve `BLOCKED`. No invoca OmniVoice ni `run_scene_v2`.

**Corrección:** runner configurable y firmado que cargue manifest, adapter, runtime, modelos y QA real.

## 16. La promoción de rama se puede falsificar con JSON manual

`promote_branch.py` acepta un JSON que diga `real_audio=true`, `blocked=0`, `failed=0`, y un segundo JSON con `valid=true` e `independent_adapter=true`. No recalcula hashes, no verifica commit, número de líneas, QA, perfil o logs.

**Corrección:** recomputar evidencias desde artefactos firmados/content-addressed y exigir criterios mínimos.

## 17. El adapter de segundo juego no valida un juego

Declara válido cualquier manifest con una lista `scenes` no vacía. La prueba hace exactamente eso. No hay extracción, mapping, tiempos, contenedor, montaje, empaquetado ni smoke real.

**Corrección:** escoger un juego real y construir adapter completo.

# P1 — Fallos de integración y metodología

## 18. Módulos aislados no usados por `run_scene_v2`

No se invocan desde el camino real:

- `route_qa`
- `ModelPool`
- `StateStore`
- `plan_repairs` / `apply_repair`
- `classify_performance` / `policy_for`
- `independent_lid` / `fuse_language_evidence`
- `mfa_adapter.align_diagnostic`

Por tanto, sus commits no cambian todavía el comportamiento de producción.

## 19. Performance mode es global, no por línea

La configuración solo tiene `qa.performance_mode`. La clasificación automática de performance no se aplica por clip/línea. Las reglas actuales son heurísticas básicas y no implementan openSMILE, torchcrepe ni Parselmouth.

## 20. Reparaciones son planes, no operaciones

Sin un executor, `apply_repair` devuelve `PLANNED`. No existen implementaciones concretas de vocal extension, atempo causal, crossfade adaptativo o regeneración dirigida dentro del pipeline.

`max_attempts` está almacenado pero no se aplica; solo se bloquea una firma exactamente repetida.

## 21. Routing por coste no gobierna el pipeline

`route_qa` es una función pura probada en aislamiento. `run_cohorts` no la usa y el modelo pool tampoco está conectado a Whisper/CTC/LID/MFA.

## 22. Scene QA sigue siendo estructural

Solo verifica apertura, frames, canales, sample rate, finitud y dos booleanos calculados antes de serializar. No verifica clipping/true peak, seams, cortes de voz, loudness contextual, bed, continuidad entre líneas o fuga lingüística en escena.

## 23. Manifest de benchmark no es content-addressed

El digest incluye strings de paths, no hashes de bytes de audios, referencias, locks, config y perfil. Cambiar un archivo sin cambiar su path conserva el mismo manifest digest.

## 24. Gold set usa etiqueta única para defectos potencialmente múltiples

Un clip puede ser simultáneamente `LEXICAL_ERROR`, `FINAL_ANCHOR_MISSING` y `TIMING_BAD`, pero `HumanLabel` solo admite una etiqueta. Esto impide construir targets independientes fiables.

## 25. `add_clip` puede alterar un clip ya etiquetado

`INSERT OR REPLACE` permite reemplazar payload/SHA de un `clip_id`. Las etiquetas anteriores pueden sobrevivir referenciando contenido distinto.

## 26. Hidden test no está realmente sellado

`hidden_test_sealed` solo significa que hay al menos un clip. No hay control de acceso, registro de aperturas ni one-shot enforcement. Los labels se exportan igual que los demás.

## 27. Feature extraction de entrenamiento enmascara faltantes

Aunque el docstring dice que datos faltantes son error, `_float` convierte campos ausentes o inválidos a cero y duración ausente a `1e-6`. Puede producir velocidades absurdas y entrenar sobre evidencia incompleta.

## 28. Promoción no exige composición mínima del hidden set

No se requiere un mínimo de negativos, positivos, modos de performance, speakers o tipos de defecto. Cero falsos PASS observados puede ser trivial.

## 29. Release check no cubre los commits nuevos

`tests/run_v2.py` descubre solo `test_v2.py`. `release_check.py` no ejecuta `unittest discover`, trainers, goldset UI, MFA real, benchmark ni adapter segundo juego.

## 30. Pruebas principales usan fakes que no imitan las APIs reales

- WhisperX fake coloca `chars` en `word_segments`.
- LID fake devuelve probabilidades lineales perfectas.
- MFA solo prueba el error de ejecutable ausente.
- FMV solo prueba PASS en el primer intento.
- Segundo juego solo prueba que `scenes` no esté vacío.

## 31. P3R adapter todavía es inventario, no integración completa

`runtime_smoke` siempre devuelve `NOT_RUN`. `runtime_destinations` no valida path traversal y no existe la lógica real de contenedor/inyección/reempaque.

# Aspectos que sí están bien encaminados

- La política general es fail-closed y mantiene `calibration_authority=false` por defecto.
- Las dos lecturas Whisper se consideran una familia correlacionada.
- Uncertainty no dispara TTS automáticamente.
- Los locks y hashes del runtime están mucho mejor defendidos.
- El calibrador usa JSON seguro en lugar de pickle.
- El producto cartesiano FMV fue retirado del orquestador, aunque la reparación local aún está incompleta.
- La separación entre auditorías raw/procesada/montada/serializada es conceptualmente correcta.

# Secuencia correctiva recomendada

1. **Fix calibration math and multi-artifact contracts**  
   Normalización train/runtime idéntica; artefactos target/final/LID separados.

2. **Parse real WhisperX character output**  
   `segments[*].chars`, fixture real y pruebas de integración.

3. **Repair gold-set review workflow and feature bridge**  
   Doble revisión real, UI escribible, adjudicación y extracción reproducible.

4. **Seal calibration splits and promotion evidence**  
   Sin validation leakage; hidden recomputado internamente y ligado por hashes.

5. **Integrate calibrated LID and correct SpeechBrain probabilities**  
   Exponenciar log score, usar perfil LID y conectar al orquestador.

6. **Consolidate and integrate MFA**  
   Una implementación, firmas oficiales, fixtures de salida y fallback real.

7. **Wire performance, cost routing, model pool, state and repairs**  
   Integración end-to-end con budgets y reanudación.

8. **Implement attributable FMV local repair and continuous scene QA**  
   Sustitución dirigida y gates contextuales reales.

9. **Build real benchmark and real second-game adapter**  
   Evidencias content-addressed, runner real, adapter real y promotion gate no falsificable.

10. **Expand CI to full integration matrix**  
    `unittest discover`, optional-dependency jobs y E2E con fixtures reales.

# Criterio de salida

No afirmar “completo” hasta que:

- el full CI ejecute todas las pruebas;
- haya fixtures reales WhisperX/SpeechBrain/MFA;
- el gold set pueda completarse desde la UI;
- los calibradores target/final/LID sean separados y matemáticamente reproducibles;
- `run_scene_v2` use realmente routing, pool, performance, repairs y fallback;
- FMV sustituya una línea tras un fallo de escena;
- el benchmark invoque el pipeline real;
- un segundo juego real pase sin modificar el core.
