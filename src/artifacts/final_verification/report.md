# Verificacion final de Dubproj

**Estado:** `IMPLEMENTED_BUT_REAL_AUDIO_BLOCKED`

Codigo auditado: `cca8466f072a71cb38f921d5f2b115fe0da7f474` en
`refactor/p3r-pipeline-v2`. La matriz completa esta en `report.json`.

## Pruebas reproducibles

- `PYTHONPATH=src src/.venv/Scripts/python.exe -m pytest -q`: codigo `0`, `172 passed, 3 skipped, 11 subtests passed`.
- `PYTHONPATH=src src/.venv/Scripts/python.exe -m unittest discover -s src/tests -p test_*.py -q`: codigo `0`, `150 tests OK`.
- `src/scripts/release_check.py --skip-pytest --out src/artifacts/release_check_ad1eddb.json`: codigo `0`, `PASS`.
- `pip check`: `No broken requirements found`.
- `compileall`, smoke, V2, portabilidad e instrucciones: todos `returncode=0` en el release check.

## Cierre AD1EDDB: cuatro commits atomicos + hotfix de integridad

| Commit | Correccion | Evidencia |
|---|---|---|
| `623379333dd932bb05c7ced103d06cca3dc91724` | Finalizacion oculta autoritativa en SQLite y consumo one-shot | 17/17 tests afectados |
| `e4f1d9a56739f61249a13743ad0cb350872e2eb6` | SHA exacto de calibracion antes de TTS e identidad unica de alineador | 75/75 tests afectados |
| `3009e7b9855570ae8095864ace75d055fdec3c1d` | Recibo de invocacion de escena y atestacion Ed25519; booleanos rechazados | 26/26 tests afectados |
| `e0f768676ded243f8624ef164a375bb773389a9e` | Decodificacion y procedencia content-addressed del segundo juego | 172 pytest; 150 unittest |
| `cca8466f072a71cb38f921d5f2b115fe0da7f474` | Hotfix: el SHA esperado debe coincidir con el checkout que ejecuta TTS | 68 focused pytest; suite completa repetida |

Los cuatro hallazgos nuevos de la auditoria estan `VERIFIED` en `report.json`. La
dependencia de firma esta declarada en `src/pyproject.toml` y fijada en
`src/constraints-ci.txt`. No se fabrico ninguna firma, benchmark o resultado de audio real.

## Que queda bloqueado

Se intento P3R y DQ3, pero no hay clips extraidos con subtitulos/timing ni extractor
autorizado disponible: las instalaciones permanecen en contenedores propietarios.
Por tanto no se declara validacion de juego, ni se inventan 20 lineas, FMV o un
benchmark. El estado permitido es `IMPLEMENTED_BUT_REAL_AUDIO_BLOCKED`.

Limitaciones restantes: obtener un extractor autorizado y ejecutar el adapter sobre
audio real; completar un gold set humano; y proporcionar un `repair_executor` real
para reparaciones tecnicas. La ausencia de esos recursos queda bloqueada, no produce
regeneraciones arbitrarias ni falsos `FINAL_PASS`.
