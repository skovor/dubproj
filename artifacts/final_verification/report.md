# Verificación final — cierre 5824FD5

**Estado:** `IMPLEMENTED_BUT_REAL_AUDIO_BLOCKED`

La corrección funcional se ejecutó sobre `refactor/p3r-pipeline-v2` y quedó en
`627b84bb2526b2af028a39b8fa7b1dfddcfa4986`. El SHA local y el remoto coinciden.
Este documento y `report.json` son evidencia documental posterior; no alteran el
código validado.

## Cinco commits funcionales

| Commit | Corrección | Evidencia local |
|---|---|---|
| `bf72d3d1015a73acde9d60b000e7872083719b85` | Etiquetas hidden y receipts del bridge autoritativas | 28 pruebas focalizadas |
| `a5524cd5dd85c5bf067177034b77a4e3483de018` | Checkout observado por Git separado del SHA declarado | 73 pruebas focalizadas |
| `0a7420c3dc444e027c42cb3fc36412f267bceade` | Trust store Ed25519 para atestaciones | 18 pruebas focalizadas |
| `bb1e49bf95ee61e764bc05e9bde1b54f26372473` | Procedencia firmada y límites del adapter del segundo juego | 14 pruebas focalizadas |
| `627b84bb2526b2af028a39b8fa7b1dfddcfa4986` | Dependencia de atestación instalada en CI | CI #71 verde sobre este SHA |

Cada commit fue probado, verificado con `git diff --check HEAD^..HEAD`, publicado
sin force-push y comprobado contra `git ls-remote`.

## Evidencia CI independiente

Workflow: `.github/workflows/ci.yml` (`generic-dubbing-ci`)

- Run: [#71](https://github.com/skovor/dubproj/actions/runs/31057970313), ID `31057970313`.
- `head_sha`: `627b84bb2526b2af028a39b8fa7b1dfddcfa4986`.
- `full-suite`: `success` (job `92479547881`).
- `platform-smoke`: los cuatro jobs Ubuntu/Windows con Python 3.10/3.12, todos `success`.
- `ml-import-contracts`: `skipped` porque solo se ejecuta con `workflow_dispatch`; no se presenta como evidencia ML.
- Artefacto: `release-check-31057970313`, ID `8951002036`, digest
  `sha256:be521a1b5c3f27c4e183203cce8a29a03f21fd6410a257e99c10e9a4c5d41536`.

## Verificación local reproducible

- `PYTHONPATH=src src/.venv/Scripts/python.exe -m pytest -q`: código `0`, **184 passed, 3 skipped, 3 warnings, 11 subtests**, 6.88 s.
- `PYTHONPATH=src src\.venv\Scripts\python.exe -m unittest discover -s src/tests -p 'test_*.py' -q`: código `0`, **162 tests OK**, 5.086 s.
- `cd src; .venv\Scripts\python.exe -m compileall -q .`: código `0`.
- `cd src; .venv\Scripts\python.exe -m pip check`: `No broken requirements found`.
- `src\scripts\release_check.py --skip-pytest --out src\artifacts\final_verification\release_check.json`: código `0`.
- `git diff --check`: código `0`.

Los skips y el job ML son explícitos; no representan audio real ni modelos cargados.

## Matriz de integridad y límites

Los cuatro hallazgos de la auditoría (`hidden` autoritativo, identidad de checkout,
trust anchor y procedencia del segundo juego) están `VERIFIED` mediante código,
pruebas adversariales y la ejecución CI indicada. El CI solo cubre contratos CPU y
smoke multiplataforma.

P3R y Dragon Quest III HD-2D permanecen `BLOCKED`: no hay escenas extraídas con
timing/subtítulos ni extractor autorizado disponible. No se inventaron 20 líneas,
FMV, benchmark, gold set ni resultados OmniVoice. `calibration_authority` permanece
`false`; WhisperX, SpeechBrain, MFA y OmniVoice GPU no se ejecutaron en CI.

El resultado de release/CI demuestra integridad de software y reproducibilidad del
repositorio, no calidad acústica ni validación dentro de un juego.

La matriz completa, hashes y estados está en
[`report.json`](report.json).
