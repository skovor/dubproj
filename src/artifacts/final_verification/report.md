# Verificación final — cierre 5824FD5

**Estado:** `IMPLEMENTED_BUT_REAL_AUDIO_BLOCKED`

La matriz canónica está en [`../../../artifacts/final_verification/report.json`](../../../artifacts/final_verification/report.json).
El código funcional validado es `627b84bb2526b2af028a39b8fa7b1dfddcfa4986` en
`refactor/p3r-pipeline-v2`; el SHA local y el remoto coinciden.

## Commits de esta corrección

1. `bf72d3d1015a73acde9d60b000e7872083719b85` — labels hidden y bridge autoritativos.
2. `a5524cd5dd85c5bf067177034b77a4e3483de018` — identidad observada por Git.
3. `0a7420c3dc444e027c42cb3fc36412f267bceade` — trust store Ed25519.
4. `bb1e49bf95ee61e764bc05e9bde1b54f26372473` — procedencia firmada del segundo juego.
5. `627b84bb2526b2af028a39b8fa7b1dfddcfa4986` — instalación de attestation en CI.

## CI del HEAD exacto

La ejecución [#71](https://github.com/skovor/dubproj/actions/runs/31057970313), ID
`31057970313`, terminó `success` sobre `627b84b…`. `full-suite` y los cuatro
`platform-smoke` (Ubuntu/Windows, Python 3.10/3.12) fueron `success`; el job
`ml-import-contracts` quedó `skipped` por su condición `workflow_dispatch`.
El artefacto `release-check-31057970313` tiene ID `8951002036` y digest
`sha256:be521a1b5c3f27c4e183203cce8a29a03f21fd6410a257e99c10e9a4c5d41536`.

## Pruebas locales

- `pytest -q`: **184 passed, 3 skipped, 3 warnings, 11 subtests**, código 0.
- `unittest discover`: **162 tests OK**, código 0.
- `compileall`, `pip check`, `git diff --check` y `release_check --skip-pytest`: código 0.

No se declara audio real: P3R y DQ3 siguen bloqueados por ausencia de assets
extraídos, timing/subtítulos verificables y extractor autorizado. No se inventaron
benchmarks, gold labels ni resultados OmniVoice. `calibration_authority` sigue en
`false`.
