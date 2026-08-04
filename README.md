# dubproj

Generic, contract-first game dubbing pipeline.

The reusable source is under [`src/`](src/README.md). It covers independent
voice files, in-engine timelines and embedded FMV/anime: mapping and subtitle
authority, source-language references, OmniVoice generation/cache, exact
timing, Empalme B, language-leak/final-word QA, montage, container adapters,
staging, hashes, rollback and atomic deploy.

The `src/ported/` directory is a code-review snapshot of 56 production-critical
modules. It contains no game data, audio, model weights or credentials. The
generic runtime is `src/dubbing_pipeline/`; project-specific facts belong in
an evidence-backed adapter under `src/adapters/`.

## Verify locally

```text
cd src
python tests/run_smoke.py
python scripts/check_port.py
python -m compileall -q .
python scripts/run_pipeline.py validate --config config/project.example.json --manifest config/scene.example.json
```

See `src/docs/GPT_REVIEW_PROMPT.md` for an evidence-only external review and
`src/PORT_MANIFEST.json` for the port inventory.

The portable instruction layer (nine skills plus promoted AC-57…AC-66 rules)
is under [`src/instructions/`](src/instructions/README.md). It is sanitized:
no KIRO files, game corpus, audio, credentials or model data are included.
