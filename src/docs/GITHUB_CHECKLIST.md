# GitHub publication checklist

Before creating the repository/commit:

1. Keep `config/project.example.json` and `config/scene.example.json`; create
   a private project config outside the public tree.
2. Do not commit WAV/USM/HCA/OGG files, model weights, generated candidates,
   runtime backups, `.env` files or game maps. The included `.gitignore`
   covers the common extensions.
3. Run from `src/`:

   ```text
   python tests/run_smoke.py
   python scripts/check_port.py
   python -m compileall -q .
   python scripts/run_pipeline.py validate --config config/project.example.json --manifest config/scene.example.json
   ```

4. Review `PORT_MANIFEST.json` and `docs/PORTING_MATRIX.md` so a reviewer
   understands which modules are generic and which are provenance snapshots.
5. Add one evidence-backed adapter under `adapters/` for the target engine or
   middleware. Never hard-code a workstation path in `dubbing_pipeline/`.
6. Require a real runtime smoke report before calling a deployment released;
   file hashes prove copying, not that the game selected the override.
