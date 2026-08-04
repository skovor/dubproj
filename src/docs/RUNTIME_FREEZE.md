# Runtime and model freeze

The V2 branch now fails closed for a real run unless the host, dependencies,
backend and model files are identified by exact revisions and SHA-256 hashes.
The checked-in `config/runtime.lock.json` and `config/models.lock.json` are
intentionally `UNPROVISIONED` templates: this generic repository cannot know
the CUDA driver, OmniVoice checkout or local model files on a user's machine.

On the generation machine, create a project-specific lock without downloading
anything:

```powershell
cd src
python scripts/freeze_runtime.py `
  --out-dir config `
  --model-id k2-fsa/OmniVoice `
  --model-revision <git-tag-or-commit> `
  --model-file <path-to-each-model-file> `
  --backend-version <omnivoice-backend-commit>
```

For more than one model, pass `--models-manifest` with a JSON array containing
`model_id`, `revision`, `sha256`, `language`, `sample_rate`, `backend`,
`backend_version` and `files`. Never use `unknown`, `main`, `latest`,
`not-installed`, `not-readable` or a null revision in a production lock.
Optional components must be marked `DISABLED_EXPLICITLY`; they may not be
represented by an ambiguous string.

The example configuration remains `lab_mode: true`, so CI can inspect the
template and report `LAB_UNPINNED`. A production configuration must set
`lab_mode: false` and reference both lockfiles; then:

```powershell
python scripts/run_pipeline.py preflight `
  --config <production-config.json> `
  --manifest <scene-manifest.json> `
  --strict-runtime
```

will return a non-zero status until every required identity is pinned. Preflight
also reopens each declared model file, checks byte count and SHA-256, and
recomputes the aggregate model hash. Moving the same bytes is allowed only
when the logical path resolves under `models_root`. Model revision changes are
part of ASR/alignment cache keys, so changing a lock cannot reuse old evidence.
