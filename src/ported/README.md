# Ported production modules

These files are a source-preserving snapshot of the production implementation
used to close the FMV and line-separated routes. They are kept for audit and
GPT code review; the generic runtime in `../dubbing_pipeline/` is the public
API.

The snapshot intentionally contains no audio, maps, models or credentials.
Some modules retain their original imports and command-line defaults so their
algorithmic provenance is not silently changed. Do not run them against a new
game without supplying a project adapter and replacing local paths with the
config-driven equivalents.

`PORT_MANIFEST.json` records the source family and intended use. The generic
modules extracted from these files are the ones that remove the original
project assumptions.
