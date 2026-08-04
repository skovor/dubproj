# Generic dubbing pipeline source

This directory is the portable code layer extracted from the production
dubbing pipelines. It is intentionally independent of a particular game,
character, language pair, engine, or mod loader.

The pipeline supports the two physical topologies that caused most production
failures:

* `LINE_SEPARATED`: one WAV/OGG/codec asset per voice line (VN, in-engine,
  barks and dialogue banks).
* `EMBEDDED_FMV`: the movie contains a multiplexed dialogue stem and each
  subtitle line is a timed replacement window (FMV, anime and prerendered
  cutscenes).

The same contract is used for both routes: mapping first, text policy second,
reference validation third, generation, processing, QA, montage/package and
runtime smoke last. English source text belongs to the reference; German (or
the configured target language) is the synthesis text.

## Layout

```text
src/
  dubbing_pipeline/       reusable, generic runtime modules
  adapters/                evidence-backed game/middleware adapter template
  scripts/                thin command-line entry points
  schemas/                project/scene/line JSON contracts
  config/                 safe example configuration (no local paths)
  tests/                  dependency-light regression tests
  ported/                 exact production modules copied for code review;
                         these are provenance/reference, not the generic API
  PORT_MANIFEST.json      source-to-destination and dependency inventory
```

The `ported/` snapshot is included so an external reviewer can inspect every
production-critical implementation (OmniVoice generation, ASR QA, splice,
USM replacement, staging and deploy). It contains no audio, maps, models,
credentials or game files. A project should call the generic API and provide
an adapter/configuration rather than editing those files.

## Quick start

```powershell
cd src
python -m venv .venv
.\.venv\Scripts\pip install -e .
python scripts\run_pipeline.py validate --config config\project.example.json --manifest path\to\manifest.json
python scripts\run_pipeline.py review-bundle --config config\project.example.json --manifest path\to\manifest.json --out review_bundle
```

Heavy dependencies are optional because a mapping or contract audit should
run without CUDA. Install the backend extras only on the machine that will
generate/QA audio. See `pyproject.toml` and `docs/PORTING_MATRIX.md`.

## Safety guarantees

* Only subtitle-authorized units enter TTS. Unproven/no-card/background audio
  is `KEEP_ORIGINAL`, not silently invented dialogue.
* FMV Empalme B preserves the original effort/onomatopoeia and synthesizes
  only the target-language body; the preserved audio is never used with a
  mismatched reference transcript.
* `...` is appended only to `tts_text`, never to canonical target text,
  source English text or `ref_text`.
* Text/WER, onset, span, rate, pause and pitch are diagnostics/ranking. A
  score cannot override a failed hard gate.
* Candidate retries are directed (maximum four normally, `4 + 4` only for a
  declared FMV/anime profile) and the OmniVoice model is persistent per run.
* Packaging stages first, verifies hashes/roundtrip, backs up destinations,
  then replaces each destination atomically. Runtime smoke remains a separate
  gate; a copied file is not claimed as runtime-tested until the game loads it.

## GPT review

`dubbing_pipeline.review` emits a deterministic JSON bundle containing only
manifests, contracts, gate failures, timing and hashes. It deliberately omits
audio/text spoilers unless the caller opts in. `docs/GPT_REVIEW_PROMPT.md`
contains a prompt suitable for an external GPT review of that bundle.
