# Generic instruction bundle

This directory is the portable, project-neutral instruction layer for the
runtime in `src/`. It contains the nine source skills and the ten latest
promoted AC rules. It deliberately excludes KIRO files, game assets, audio,
subtitle dumps, model weights, credentials and project-specific reports.

## Contents

- `skills-src/` — authoritative skill sources used to route mapping, text
  policy, FMV timing/splicing, generation/QA, packaging and diagnosis.
- `rules/AC-57.md` … `rules/AC-66.md` — literal bodies of the rules promoted
  from operational lessons. Their relationship to the lesson IDs and their
  generic titles is recorded in `PROMOTION_MANIFEST.json`.

The rules are additive to the earlier AC rule set. They do not change the
source text, reference-audio contract or topology-specific adapters. A game
adapter must still prove its own map, timebase, codec and runtime behavior.

## Operational contract

1. Map subtitle-authorized deliveries to physical owners before generation.
2. Keep source-language audio/text paired for reference; synthesize only the
   configured target-language text.
3. Use topology-specific cohorts and directed retries; a failed candidate is
   never promoted merely because it is the best available failure.
4. For embedded FMV, preserve source efforts/onomatopoeia and replace only the
   mapped dialogue body when the acoustic, linguistic and map evidence agrees.
5. Treat diagnostics as evidence until the hard gates, montage audit, package
   roundtrip and runtime smoke all pass.

## Reproduce the instruction build

The original instruction system remains the authoritative build environment.
After changing a skill or promoting a lesson there, run its source resync,
skill build and packet validators, then copy only the generic artifacts into
this directory. This repository is a reviewable, sanitized distribution; it
is not a copy of any game's private corpus.
