# V2 implementation and measurement report

This branch implements the plan maestro as a guarded strangler migration. The
functional `main` worktree is untouched; P3R data is referenced read-only by a
hash-only baseline in the external sandbox.

## Implemented invariants

- `contracts/` rejects unknown fields, missing IDs, invalid windows, undeclared
  FMV overlap and malformed evidence. Gate values are typed (`PASS`, `FAIL`,
  `NOT_APPLICABLE`, `NOT_RUN`, `ERROR`).
- `reference.py` materializes the declared sample range/channel and pairs that
  exact audio hash with its segment transcript. `Line.reference_text` now uses
  segment text instead of blindly using the full source sentence.
- `generation_v2.py` keeps one backend runtime, uses a batch method when the
  backend exposes one, preserves native sample rate, validates readback and
  hashes every generation parameter that can change audio.
- `hashing.py` uses unique temporary files, fsync/readback and semantic hashes
  that do not change when a reference moves to another machine.
- `qa_v2.py` measures finite audio, rate, channels, frames, clipping, active
  level, tail, content order, final-word suffix and source-language leakage.
  Missing ASR is `NOT_RUN`, never a fabricated PASS. Ranking cannot rescue a
  failed candidate.
- Linguistic QA now records evidence families. Forced-target Whisper and
  automatic Whisper are correlated `WHISPER_ASR` screening, never two
  independent votes. `PASS_SCREENED` is promoted only by selective
  `CTC_FORCED_ALIGNER`/`KALDI_FORCED_ALIGNER` evidence; source-language
  confirmation can additionally require independent `AUDIO_LANGUAGE_ID`.
  WhisperX alignment compares the known target and source subtitle texts and
  does not perform a third Whisper transcription. If the second family is
  unavailable or inconclusive, the candidate is held rather than passed or
  regenerated.
- Raw target-versus-source CTC margins are now diagnostic telemetry only;
  they cannot confirm a German pass or an English leak because separate
  language models are not calibrated onto one score scale. English leakage
  requires source-favoring Whisper plus independent LID and weak target CTC.
- `post_qa.py` makes the delivery boundary explicit: every raw candidate is
  re-audited after processing, after surgical mounting, and after reopening
  the serialized artifact. FMV scenes receive a final full-scene audit that
  checks frame/rate/channel contracts, protected Empalme B samples and
  byte-identical non-dialogue channels. A raw PASS is therefore never emitted
  as `FINAL_PASS`.
- `orchestration_v2.py` selects line-separated winners only after serialized
  QA and selects FMV winners from bounded candidate combinations that pass the
  complete scene audit. Failed post-transform stages remain in the report as
  stage evidence instead of silently disappearing. Candidate linguistic
  summaries and selective alignment requests are candidate-specific; no
  `raw_rows[0]` value acts as line authority.
- `montage.py` replaces only the declared speech mask, preserves Empalme B
  intervals and resume tails, keeps non-dialogue channels sample-identical and
  rejects a body that would be actively cut.
- `deploy_v2.py` validates relative paths, records EXISTS/ABSENT pre-state and
  removes newly created files during rollback. `package.py` now delegates to
  this implementation.
- `scheduler.py`, `state.py` and `telemetry.py` implement global cohorts,
  run-scoped events and append-only resumable state. Retry candidates are
  limited to stochastic TTS failures. The cohort scheduler no longer reports
  selection, mounting, packaging or runtime smoke phases that it did not
  execute; the owning orchestration layer appends those stages only after
  evidence exists.
- The CLI has `preflight`, `plan`, `status`, `resume` and explicit dry-run
  heavy-phase commands. `lab_mode` requires sandbox roots before work starts.
- `validate_instructions.py`, `reference-index.json`, and the V2 promotion
  manifest make the sanitized skill bundle reproducible without copying KIRO
  or private lesson corpora.
- `runtime_lock.py` and `scripts/freeze_runtime.py` make the runtime/model
  identity explicit. The checked-in locks are deliberately unprovisioned
  templates; strict production preflight rejects unknown revisions, backend
  versions and model SHA-256 values. Lab mode reports `LAB_UNPINNED` rather
  than disguising that state as a reproducible run.

## Baseline and benchmark

The external sandbox reports are:

`work/sandbox/p3r-pipeline-v2/data/input_manifest/BASELINE_MANIFEST.json`

and

`work/sandbox/p3r-pipeline-v2/reports/performance/before_after.json`.

The baseline records the frozen commit `5ca877e`, 20 P3R scene reports, and the
current summary (`required=170`, `mounted=170`, `release_ready=false` because
perceptual review remains a real gate). Its 1,350 legacy timing rows sum to
304.44 minutes (QA 198.65 min, generation 89.27 min, continuous audit 13.23
min, mount 3.29 min); those rows are explicitly marked not run-scoped, so they
are a historical reference rather than a precise per-run benchmark. It does
not copy or write production audio.

The benchmark compares 170 synthetic units: roughly 0.266 s legacy vs 0.044 s
V2 (6.00x in the latest synthetic harness), while the old QA admitted 5 known bad cases
and V2 admitted 0. These are architecture/QA measurements, not a claim about
OmniVoice GPU throughput; the report lists the limitations.

## Still intentionally gated

The branch does not load OmniVoice weights, rewrite the real P3R output, or
claim an in-game smoke result. The next safe phase is a P3R 16–24-line
microset through the adapter, followed by legacy/V2 shadow classification and
only then a full 20-scene run in a disposable runtime clone.
