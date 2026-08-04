# Independent-evidence linguistic QA

The V2 linguistic verdict is deliberately split into screening and
confirmation.  Faster-Whisper is run twice on each candidate artifact:

- `forced_target`: decode with the configured target language;
- `automatic`: decode with automatic language detection.

Those readings are correlated evidence, not two independent votes.  They
share the model, tokenizer, acoustic representation and failure modes, so they
can produce `PASS_SCREENED`, `ASR_UNCERTAIN`, or `LANGUAGE_LEAK_SUSPECTED`, but
they cannot by themselves produce a confirmed linguistic hard pass/fail.

## Evidence contract

Every durable reading is represented by an `EvidenceRecord` containing the
family, backend/model/revision, mode, artifact SHA-256, semantic alias, output,
confidence and evidence hash.  The current families are:

- `WHISPER_ASR` — both forced and automatic Whisper modes;
- `CTC_FORCED_ALIGNER` — WhisperX's wav2vec2/CTC alignment path;
- `KALDI_FORCED_ALIGNER` — the optional MFA path for difficult cases;
- `AUDIO_LANGUAGE_ID` — optional SpeechBrain VoxLingua107 spoken-language ID;
- `HUMAN_REVIEW` — an explicit human decision.

A hard linguistic confirmation requires at least two distinct families.  The
deterministic technical gates (corrupt audio, clipping, wrong sample rate,
impossible window, and similar physical failures) remain independently
decidable and do not need an acoustic-language second family.  Alignment/LID
records whose artifact SHA does not match the Whisper artifact are rejected as
`ALIGNMENT_UNCERTAIN` rather than being fused.

### Calibration authority

CTC/Kaldi scores are diagnostic by default. `QAConfig.calibration_authority`
is `false` unless a gold-set profile is deliberately enabled. Enabling it
does not suffice by itself: the profile must be authorized and match the
active aligner `model_id`, `model_revision`, source/target languages, and
performance mode. A missing, stale, or mismatched profile is `BLOCKED`; the
runner never falls back to the default `.65` threshold as a hard verdict.
Calibration blocks are classified as `FailureClass.DETERMINISTIC_CALIBRATION`
and are held without OmniVoice regeneration; they are not acoustic ASR
uncertainty.

## Decision states

| State | Meaning | OmniVoice retry? |
| --- | --- | --- |
| `PASS_SCREENED` | Both correlated Whisper readings match target text/final word and automatic language is target. | No; align the provisional winner. |
| `PASS_SCREENED_WITH_ALIGNMENT_SUPPORT` | Uncalibrated target alignment supports a screened candidate. | No; hold, not production eligible. |
| `TARGET_ALIGNMENT_SUPPORT` | Uncalibrated target alignment is numerically strong. | No; diagnostic only. |
| `TARGET_ALIGNMENT_WEAK` | Uncalibrated target alignment is below threshold. | No; diagnostic only. |
| `EVIDENCE_CONFLICT` | Whisper/source evidence conflicts with an uncalibrated strong target score. | No; hold. |
| `PASS_CONFIRMED` | A matched calibration profile and second family confirm a screened candidate. | No. |
| `PASS_PHONETIC` | A matched calibration profile confirms target phonetic content after Whisper disagreement. | No. |
| `ASR_UNCERTAIN` | Whisper evidence disagrees or is insufficient. | No; selective alignment/hold. |
| `ALIGNMENT_UNCERTAIN` | Target-only alignment is unavailable or not calibrated. | No; hold. |
| `LANGUAGE_LEAK_SUSPECTED` | Whisper or CTC suggests source speech without independent confirmation. | No; hold/escalate. |
| `LANGUAGE_LEAK_STRONG_SUSPICION` | Whisper and/or independent LID favor source language while uncalibrated target CTC is weak. | No; hold/calibrate. |
| `LANGUAGE_LEAK_CONFIRMED` | Matched calibration plus Whisper and independent LID favor source language while target CTC is weak. | Regeneration may be scheduled explicitly. |
| `LEXICAL_FAILURE_SUSPECTED` | Target-only alignment is weak; calibration is still required. | No; hold/calibrate. |
| `LEXICAL_FAILURE_CONFIRMED` | Matched calibration confirms target lexical failure. | Regeneration may be scheduled explicitly. |
| `ALIGNER_NOT_APPLICABLE` | No configured independent aligner. | No; never silently pass. |
| `HUMAN_REVIEW` | Automated evidence cannot decide. | No; explicit review. |

The old generic `PASS_EXACT` and `FAIL_CONFIRMED` labels are not emitted by
the V2 linguistic layer: they hid whether the result was an actual second
family confirmation and incorrectly treated two Whisper modes as independent.

## Selective escalation

The cost-controlled order is:

1. technical QA + both Whisper modes for every candidate;
2. rank provisional candidates and align only the provisional winner;
3. align a fallback only when the winner is rejected or ambiguous;
4. for source-preferred alignment, optionally run independent VoxLingua107
   LID; reserve MFA for persistent difficult cases.

The CTC adapter aligns the known German subtitle and the original English
subtitle for diagnostics, but it never compares raw cross-language scores as a
hard gate: German and English alignment models are not calibrated onto one
probability scale. Before calibration, a target-only score can emit only
`TARGET_ALIGNMENT_SUPPORT`, `TARGET_ALIGNMENT_WEAK`,
`PASS_SCREENED_WITH_ALIGNMENT_SUPPORT`, or `EVIDENCE_CONFLICT`; all remain
ineligible and map to `ASR_UNCERTAIN` for scheduling. A matched calibration
profile is required before `PASS_CONFIRMED`, `PASS_PHONETIC`,
`LANGUAGE_LEAK_CONFIRMED`, or `LEXICAL_FAILURE_CONFIRMED` can be emitted. A
raw cross-language margin is preserved as `cross_language_margin` telemetry
only, and it never overrides a source-language gate before calibration.

## Character-level alignment evidence

The WhisperX adapter requests `return_char_alignments=True` for the target
subtitle and stores normalized character segments, native character coverage,
mean/minimum/p10 scores, unaligned and interpolated characters, and a
compression ratio. The final token is represented by a structured
`final_anchor_evidence` object with timing, coverage, score, interpolation,
speech-tail gap, and one of `FINAL_ANCHOR_EVIDENCE_COLLECTED`,
`FINAL_ANCHOR_UNALIGNED`, `FINAL_ANCHOR_WEAK`, or
`FINAL_ANCHOR_INTERPOLATED`.

All of these fields are `DIAGNOSTIC_ONLY`. The character adapter does not
emit a boolean `final_anchor_present` authority signal. A validated profile
using `feature_schema_version = char-alignment-v1` may use the collected,
non-interpolated final anchor only after its calibrated minimum score
threshold is met.

The line report is candidate-aware (`candidate_linguistic_decisions`,
`line_linguistic_summary`, and `selected_candidate_linguistic_decision`).  An
escalation record includes the candidate ID, so a weak first take cannot mask a
good second take.

## Cache boundaries

Whisper and CTC caches include artifact SHA-256, mode/text, language, backend,
model and revision.  A semantic alias is used by the orchestrator only for an
explicit resample known to preserve speech.  Trim, atempo/time-stretch,
montage, splicing, mixing, or any other speech-changing transform has no
semantic alias and therefore gets fresh evidence.  Alignment cache entries
are separately keyed by audio SHA, hypothesis text, language and model
revision.

## Optional dependencies

Install `generic-dubbing-pipeline[alignment]` to enable WhisperX and
SpeechBrain adapters.  The core package remains importable without them; if no
aligner is configured, candidates are held as `ALIGNER_NOT_APPLICABLE` rather
than being marked PASS or regenerated as if a linguistic failure were proven.

The regression suite covers correlated Whisper failures, CTC confirmation,
source-language confirmation requiring independent LID, missing aligners,
candidate-specific reporting, alignment caching, and the no-retry scheduler
policy.
