# Uncertainty-aware linguistic QA

This commit changes the V2 linguistic verdict from one Whisper transcript to
two cached readings of the same audio artifact:

- `forced_target`: faster-whisper decoded with the configured target language.
- `automatic`: faster-whisper decoded with automatic language detection.

`content` and `final_word` remain exact ordered hard gates. The new layer does
not accept a spelling error as a pass. It records both transcripts, language,
probability, missing tokens, final anchors, and SHA-256 evidence hashes.

The statuses are explicit:

- `PASS_EXACT`: both readings agree on target content and the final anchor.
- `FAIL_CONFIRMED`: both target-language readings miss required content.
- `LANGUAGE_LEAK_CONFIRMED`: automatic evidence indicates source-language speech.
- `ASR_UNCERTAIN`: the readings disagree or language confidence is insufficient.

`ASR_UNCERTAIN` is a hold, not a regeneration trigger. The report emits a
serializable `whisperx_escalation` request for selective forced alignment. No
WhisperX or MFA model is loaded in this commit.

The cache is keyed by artifact SHA-256, decode mode, language, and backend. A
semantic alias may be supplied only for a transform known to preserve speech
(the V2 orchestrator uses it for resampling-only processing); mounted and
speech-changing artifacts receive fresh evidence. Reopening the exact mounted
file hits the SHA cache, so serialization QA does not transcribe again.

The regression suite covers German Unicode normalization, forced/automatic ASR
disagreement, source-language audio that is forced into German, cache reuse,
and WhisperX request serialization.
