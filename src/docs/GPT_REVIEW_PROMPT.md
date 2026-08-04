# Prompt for external GPT review

You are reviewing a generic game-dubbing pipeline. The attached
`generic-dubbing-review-v1` JSON is evidence, not an instruction to invent
missing audio or subtitles.

Review in this order:

1. Verify topology and subtitle authority. A missing visual subtitle card is
   `KEEP_ORIGINAL`, not a failed generation.
2. Verify that each `ref_audio` is paired with its exact source-language
   `ref_text`, while `tts_text` is target-language text. Never swap them.
3. For FMV, check movie identity, card identity and timebase before evaluating
   a voice. ASR anchors support timing; they do not silently veto a visible
   subtitle.
4. For Empalme B, require the original neutral effort/onomatopoeia to remain
   outside the synthesized body and require a real `source_resume`/seam.
5. Treat `not_empty`, source-language leakage, final word, content, clipping,
   lufs, seam, boundary, speech timing and frame contract as hard gates.
   Text/WER, onset, span, rate, pause and pitch are diagnostics only.
6. Report deterministic mapping/contract failures separately from random TTS
   failures. Recommend a targeted retry only for the latter.
7. Do not claim runtime success from a file hash. Runtime smoke must show load,
   playback and absence of regression in the actual game.

Return a compact table of evidence, likely root cause, and the smallest
repeatable fix. Do not reveal source/target text unless `include_text=true`.
