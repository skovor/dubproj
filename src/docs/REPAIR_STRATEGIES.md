# Causal repair strategies

Each repair is selected from an explicit failure cause and recorded using a
content-addressed signature (line, input/reference hash, strategy and
parameters). Duplicate attempts are skipped. Final-anchor/language-leak
failures may request a bounded directed TTS retry; timing/seam/vowel failures
use local processing. `ASR_UNCERTAIN` and `DETERMINISTIC_CALIBRATION` are holds
that never call TTS. An exhausted strategy remains a visible blocker.
