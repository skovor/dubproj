# MFA diagnostic fallback

MFA is an optional, sandboxed forced aligner for OOV names, syllables and CTC
ambiguity. The adapter records executable/version, acoustic model, dictionary,
G2P and hashes, uses one explicitly detected command variant, and enforces a
timeout. TextGrid coverage is evidence only (`DIAGNOSTIC_ONLY`); MFA cannot
turn an uncertain clip into PASS or cause a TTS retry by itself.
