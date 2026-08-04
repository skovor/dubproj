# Cost-aware QA routing

Generation is a sealed cohort and OmniVoice is released before heavy QA. The
router then performs level 0 technical checks, level 1 dual Whisper screening,
level 2 CTC for the provisional winner, level 3 independent LID/CTC for a
second candidate only when useful, and level 4 MFA/performance diagnostics on
request. `ModelPool` keeps one resident instance per role/model/revision/device
and closes it at run end. SQLite transitions make resume idempotent. Routing
never converts uncertainty into PASS or starts a TTS retry.
