# Independent language identification

LID is a separate evidence family (for example SpeechBrain VoxLingua107),
recorded with model revision, sample rate, duration, speech ratio and audio
SHA. Short clips, non-linguistic efforts and low-speech windows are
`LID_NOT_APPLICABLE`/`LID_UNCERTAIN`, never an invented language. An English
leak requires concordant Whisper + independent LID + weak target alignment;
strong target alignment yields `EVIDENCE_CONFLICT` for review rather than a
false leak or automatic retry.
