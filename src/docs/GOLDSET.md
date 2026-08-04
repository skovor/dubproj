# Human gold set

`goldset.py` creates an auditable SQLite review queue and JSONL export. It does
not label clips. Reviewers must listen independently, label A/B without seeing
ASR/CTC/LID scores, and adjudicate disagreements. Splits are deterministic and
grouped by scene/source line; a candidate variant cannot cross a split.

The calibration gate remains closed until the exported manifest has two
independent reviewers per clip, adjudications, valid audio, and a sealed hidden
test. `UNDECIDABLE` is a valid human outcome and is never coerced.
