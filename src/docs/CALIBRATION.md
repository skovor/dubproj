# Calibration (Commit 3)

`dubbing_pipeline.calibration` freezes the `char-alignment-v2` target and
final-anchor feature order and exports only JSON `DRAFT` Platt artifacts. The
trainer refuses `hidden_test`, missing classes, or non-finite values. The
training fixture in the unit tests is synthetic and is not a production gold
set; no artifact is enabled in the default config. A human-labelled manifest
must be validated by `goldset.py` before this script is run.
