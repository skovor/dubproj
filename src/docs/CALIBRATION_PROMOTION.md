# Calibration validation and promotion

Validation thresholds are selected on the calibration/validation groups. The
hidden test is evaluated once per run ID and cannot be used for tuning. The
promotion helper recomputes dataset hashes and refuses any hidden false PASS,
missing lock/provenance, missing artifact, or invalid threshold. A promoted
profile is the only object allowed to set `calibration_authority=true`; no
profile is shipped by this repository because there is no human gold set here.
