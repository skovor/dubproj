# Final benchmark and promotion

`BenchmarkManifest` fixes line/audio/reference lists, model/runtime locks,
calibration profile, config and commit. The report includes elapsed time,
lines/minute, stage timings, blockers and quality rows. Synthetic runs are
explicitly `real_audio=false`; they cannot promote a branch. Promotion also
requires an independent second-game adapter report with valid container and
timing evidence. This repository ships only a template adapter, not a claim of
second-game validation.
