# FMV local scene repair

FMV selection is greedy/local and deterministic: candidates are ranked per
line, mounted from the source scene, and only bounded one-line substitutions
are attempted. The old Cartesian product is removed; no hidden exponential
fallback remains. The report includes a line/candidate matrix, blockers and
recommended causal actions. A scene PASS still requires protected intervals,
untouched channels, container/serialization QA and every subtitle-owned line.
