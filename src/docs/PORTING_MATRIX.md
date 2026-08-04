# Production porting matrix

The generic modules are the stable API. `ported/` keeps the full production
implementations used to close the original project so reviewers can compare
behaviour line by line. The table makes the boundary explicit.

| Generic module | Production responsibility preserved | Adapter required |
|---|---|---|
| `policy.py` | TTS / SHORT_TTS_QA / KEEP_ORIGINAL, subtitle-only rule, isolated calls | corpus language and optional call allowlist |
| `mapping.py` | text -> event/cue -> physical stream and FMV visual evidence | engine/middleware inventory |
| `generation.py` | persistent OmniVoice, source reference text, 4+4 FMV rounds, hash cache | TTS backend |
| `timing.py` / `audio.py` | exact frames, onset, +/-0.35 s duration window, safe atempo rescue | ffmpeg path |
| `splice.py` | Empalme B, preserved effort, source resume, crossfade and room tone | none for PCM |
| `qa.py` | hard gates, English-leak confirmation, final-word detector, soft diagnostics | ASR implementation |
| `ported/audit_final_scene_language.py` | continuous post-mount language audit | ASR model/device |
| `package.py` | clean staging, backup, hash, atomic replacement and rollback | runtime path adapter |
| `ported/rebuild_anime_usm.py` | exact CRI USM ADX track replacement | CRI tools / container format |
| `ported/produce_anime_scene.py` | complete OmniVoice scene producer and HTML evidence | reference/project config |
| `ported/produce_in_engine_bank.py` | independent line/bank route | bank/container adapter |

The port intentionally excludes game data, maps, source audio, generated
audio, model weights, URLs containing credentials and absolute workstation
paths from the generic runtime. Those belong in a private project adapter.
