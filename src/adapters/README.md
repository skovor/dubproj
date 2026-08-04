# Project adapter template

Put one adapter per game/middleware here. The adapter owns only facts that
cannot be generic: how to enumerate dialogue assets, how to resolve a line to
a physical file/window, how to rebuild the container, and where the mod loader
expects the result. Keep maps/audio/models outside this public source tree.

An adapter must provide evidence for:

1. input inventory and topology;
2. text/event/cue/stream mapping;
3. reference extraction and visual subtitle timebase;
4. container rebuild/roundtrip;
5. runtime paths and smoke procedure.

Do not fill fields from memory. Use `SIN EVIDENCIA TODAVIA` until a real file,
binary, log or official document proves them.
