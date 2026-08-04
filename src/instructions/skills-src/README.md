# skills-src/ — fuente única de skills

Cada subcarpeta es una skill: un procedimiento acotado para un problema
concreto del pipeline, no otro prompt maestro. `scripts/build_agent_skills.py`
las copia, byte-idénticas, a `.agents/skills/` (Codex) y `.claude/skills/`
(Claude). **Nunca
se edita directamente en esos dos destinos**: una edición manual ahí se
pierde en la próxima generación, y las dos copias podrían divergir sin que
nadie lo note.

## Skills existentes

| Skill | Fases | Qué decide |
|---|---|---|
| `dubbing-router` | todas | Fase A–K, conjunto mínimo, `X_DIAGNOSIS` o `ALL` |
| `unknown-game-diagnosis` | `X_DIAGNOSIS` | Topología de un motor/formato desconocido |
| `inventory-roundtrip` | A, B | Inventario físico y round-trip de contenedores |
| `text-discovery` | B, C | Localizar texto oficial fuera del lugar obvio |
| `dialogue-mapping` | C | Cadena texto↔evento↔cue↔stream con control positivo |
| `text-policy` | D | TTS/SHORT_TTS_QA/KEEP_ORIGINAL, texto oficial vs. entrega |
| `fmv-audio` | E, G, H | Doblaje de video prerenderizado sin destruir el bed |
| `generation-qa` | G, H, I | Generación, compuertas duras, revisión perceptual |
| `packaging-runtime` | J, K | Empaquetado, deploy, smoke real |

## Estructura de una skill

```text
skills-src/<nombre>/
  SKILL.md    # descripción, triggers, inputs, límites, procedimiento,
              # artefactos de salida, gates afectados, referencias
              # canónicas (por ruta, no copia masiva), validación
  meta.json   # {"name", "phases", "triggers", "description"}
```

`phases: []` en `meta.json` significa "aplica a cualquier fase" (caso de
`dubbing-router`, que decide la ruta antes de saber la fase). Una lista con
fases concretas restringe a esas fases exactas.

## Crear una skill nueva

1. `skills-src/<nombre>/SKILL.md` siguiendo el formato de las existentes.
2. `skills-src/<nombre>/meta.json` con `name` igual al nombre de la carpeta.
3. `python scripts/build_agent_skills.py` para generarla en ambos agentes.
4. `python scripts/build_agent_skills.py --check` para confirmar que no hay
   divergencia.

## Validación

```bash
python scripts/build_agent_skills.py --check
python scripts/test_build_agent_skills.py
```
