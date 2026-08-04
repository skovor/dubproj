#!/usr/bin/env python3
"""Orquestador de doblaje de P3R: decide y ejecuta línea por línea.

Junta todo lo validado:
  * line_policy.classify_line -> KEEP_ORIGINAL / SHORT_EXTEND_CUT / TTS.
  * empalme de gritos (inicial y final), reutilizando los helpers ya pulidos de
    gen_scream_comparison / gen_trailing_splice (crossfade real, room-tone de
    costura, recorte de silencio de entrada, sin cortes).
  * corrección de longitud SOLO en líneas cinemáticas que desbordan
    (cine_index.json + prod_timing: duration= base, atempo directo de rescate).
  * puntos 6/7 (gritos de palabra alemana y onomatopeyas) ya caen en KEEP_ORIGINAL
    dentro de line_policy.

Por defecto NO genera nada: hace un DRY-RUN que clasifica las 28.157 líneas y
escribe el plan (qué se le hará a cada una). Con --run carga OmniVoice (build
SUKAKU) y produce los .wav. Así se puede auditar el plan sin gastar GPU.

Uso:
  python prod_dub.py                 # dry-run: escribe plan_doblaje.json
  python prod_dub.py --run [--limit N] [--event Main_130_080_C]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
import line_policy as lp
import final_policy_runtime as final_policy
from classify_screams import is_scream          # grito de verdad, no "oh"/"ah" suave

ROOT = Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
CORPUS = ROOT / "corpus_lines.jsonl"
MISC_LINES = ROOT / "misc_lines.jsonl"
BATTLE_LINES = ROOT / "battle_lines.jsonl"
FLDSUPPORT_LINES = ROOT / "fldsupport_lines.jsonl"
POOL_FALLBACK_LINES = ROOT / "pool_fallback_lines.jsonl"
CINE = ROOT / "cine_index.json"
ORDEN = ROOT / "orden_cronologico.json"           # orden_cronologico.py: Main real + resto agrupado
OUT_DIR = ROOT / "produccion"                     # wavs de salida (con --run)
PLAN = ROOT / "plan_doblaje.json"
FFMPEG = Path(r"C:\Users\juand\Desktop\moddeutsch\ffmpeg7"
              r"\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin\ffmpeg.exe")


def _event_short(path: str) -> str:
    return path.rsplit("/", 1)[-1].replace("BMD_", "")


def load_corpus() -> list[dict]:
    """Junta los 4 jsonl (narrativa + misc/NPC-tiendas + battle + fldsupport-
    mazmorra) en una sola lista, con 'event' siempre en forma corta (la
    narrativa ya viene corta; los otros tres traen la ruta BMD completa).
    Antes prod_dub.py solo leia CORPUS (narrativa): NPC/tiendas/batalla/
    mazmorra nunca se generaban. resolve_ref() sabe distinguir cada familia
    por el nombre de evento (ver comentario junto a esa funcion)."""
    recs = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for fn in (MISC_LINES, BATTLE_LINES, FLDSUPPORT_LINES):
        if not fn.exists():
            continue
        for line in fn.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["event"] = _event_short(r["event"])
            recs.append(r)
    # Clips huerfanos de pool compartido (resuelve_pool_fallback.py): ya traen
    # su propio "ref_wav" (el clip exacto, no hay nada que emparejar) y una
    # traduccion propia por MT en vez de la oficial de Atlus -- ver
    # resolve_ref() para el porque. event ya viene corto (PoolFallback_<pool>),
    # no necesita _event_short().
    if POOL_FALLBACK_LINES.exists():
        for line in POOL_FALLBACK_LINES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _sorted_by_chronology(recs: list[dict]) -> list[dict]:
    """Reordena por orden_cronologico.json (Main en orden real de historia,
    el resto agrupado por familia/personaje-zona; ver ese script para el
    porque no existe un orden dia-a-dia real derivable del propio corpus).
    Las lineas dentro de un mismo evento mantienen su stream_index."""
    if not ORDEN.exists():
        return recs
    order = {ev: i for i, ev in enumerate(json.loads(ORDEN.read_text(encoding="utf-8")))}
    return sorted(recs, key=lambda r: (order.get(r["event"], len(order)), r["stream_index"]))

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Palabras alemanas que PARECEN grito pero dicen algo (el inglés dice otra cosa):
# un grito FINAL que es palabra alemana no se empalma desde el audio inglés.
GERMAN_WORD = re.compile(r"^(nein|hilfe|los|was|falsch|ja|bitte|mama|papa|opa|oma|"
                         r"halt|warte|neein|neeein|hiilfe|hiiilfe)$", re.I)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def _collapse(t: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", t)


def strip_first_word(text: str) -> str:
    """Quita el primer token (el grito) y la puntuación pegada a él."""
    m = re.match(r"^\s*[^\w]*[^\W\d_]+\s*(?:[.!?,…]+|\.{2,})?\s*(.*)$", text, re.S | re.U)
    return (m.group(1).strip() if m else text).strip()


def strip_tail(text: str) -> str:
    return re.sub(r"[\s.,…!?\"'-]*[^\W\d_]+[\s.!?…\"'-]*$", "", text,
                  flags=re.UNICODE).strip()


def _qa_edit_distance(a: list[str], b: list[str]) -> int:
    """Distancia de edicion a nivel palabra (Levenshtein), sin depender de
    qa_tomas_crudas.py (que arrastra librosa/pyannote solo por esta funcion)."""
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        nxt = [i]
        for j, right in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = nxt
    return row[-1]


def plan_line(rec: dict, cine: dict) -> dict:
    """Decide qué se le hace a la línea, sin generar audio."""
    en = rec.get("text_en") or ""
    de = rec.get("text_de") or ""
    ev, si = rec["event"], rec["stream_index"]
    # build_cine_index.py ahora tambien indexa corpus_msg_battle/_fldsupport
    # linea por linea (vp sin bup = cinematica, igual criterio que narrativa).
    # Verificado a mano: BtlSupportFuka/Common y FldSupport_DungeonTalk_Common
    # son 100% cinematicas (se cortan sin caja que espere) pero
    # FldSupport_Tactics es 100% caja -- una suposicion a nivel de familia
    # entera habria sido incorrecta para esa ultima, por eso se verifica linea
    # por linea en vez de asumir por nombre de evento.
    is_cine = bool(cine.get(ev, {}).get(str(si), False))

    reviewed = final_policy.review_for(ev, si)
    if reviewed and reviewed.get("policy_ready") != "1":
        return {
            "event": ev,
            "stream": si,
            "cine": is_cine,
            "policy": "BLOCKED_REVIEW",
            "reason": reviewed.get("generation_blocker") or "policy_not_ready",
            "de": de,
            "accion": "bloqueado_revision",
        }

    if reviewed:
        reviewed_action = final_policy.reviewed_action(reviewed)
        reviewed_text = final_policy.reviewed_delivery(reviewed)
        montage = reviewed.get("montage_hint") or ""
        plan = {
            "event": ev,
            "stream": si,
            "cine": is_cine,
            "policy": reviewed_action,
            "reason": reviewed.get("reason_final") or "final_review",
            "de": reviewed_text,
            "reviewed": True,
            "montage_hint": montage,
            "preserve_original_component": (
                reviewed.get("preserve_original_component") or ""
            ),
        }
        # FULL_KEEP_ORIGINAL is authoritative for acoustic/nonlexical
        # preservation even when a text-only classifier disagreed.
        if reviewed_action in {"conservar_original", "excluir_no_hablado"} or (
            montage == "FULL_KEEP_ORIGINAL"
        ):
            plan["accion"] = "conservar_original"
            return plan
        if montage == "EMPALME_B_PREFIX":
            plan["accion"] = "empalme_inicial"
            plan["frase"] = reviewed_text
            return plan
        if montage == "EMPALME_B_SUFFIX":
            plan["accion"] = "empalme_final"
            plan["frase"] = reviewed_text
            return plan
        if montage == "EMPALME_B_BOTH":
            plan["accion"] = "bloqueado_revision"
            plan["reason"] = "empalme_doble_requiere_segmentacion_acustica"
            return plan
        plan["accion"] = (
            "tts_corto_qa"
            if reviewed_action == "tts_corto_qa"
            else "tts"
        )
        plan["corrige_timing"] = is_cine
        return plan

    d = lp.classify_line(en, de)
    plan = {"event": ev, "stream": si, "cine": is_cine,
            "policy": d.action, "reason": d.reason, "de": de}

    if d.action == lp.KEEP_ORIGINAL:
        plan["accion"] = "conservar_original"
        return plan
    if not de.strip():
        plan["accion"] = "conservar_original"
        plan["reason"] = "sin_texto_de"
        return plan

    # ¿empalme? Solo si la línea lleva un GRITO de verdad (no "oh"/"ah" suave,
    # que OmniVoice ya saca bien) Y el inglés también grita ahí (el empalme toma
    # el grito del audio inglés; si el inglés no grita, no hay de dónde sacarlo).
    de_toks = WORD.findall(de)
    en_toks = WORD.findall(en)
    en_has_scream = any(is_scream(t.lower(), t) for t in en_toks)
    if (len(de_toks) >= 2 and is_scream(de_toks[0].lower(), de_toks[0])
            and en_has_scream
            and not GERMAN_WORD.match(_collapse(_fold(de_toks[0])))):
        rest = strip_first_word(de)
        if len(WORD.findall(rest)) >= 1:
            plan["accion"] = "empalme_inicial"
            plan["frase"] = rest
            return plan
    if (len(de_toks) >= 3 and is_scream(de_toks[-1].lower(), de_toks[-1])
            and en_has_scream
            and not GERMAN_WORD.match(_collapse(_fold(de_toks[-1])))):
        rest_t = strip_tail(de)
        if len(WORD.findall(rest_t)) >= 2:
            plan["accion"] = "empalme_final"
            plan["frase"] = rest_t
            return plan

    # síntesis normal; corrección de longitud solo si es cinemática
    if d.action == lp.SHORT_TTS_QA:
        plan["accion"] = "tts_corto_qa"   # plana + revisión "oder" + regenerar
    else:
        plan["accion"] = "tts"
    plan["corrige_timing"] = is_cine
    return plan


def dry_run(limit: int | None) -> None:
    cine = json.loads(CINE.read_text(encoding="utf-8"))
    recs = _sorted_by_chronology(load_corpus())
    plans = []
    tally: dict[str, int] = {}
    cine_tts = 0
    for i, rec in enumerate(recs):
        if limit and i >= limit:
            break
        p = plan_line(rec, cine)
        plans.append(p)
        tally[p["accion"]] = tally.get(p["accion"], 0) + 1
        if p["accion"] in ("tts", "tts_corto_qa") and p.get("corrige_timing"):
            cine_tts += 1
    PLAN.write_text(json.dumps(plans, ensure_ascii=False), encoding="utf-8")
    print(f"plan de {len(plans)} lineas -> {PLAN.name}\n")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v:>7}")
    print(f"\n  de esas, TTS cinematicas que pasan por corrector de timing: {cine_tts}")


# --------------------------------------------------------------------------
# Ejecución real (--run): carga OmniVoice (SUKAKU) y genera. Imports pesados
# aquí dentro para que el dry-run no toque torch.
# --------------------------------------------------------------------------
def run(
    limit: int | None,
    only_event: str | None,
    only_ids: set[str] | None = None,
    reference_overrides: dict[str, str] | None = None,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    from gen_scream_comparison import (  # helpers de empalme ya pulidos
        build_leading_splice, build_trailing_splice, OmniVoice,
        OmniVoiceGenerationConfig, energy_end, speech_resume, trim_lead_silence,
    )
    from score_screams_acoustic import lead_measure, tail_measure, scream_score  # P5
    import prod_timing as pt

    # P5: el texto solo no distingue un grito real de una exclamacion suave
    # ("Oooh, he did it!" se escribe igual que "Gah!"). Antes de empalmar se
    # confirma con el AUDIO de referencia: si el score acustico (loudness +
    # pitch del arranque/final relativo al resto de la linea) queda por debajo
    # del umbral, se hace TTS normal en vez de empalmar. Umbral fijado con
    # datos reales (screams_acoustic.json): los gritos de verdad (Gah, Argh,
    # Gyaaaaaah, Aaaaaargh, Ughhh) puntuan 2-3; solo una exclamacion suave
    # ("Ohhh, come on, man...") bajo a 1.
    SCREAM_SCORE_MIN = 2
    downgraded: list[dict] = []

    # QA local de señal + cola (regla "oder"). Whisper no se carga dentro del
    # bucle de síntesis: el contenido/referencia se valida antes de generar.
    # Para cue/VN/in-engine/battle normal, ASR no participa después ni como
    # gate, score o desempate; la decisión es sólo acústica y contractual.
    # Toda línea TTS, corta o larga, pasa por qa_tail_clean() cuando termina con
    # energía alta. La regla no depende de la longitud del texto.
    QA_MAX_ATTEMPTS = 4
    QA_TAIL_GUARD_S = 0.08
    qa_reports: list[dict] = []

    cine = json.loads(CINE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda",
                                      dtype=torch.float16)
    cfg = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=1.0,
        postprocess_output=False,
    )
    sr_m = model.sampling_rate

    def gen(text_de, ref, ref_text, duration=None):
        kw = {"text": [text_de], "language": ["de"], "ref_audio": [ref],
              "ref_text": [ref_text], "generation_config": cfg}
        if duration is not None:
            kw["duration"] = [duration]
        arr = model.generate(**kw)[0]
        t = arr.detach().cpu() if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
        return t.squeeze().numpy().astype(np.float64)

    def read_ref(rec):
        """Resuelve el audio de referencia EN. resolve_ref() debe apuntar al AWB
        del evento / workspace; ver punto 8 (extraer AWB de bancos restantes)."""
        stem = f"{rec['event']}_L{rec['stream_index']:03d}"
        p = (
            reference_overrides.get(stem)
            if reference_overrides and stem in reference_overrides
            else resolve_ref(rec)
        )
        if not p or not Path(p).exists():
            return None, None, None
        y, sr = sf.read(str(p), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return np.asarray(y, float), sr, str(p)

    def qa_tail_clean(cand: "np.ndarray") -> bool:
        n = int(QA_TAIL_GUARD_S * sr_m)
        if len(cand) == 0:
            return False
        if len(cand) < n or n <= 0:
            return bool(float(np.abs(cand).max()) > 1e-6)
        peak = float(np.abs(cand).max())
        if peak <= 1e-6:
            return False
        tail_rms = float(np.sqrt(np.mean(cand[-n:].astype(np.float64) ** 2)))
        return tail_rms < 0.10 * peak

    def qa_regen_content_or_tail(audio0, synth_fn, expected_text: str):
        """Ver comentario junto a QA_MAX_ATTEMPTS. Devuelve (audio_elegido,
        reporte | None); el reporte solo se rellena si hizo falta mirar mas de
        un intento (para no llenar el JSON de miles de lineas que pasaron a la
        primera)."""
        expected = lp.words(expected_text)
        if not expected:
            return audio0, None
        def evaluate(cand):
            tail_ok = qa_tail_clean(cand)
            peak = float(np.abs(cand).max()) if len(cand) > 0 else 0.0
            not_empty = peak > 1e-5
            clipped = np.abs(cand) >= 0.999
            clipping_pct = (
                float(np.mean(clipped) * 100.0) if len(cand) else 100.0
            )
            longest_clipped_run = 0
            current_run = 0
            for value in clipped:
                if value:
                    current_run += 1
                    longest_clipped_run = max(
                        longest_clipped_run, current_run
                    )
                else:
                    current_run = 0
            not_clipped = (
                clipping_pct <= 0.15 and longest_clipped_run < 8
            )

            n = int(QA_TAIL_GUARD_S * sr_m)
            tail_rms = float(np.sqrt(np.mean(cand[-n:].astype(np.float64) ** 2))) if (len(cand) >= n and n > 0) else 0.0
            tail_ratio = (tail_rms / max(peak, 1e-6)) if peak > 1e-6 else 1.0

            return {
                "tail_ok": tail_ok,
                "not_empty": not_empty,
                "not_clipped": not_clipped,
                "clipping_pct": clipping_pct,
                "longest_clipped_run": longest_clipped_run,
                "tail_ratio": tail_ratio,
                "peak": peak,
                "passed": bool(tail_ok and not_empty and not_clipped)
            }

        candidates = [(audio0, evaluate(audio0))]
        attempt = 1
        while not candidates[-1][1]["passed"] and attempt < QA_MAX_ATTEMPTS:
            attempt += 1
            cand = synth_fn()
            candidates.append((cand, evaluate(cand)))

        if candidates[-1][1]["passed"] and len(candidates) == 1:
            return candidates[0][0], None
        if candidates[-1][1]["passed"]:
            winner_audio, winner_info = candidates[-1]
        else:
            # Se conserva el candidato técnicamente menos malo únicamente para
            # revisión. El caller NO debe promoverlo a produccion/.
            winner_audio, winner_info = min(
                candidates,
                key=lambda c: (not c[1]["passed"], c[1]["tail_ratio"]),
            )
        report = {"passed": winner_info["passed"],
                  "attempts": [{"attempt": i + 1, **c[1]} for i, c in enumerate(candidates)]}
        return winner_audio, report

    recs = _sorted_by_chronology(load_corpus())
    if only_event:
        recs = [r for r in recs if r["event"] == only_event]
    if only_ids is not None:
        recs = [
            r for r in recs
            if f"{r['event']}_L{r['stream_index']:03d}" in only_ids
        ]
    total_recs = len(recs)
    existing_count = sum(1 for r in recs if (OUT_DIR / f"{r['event']}_L{r['stream_index']:03d}.wav").exists())
    print(f"Corpus cargado: {total_recs} lineas totales ({existing_count} ya generadas previamente)\n")

    done = 0
    for idx, rec in enumerate(recs, 1):
        if limit and done >= limit:
            break
        stem = f"{rec['event']}_L{rec['stream_index']:03d}"
        out = OUT_DIR / f"{stem}.wav"

        pct = (idx / total_recs) * 100
        bar_len = 30
        filled = int(bar_len * idx // total_recs)
        bar = "=" * filled + (">" if filled < bar_len else "") + "-" * (bar_len - filled - (1 if filled < bar_len else 0))

        if out.exists():
            if idx % 500 == 0 or idx == 1 or idx == total_recs:
                sys.stdout.write(f"\r[{idx:>5}/{total_recs:<5}] [{pct:5.1f}%] [{bar}] Reanudando... saltando ya generadas ({done} creadas en esta sesion)   ")
                sys.stdout.flush()
            continue

        p = plan_line(rec, cine)
        if p["accion"] == "bloqueado_revision":
            continue
        y, sr, refp = read_ref(rec)
        if y is None:
            continue
        acc = p["accion"]

        sys.stdout.write(f"\r[{idx:>5}/{total_recs:<5}] [{pct:5.1f}%] [{bar}] Generando {stem} ({acc})...                           ")
        sys.stdout.flush()

        if acc in ("empalme_inicial", "empalme_final"):
            measure = (lead_measure if acc == "empalme_inicial" else tail_measure)(Path(refp))
            score = scream_score(measure) if measure else -1
            # el texto ya exigio que el aleman grite Y el ingles tambien (ver
            # plan_line); esto solo confirma con el audio que ese grito ingles
            # es de verdad intenso y no una exclamacion suave mal detectada.
            if score < SCREAM_SCORE_MIN and not p.get("reviewed"):
                downgraded.append({"stem": stem, "accion_original": acc,
                                    "score": score, "de": p.get("frase"),
                                    "en": rec["text_en"]})
                acc = "tts"
        try:
            if acc == "conservar_original":
                sf.write(str(out), y, sr)
            elif acc == "empalme_inicial":
                b = energy_end(y, sr, min(1.2, len(y) / sr * 0.6))
                cut = speech_resume(y, sr, b)
                raw_body = gen(p["frase"], refp, rec["text_en"])
                # Empalme B/V7 aprobado en escucha: ajuste localizado, 35 ms
                # equal-gain y pausa mínima 30 ms. Se aplica transversalmente
                # a VN/in-engine, no solo a películas prerenderizadas.
                spl, seam, _ = build_leading_splice(
                    y, sr, raw_body, sr_m, b, cut,
                    min_pause_seconds=0.030,
                    crossfade_seconds=0.035,
                    crossfade_curve="equal_gain",
                )
                sf.write(str(out), spl, sr_m)
            elif acc == "empalme_final":
                raw_body = gen(p["frase"], refp, rec["text_en"])
                spl, _seam, _ = build_trailing_splice(
                    y, sr, raw_body, sr_m,
                    min_pause_seconds=0.030,
                    crossfade_seconds=0.035,
                    crossfade_curve="equal_gain",
                )
                sf.write(str(out), spl, sr_m)
            else:  # tts / tts_corto_qa
                text = p.get("prompt", p.get("de", rec["text_de"]))

                def synth():
                    # regla 3: recorte de silencio artificial al inicio. Antes solo
                    # se aplicaba dentro de la rama corrige_timing (cinematicas);
                    # el resto de lineas (caja de dialogo / VN, la mayoria del
                    # corpus) nunca pasaban por trim_lead_silence. El silencio
                    # inventado por el modelo es un artefacto de generacion, no
                    # algo que dependa de si la linea es cinematica o no.
                    if p.get("corrige_timing"):
                        t_en = len(y) / sr
                        a = gen(text, refp, rec["text_en"], duration=round(t_en, 3))
                        a = trim_lead_silence(a, sr_m)
                        # Regla cinematográfica confirmada: después de duration
                        # nativo, llevar el final de voz a la ventana simétrica
                        # ±0.35 s mediante atempo. Solo se ejecuta para vp sin
                        # bup; VN/caja nunca entra aquí. El QA posterior decide
                        # si la transformación sigue siendo natural.
                        a, _ = pt.correct_length(
                            a, sr_m, pt.speech_end(y, sr), FFMPEG, tmp,
                        )
                    else:
                        a = gen(text, refp, rec["text_en"])
                        a = trim_lead_silence(a, sr_m)
                    return a

                audio = synth()
                # Regla "oder" GLOBAL: cualquier TTS con cola abrupta entra a
                # QA/regeneracion, aunque sea una frase larga.
                if True:  # Toda salida TTS pasa QA, incluso arrays vacíos.
                    audio, qa_info = qa_regen_content_or_tail(audio, synth, text)
                    if qa_info is not None:
                        qa_reports.append({"stem": stem, **qa_info})
                        if not qa_info["passed"]:
                            review_dir = OUT_DIR / "_review_failed"
                            review_dir.mkdir(exist_ok=True)
                            review_wav = review_dir / f"{stem}.wav"
                            sf.write(str(review_wav), audio, sr_m)
                            qa_reports[-1]["review_wav"] = str(review_wav)
                            print(f"\n  CODEX_REVIEW {stem}: no pasó QA tras "
                                  f"{QA_MAX_ATTEMPTS} intentos")
                            continue
                if acc == "tts_corto_con_corte":
                    audio = cut_after_words(audio, sr_m, p["cut_after_words"])
                sf.write(str(out), audio, sr_m)
            done += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"\n  ERROR {stem} ({acc}): {str(exc)[:120]}")
    sys.stdout.write("\n")
    if downgraded:
        (OUT_DIR / "empalmes_degradados.json").write_text(
            json.dumps(downgraded, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {len(downgraded)} empalmes degradados a TTS por score acustico bajo "
              f"(< {SCREAM_SCORE_MIN}) -> {OUT_DIR / 'empalmes_degradados.json'}")
    if qa_reports:
        sin_pasar = [r for r in qa_reports if not r["passed"]]
        (OUT_DIR / "qa_corto_regeneraciones.json").write_text(
            json.dumps(qa_reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {len(qa_reports)} lineas necesitaron mas de un intento "
              f"(de las cuales {len(sin_pasar)} no pasaron el QA tras "
              f"{QA_MAX_ATTEMPTS} intentos -> revisar a mano) "
              f"-> {OUT_DIR / 'qa_corto_regeneraciones.json'}")
    print(f"generadas {done} lineas -> {OUT_DIR}")


# Punto 8 (extracción general de AWB), resuelto: awb_index.py mapea evento ->
# banco de voz .awb para los 825 eventos Main/Cmmu/Extr/Fild/Qest (patrón
# Voice_Event_<event>.awb bajo Stream/en/ en pakchunk4/5; ver docstring de
# awb_index.py para los esquemas de nombre AUN no cubiertos: Voice_NPC_*,
# Voice_Facility_*, bancos de batalla). El audio nunca sale de retoc (solo
# toca IoStore): se extrae con repak.exe sobre el .pak legacy, y CADA subsong
# se vuelca con vgmstream. repak/vgmstream se invocan siempre con un handle
# binario de Python (nunca `>` de PowerShell: corrompe el .awb en silencio,
# confirmado -- el tamaño extraído difería y vgmstream no podía abrirlo).
AWB_INDEX_PATH = ROOT / "awb_index.json"
AWB_CACHE = ROOT / "awb_cache"
AWB_MISMATCH_LOG = ROOT / "awb_stream_count_mismatches.json"
REPAK = Path(r"C:\Users\juand\Desktop\moddeutsch\repak\repak.exe")
VGMSTREAM = Path(r"C:\Users\juand\Desktop\moddeutsch\vgmstream\vgmstream-cli.exe")
AES_KEY = "0x92BADFE2921B376069D3DE8541696D230BA06B5E4320084DD34A26D117D2FFEE"

_awb_index: dict | None = None
_awb_voiced_counts: dict | None = None
_awb_mismatches: list[dict] | None = None


def _load_awb_index() -> dict:
    global _awb_index
    if _awb_index is None:
        _awb_index = (json.loads(AWB_INDEX_PATH.read_text(encoding="utf-8"))
                       if AWB_INDEX_PATH.exists() else {})
    return _awb_index


def _load_voiced_counts() -> dict:
    """Cuántas líneas de voz tiene cada evento según el texto: la compuerta
    obligatoria n_voiced == stream_count del AWB, pendiente desde el hallazgo
    de 100_180_C (CLAUDE.md), nunca antes aplicada de forma general. Debe
    usar el corpus FUSIONADO (load_corpus, no solo CORPUS/narrativa): si no,
    eventos de battle_lines.jsonl como BtlEvent700 siempre dan expected=0 y
    la compuerta los rechaza aunque el banco SÍ alinee (bug encontrado al
    probar BtlEvent700: resolve_ref devolvía None con el banco bien extraído)."""
    global _awb_voiced_counts
    if _awb_voiced_counts is None:
        counts: dict[str, int] = {}
        for r in load_corpus():
            counts[r["event"]] = counts.get(r["event"], 0) + 1
        _awb_voiced_counts = counts
    return _awb_voiced_counts


def _record_mismatch(entry: dict) -> None:
    global _awb_mismatches
    if _awb_mismatches is None:
        _awb_mismatches = (json.loads(AWB_MISMATCH_LOG.read_text(encoding="utf-8"))
                           if AWB_MISMATCH_LOG.exists() else [])
    if not any(m["event"] == entry["event"] for m in _awb_mismatches):
        _awb_mismatches.append(entry)
        AWB_MISMATCH_LOG.write_text(json.dumps(_awb_mismatches, ensure_ascii=False, indent=1),
                                     encoding="utf-8")


def _extract_awb_streams(event: str, pak_entry: dict) -> Path | None:
    """Extrae (si hace falta) TODOS los streams del banco de voz de `event` a
    awb_cache/<event>/sN.wav, y aplica la compuerta n_voiced == stream_count.

    Nunca se asume un desplazamiento cuando no coincide (170_170_A y 100_180_C
    ya demostraron que la regla secuencial 1:1 falla a veces): si no coincide,
    NO se generan líneas de ese evento y queda registrado en
    awb_stream_count_mismatches.json para revisión/alineación manual, en vez
    de arriesgar un doblaje desalineado en silencio."""
    import subprocess

    cache_dir = AWB_CACHE / event
    marker = cache_dir / "_estado.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        return cache_dir if info.get("aligned") else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    awb_path = cache_dir / f"{event}.awb"
    with awb_path.open("wb") as fh:
        r = subprocess.run([str(REPAK), "-a", AES_KEY, "get",
                            pak_entry["pak"], pak_entry["internal"]],
                           stdout=fh, stderr=subprocess.PIPE)
    if r.returncode != 0 or awb_path.stat().st_size == 0:
        marker.write_text(json.dumps({"aligned": False, "reason": "repak_failed"}),
                           encoding="utf-8")
        return None

    subprocess.run([str(VGMSTREAM), "-s", "1", "-S", "0",
                    "-o", str(cache_dir / "s?s.wav"), str(awb_path)],
                   capture_output=True)
    awb_path.unlink(missing_ok=True)   # solo hacía falta para la extracción

    stream_count = len(list(cache_dir.glob("s*.wav")))
    expected = _load_voiced_counts().get(event, 0)
    aligned = stream_count > 0 and stream_count == expected
    marker.write_text(json.dumps({"aligned": aligned, "stream_count": stream_count,
                                   "expected_voiced_lines": expected}, ensure_ascii=False),
                       encoding="utf-8")
    if not aligned:
        _record_mismatch({"event": event, "stream_count": stream_count,
                           "expected_voiced_lines": expected})
        return None
    return cache_dir


AWB_ALIGNMENT_FIXES_PATH = ROOT / "awb_alignment_fixes.json"
_awb_alignment_fixes: dict | None = None


def _load_alignment_fixes() -> dict:
    """event -> {stream_index_texto: stream_index_audio|null}, resuelto por
    transcripción (resuelve_alineacion_awb.py) para los eventos donde
    n_voiced != stream_count. Ver ese script: alineamiento Needleman-Wunsch
    contra Whisper, nunca posición asumida."""
    global _awb_alignment_fixes
    if _awb_alignment_fixes is None:
        _awb_alignment_fixes = (json.loads(AWB_ALIGNMENT_FIXES_PATH.read_text(encoding="utf-8"))
                                 if AWB_ALIGNMENT_FIXES_PATH.exists() else {})
    return _awb_alignment_fixes


BTLEVENT_INDEX_PATH = ROOT / "awb_index_btlevent.json"
BTLEVENT_CACHE = ROOT / "awb_cache_btlevent"
_btlevent_index: dict | None = None


def _load_btlevent_index() -> dict:
    global _btlevent_index
    if _btlevent_index is None:
        _btlevent_index = (json.loads(BTLEVENT_INDEX_PATH.read_text(encoding="utf-8"))
                            if BTLEVENT_INDEX_PATH.exists() else {})
    return _btlevent_index


def _extract_btlevent_streams(event: str, pak_entry: dict) -> Path | None:
    """Igual que _extract_awb_streams (misma compuerta n_voiced==stream_count)
    pero en su propia cache: BtlEvent SÍ mantiene 1 banco = 1 evento (ver
    awb_index_btlevent.py), a diferencia de CmmuNPC/Facility/Dungeon."""
    import subprocess

    cache_dir = BTLEVENT_CACHE / event
    marker = cache_dir / "_estado.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        return cache_dir if info.get("aligned") else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    awb_path = cache_dir / f"{event}.awb"
    with awb_path.open("wb") as fh:
        r = subprocess.run([str(REPAK), "-a", AES_KEY, "get",
                            pak_entry["pak"], pak_entry["internal"]],
                           stdout=fh, stderr=subprocess.PIPE)
    if r.returncode != 0 or awb_path.stat().st_size == 0:
        marker.write_text(json.dumps({"aligned": False, "reason": "repak_failed"}), encoding="utf-8")
        return None

    subprocess.run([str(VGMSTREAM), "-s", "1", "-S", "0",
                    "-o", str(cache_dir / "s?s.wav"), str(awb_path)], capture_output=True)
    awb_path.unlink(missing_ok=True)

    stream_count = len(list(cache_dir.glob("s*.wav")))
    expected = _load_voiced_counts().get(event, 0)
    aligned = stream_count > 0 and stream_count == expected
    marker.write_text(json.dumps({"aligned": aligned, "stream_count": stream_count,
                                   "expected_voiced_lines": expected}, ensure_ascii=False),
                       encoding="utf-8")
    if not aligned:
        return None
    return cache_dir


POOL_RESOLUTION_PATH = ROOT / "pool_resolution.json"
POOL_CACHE = ROOT / "pool_cache"
_pool_resolution: dict | None = None

# Mismo mapeo bank-suffix -> substring de evento usado en resuelve_pool_compartido.py
FACILITY_NAME_MAP = {"velvetroom": "VelvetRoom", "item": "ItemShop", "weapon": "WeaponShop",
                     "antique": "AntiqueShop", "request": "Request"}


def _load_pool_resolution() -> dict:
    global _pool_resolution
    if _pool_resolution is None:
        _pool_resolution = (json.loads(POOL_RESOLUTION_PATH.read_text(encoding="utf-8"))
                            if POOL_RESOLUTION_PATH.exists() else {})
    return _pool_resolution


def _pool_key_for(event: str) -> str | None:
    m = re.match(r"^CmmuNPC_(\d+)_", event)
    if m:
        return f"npc_{m.group(1)}"
    for suffix, needle in FACILITY_NAME_MAP.items():
        if needle.lower() in event.lower():
            return f"facility_{suffix}"
    if event.startswith("FldSupport_") or event.startswith("BtlSupport"):
        return "dungeon"
    return None


def _norm_pool_text(t: str) -> str:
    return re.sub(r"\s+", " ", _fold(t or "")).strip()


def resolve_ref_pool(rec) -> str | None:
    """CmmuNPC/Facility/Dungeon(+BtlSupport): pool de audio COMPARTIDO entre
    muchas lineas de texto, no 1 banco por evento (ver
    resuelve_pool_compartido.py: 4 bancos de Yukari solo cubrian 11/143
    textos). Se resuelve por MEJOR TEXTO (transcripción), no por posición.
    Si el texto no aparece resuelto en pool_resolution.json, la línea se deja
    SIN audio -- decisión del usuario: si el original en inglés tampoco lo
    dobla (silencio), el alemán tampoco necesita voz ahí."""
    key = _pool_key_for(rec["event"])
    if key is None:
        return None
    pool = _load_pool_resolution().get(key)
    if not pool or "_error" in pool:
        return None
    entry = pool.get(_norm_pool_text(rec.get("text_en") or ""))
    if not entry:
        return None
    p = POOL_CACHE / key / entry["wav"]
    return str(p) if p.exists() else None


def resolve_ref(rec):
    """Devuelve la ruta del audio EN de referencia para la línea, extraído de
    su banco .awb (ver comentario arriba). Si el evento alineó limpio
    (n_voiced == stream_count) usa la posición directa; si no, consulta el
    mapeo verificado por transcripción en awb_alignment_fixes.json. Devuelve
    None si el evento no tiene banco indexado, o si no hay una pareja
    confiable para esta línea concreta (mejor no generar que desalinear).

    Despacha por familia: narrativa (awb_index.json, 1:1) -> BtlEvent
    (awb_index_btlevent.json, también 1:1) -> pool compartido CmmuNPC/
    Facility/Dungeon (resolve_ref_pool, mejor-emparejamiento por texto) ->
    fallback de clip huérfano (ref_wav ya conocido, ver resuelve_pool_fallback.py)."""
    if rec["event"].startswith("PoolFallback_"):
        p = rec.get("ref_wav")
        return p if p and Path(p).exists() else None

    entry = _load_awb_index().get(rec["event"])
    if entry is not None:
        cache_dir = _extract_awb_streams(rec["event"], entry)
        if cache_dir is not None:
            p = cache_dir / f"s{rec['stream_index']}.wav"
            return str(p) if p.exists() else None

        fix = _load_alignment_fixes().get(rec["event"])
        if not fix:
            return None
        audio_idx = fix.get(str(rec["stream_index"]))
        if audio_idx is None:
            return None
        p = AWB_CACHE / rec["event"] / f"s{audio_idx}.wav"
        return str(p) if p.exists() else None

    entry_b = _load_btlevent_index().get(rec["event"])
    if entry_b is not None:
        cache_dir = _extract_btlevent_streams(rec["event"], entry_b)
        if cache_dir is None:
            return None
        p = cache_dir / f"s{rec['stream_index']}.wav"
        return str(p) if p.exists() else None

    return resolve_ref_pool(rec)


def cut_after_words(audio, sr, n_words):
    """PENDIENTE: cortar tras la palabra n usando timestamps de Whisper sobre la
    salida alemana (SHORT_EXTEND_CUT genera prompt ampliado y se corta). De
    momento devuelve el audio sin cortar para no romper la ejecución."""
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="genera de verdad (carga GPU)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--event", default=None, help="solo este evento (con --run)")
    ap.add_argument(
        "--ids-file",
        type=Path,
        help="lista UTF-8 de stems event_LNNN autorizados para esta ejecucion",
    )
    ap.add_argument(
        "--reference-manifest",
        type=Path,
        help="JSONL maestro con unit_id y reference_wav exactos",
    )
    args = ap.parse_args()
    if args.run:
        only_ids = None
        if args.ids_file:
            only_ids = {
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            if not only_ids:
                raise ValueError("--ids-file no contiene stems")
        reference_overrides = None
        if args.reference_manifest:
            manifest_rows = [
                json.loads(line)
                for line in args.reference_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            reference_overrides = {
                row["unit_id"].removeprefix("VU_"): row["reference_wav"]
                for row in manifest_rows
                if row.get("reference_wav")
            }
        run(args.limit, args.event, only_ids, reference_overrides)
    else:
        dry_run(args.limit)


if __name__ == "__main__":
    main()
