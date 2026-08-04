#!/usr/bin/env python3
"""Política determinista por línea para el doblaje de P3R.

No usa un LLM: decide qué conservar, qué sintetizar normalmente y qué frase
corta debe generarse con contexto auxiliar para luego cortarse por palabra.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

KEEP_ORIGINAL = "KEEP_ORIGINAL"
BLOCKED = "BLOCKED"                 # mapping/reference correction required
SHORT_EXTEND_CUT = "SHORT_EXTEND_CUT"     # (obsoleto: prompt ampliado + corte)
SHORT_TTS_QA = "SHORT_TTS_QA"             # frase corta: TTS normal + revisión/regenerar
TTS = "TTS"
SKIP_MUSIC_MOVIE = "SKIP_MUSIC_MOVIE"

EFFORTS = {
    "ah", "aah", "ahh", "agh", "argh", "augh", "eh", "gah", "gugh",
    "ha", "hah", "heh", "hm", "hmm", "mm", "hng", "hngh", "huh", "ngh", "oh",
    "ooh", "oof", "ow", "ugh", "uh", "uhh", "uff", "urgh",
    "um", "erm", "ahm",
    # Auditado sobre las 28.157 lineas del juego (audit_keep_original.py):
    # 322 lineas 100% vocales acababan en SHORT_EXTEND_CUT o TTS, es decir, el
    # modelo intentaba *actuar* un ladrido o un sollozo. Familias que faltaban:
    # perro y gato (Koromaru tiene decenas de lineas)
    "wuff", "wuf", "wau", "rawuff", "rawuf", "arf", "raff", "ruff", "woof", "bark",
    "japs", "miau", "meow", "winsel",
    # aullido de Koromaru ("Awoo!"/"Ahuu!" corto no llega a 3 letras iguales
    # seguidas, asi que is_shouted_or_onomatopoeia lo rechazaba por token aunque
    # fuera parte de un aullido largo -- Cmmu_009_030_C#10 "Awoo! Awoooo!" /
    # "Ahuu! Ahuuuu!"): raiz colapsada explicita, sin depender del alargamiento.
    "ahu", "awu", "wu", "awo", "owo",
    # gritos de batalla y de dolor
    "gyah", "gya", "hra", "hrah", "hrgh", "hrghgh", "wah", "wa", "uah",
    "oah", "au", "aua", "autsch", "aie", "aiee", "iip", "eep", "ih", "iih",
    "ungh", "nng", "urg", "arg",
    # kiai / esfuerzo fisico al golpear (Cmmu_003_025_C#52 "Hyah!" se coló como
    # TTS por no estar aqui, pese a que classify_screams.SCREAM_STEMS ya cubre
    # la familia ya/yah -- se iguala la cobertura entre los dos sistemas)
    "ya", "yah", "yaah", "hyah", "hiyah", "hiah", "haa",
    # risa, llanto, respiracion, disgusto
    "hehe", "hihi", "hoho", "huhu", "buhu", "haha", "schluchz", "snif",
    "sniff", "seufz", "keuch", "stohn", "achz", "rulps", "hicks",
    "tsk", "tch", "tz", "psst", "sh", "shh", "pff", "pf", "puh", "phew",
    "hmpf", "grr", "gr", "gnn", "igitt", "bah", "pah",
    # sorpresa
    "woah", "whoa", "boah", "oha", "huch", "hui", "ups", "oje", "och",
    "ach", "aha", "oho", "mh", "mhm", "mmm", "ahem",
    "wuhu", "juhu", "juchu", "yay", "wuhuu",
    # saludos/muletillas casuales sueltas (correcciones_a_mano.md: "...Hey." se
    # trataba como nombre/llamada por ser una palabra suelta con mayuscula)
    "hey", "he", "ey", "na",
    # reacciones cortas identicas EN/DE mal tratadas como nombre/llamada o como
    # grito de palabra por su alargamiento ("Wow...", "Okay!"/"Okaaay!", "Man...")
    "wow", "okay", "ok", "man", "mann", "sorry", "sory",
}

# Subconjunto de EFFORTS que son muletillas conversacionales suaves: el
# usuario decidió que OmniVoice las saca bien solas (voz alemana consistente),
# así que YA NO se conservan en inglés -- se doblan como frase corta normal.
# El resto de EFFORTS (dolor, animales, llanto, risa) sigue conservándose:
# suenan peor sintetizadas y no dependen del idioma.
MILD_FILLERS = {
    "ah", "aah", "ahh", "oh", "ooh", "oha", "boah", "woah", "whoa", "huch", "hui", "ups", "oje",
    "och", "ach", "aha", "oho", "mh", "mhm", "mmm", "mm", "ahem", "hm", "hmm",
    "huh", "eh", "uh", "uhh", "um", "erm", "ahm", "ohm", "ha", "hah", "heh",
    "tsk", "tch", "tz", "psst", "sh", "shh", "pff", "pf", "puh", "phew",
    "hmpf", "wuhu", "juhu", "juchu", "yay", "wuhuu", "ow", "oof", "ugh", "uff",
    "hey", "he", "ey", "na", "wow", "okay", "ok", "man", "mann", "sorry", "sory",
}

# Acotaciones escenicas entre asteriscos: *seufz*, *keuch*, *winsel*. No son
# habla, son efectos; leerlas en voz alta es siempre un error.
STAGE_DIRECTION = re.compile(r"^\s*(?:\*[^*]+\*\s*)+$")
INLINE_STAGE_DIRECTION = re.compile(r"\*[^*]+\*")

# Negaciones. GRITADAS valen en cualquier idioma: un "NEEEEIN!" desgarrado y el
# "NOOOOOO!" del actor ingles son el mismo sonido, y el modelo las destroza al
# intentar actuarlas (timbre aspero, y en un caso medido 44 muestras saturadas).
# HABLADAS no: un "Nein." tranquilo hay que doblarlo, ahi si se oye que dice "no".
NEGATIONS = {"nein", "no", "nee", "ne", "nie", "niemals", "never"}
HARD_SHOUTABLE_VOCALIZATIONS = {
    # Ah/Aah/Ahhh are deliberately excluded: from text alone they are often
    # conversational hesitation/surprise and must go to short TTS + audio QA.
    "agh", "argh", "augh", "gah", "gugh", "hng",
    "hngh", "ngh", "oof", "ow", "ugh", "uff", "urgh", "ungh", "nng",
    "urg", "arg", "gyah", "gya", "hra", "hrah", "hrgh", "wah", "uah",
}

ONOMATOPOEIA_STEMS = {
    "bam", "bang", "boom", "kablam", "pcho", "fiu", "fyoo", "zisch",
    "hiss", "dscht", "womwom", "wom", "whoosh", "woosh",
    # Equivalentes escritos habituales en subtítulos alemanes. Son efectos
    # vocales/sonoros, no palabras que OmniVoice deba pronunciar.
    "bum", "bumm", "wum", "wumm", "rums", "krach", "peng", "puff",
    "platsch", "klirr", "zack", "knall", "surr", "brumm",
}


def _is_shouted(raw: str, token: str) -> bool:
    """Gritada = alargamiento ortografico fuerte o mayusculas sostenidas."""
    if re.search(r"(.)\1{2,}", token):
        return True
    return len(raw) >= 3 and raw.isupper()


def is_hard_shouted_vocalization(text: str) -> bool:
    """Keep an actual scream/groan even when its normalized stem is a filler."""
    raws = re.findall(r"[^\W\d_]+", text or "", re.UNICODE)
    if not raws:
        return False
    folded = [_fold(raw) for raw in raws]
    collapsed = [_collapse_runs(token) for token in folded]
    return (
        all(token in HARD_SHOUTABLE_VOCALIZATIONS for token in collapsed)
        and any(_is_shouted(raw, token) for raw, token in zip(raws, folded))
    )

MUSIC_MOVIE_PATTERNS = (
    re.compile(r"(?:^|[_\\/])op(?:ening)?(?:[_\\/.]|$)", re.I),
    re.compile(r"(?:^|[_\\/])ed(?:ending)?(?:[_\\/.]|$)", re.I),
    re.compile(r"op_sound", re.I),
    re.compile(r"ending_movie", re.I),
)


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    word_count: int
    synthesis_text: str | None = None
    cut_after_words: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _fold(text))


def _collapse_runs(token: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", token)


def _is_laughter_token(token: str) -> bool:
    """Detect written laughter before run collapsing turns it into a filler.

    For example, ``Hehehehe`` used to collapse to ``heh``. Since ``heh`` is a
    mild conversational reaction, a complete laugh could then be sent to TTS.
    This intentionally remains conservative: it only accepts long standalone
    tokens made from common laugh phonemes.
    """
    folded = _fold(token)
    return (
        len(folded) >= 5
        and folded.count("h") >= 2
        and bool(re.fullmatch(r"[abegihou]+", folded))
    )


def is_music_movie(path_or_name: str) -> bool:
    return any(pattern.search(path_or_name) for pattern in MUSIC_MOVIE_PATTERNS)


def is_punctuation_only(text: str) -> bool:
    return not words(text)


def _merge_stutters(tokens: list[str]) -> list[str]:
    """'Ä-Ähm' arrives as ['a','ahm'] and 'O-Oh' as ['o','oh']: the stuttered
    onset is a separate token whose single letter opens the next one. Merge it
    so the pair is judged as the interjection it actually is."""
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if len(token) == 1 and following and following.startswith(token):
            merged.append(following)
            index += 2
            continue
        merged.append(token)
        index += 1
    return merged


def is_effort_or_scream(text: str) -> bool:
    tokens = _merge_stutters(words(text))
    if not tokens:
        return True
    if all(_is_laughter_token(token) for token in tokens):
        return True
    collapsed = [_collapse_runs(token) for token in tokens]
    if all(token in EFFORTS or folded in EFFORTS
           for token, folded in zip(tokens, collapsed)):
        # Si TODOS los tokens son muletillas suaves (oh, ähm, hm, ach...) no
        # se conserva: se dobla como frase corta normal. Solo cuenta como
        # "hay que conservar" si queda al menos un esfuerzo DURO (dolor,
        # animal, llanto, risa) entre los tokens.
        if any(token not in MILD_FILLERS and folded not in MILD_FILLERS
               for token, folded in zip(tokens, collapsed)):
            return True
        return False
    # AAAAAH y equivalentes: una llamada vocal prolongada no se traduce.
    # Pero el alargamiento por si solo NO basta: 'H-Hiiiilfe! N-Notfall!' es
    # aleman de verdad ("¡Ayuda! ¡Emergencia!") y marcarlo aqui lo dejaria sin
    # doblar. Se exige que al colapsar el alargamiento quede una interjeccion
    # real, o que el token sea solo vocales.
    if len(tokens) <= 2:
        for token in tokens:
            if not re.search(r"(.)\1{3,}", token):
                continue
            stripped = _collapse_runs(token)
            if stripped in EFFORTS or re.fullmatch(r"[aeiouy]+[hn]?", stripped):
                return True
    return False


def is_stage_direction_plus_effort(text: str) -> bool:
    """True when a line is only stage directions plus a nonverbal effort.

    Example: ``*hechel* Wuff! Wuff!``.  The older full-line stage-direction
    regex missed this mixed notation and could send a pant/bark sequence to
    TTS even though neither part is spoken language.
    """
    if not INLINE_STAGE_DIRECTION.search(text or ""):
        return False
    remainder = INLINE_STAGE_DIRECTION.sub(" ", text or "").strip()
    return not remainder or is_effort_or_scream(remainder)


def is_shouted_negation_line(text: str) -> bool:
    """La linea entera son negaciones y vocalizaciones, y al menos una va gritada.

    Cubre "Nein ... Nein! NEEEEIN!" frente al ingles "No... No! NOOOOOO!" y
    "Nein, nein, nein ... Neeeeeeeiiiin!". Se conserva el original: el grito no
    tiene idioma y sintetizarlo suena mucho peor que dejarlo.
    """
    raws = re.findall(r"[^\W\d_]+", text or "", re.UNICODE)
    if not raws:
        return False
    tokens = [_fold(r) for r in raws]
    shouted = False
    for raw, token in zip(raws, tokens):
        collapsed = _collapse_runs(token)
        if token in NEGATIONS or collapsed in NEGATIONS:
            shouted = shouted or _is_shouted(raw, token)
            continue
        if token in EFFORTS or collapsed in EFFORTS:
            continue
        return False
    return shouted


# --- Punto 6 (grito de palabra alemana -> conservar el original inglés) y
# --- Punto 7 (onomatopeyas multisegmento: DSSSCHT / WOMMWOMM / BAAAAM).
# Un grito de palabra ("NEEIIIN", "HILFEE", "DOOONT") o una onomatopeya alargada
# suena mucho peor sintetizado que dejando el original; el usuario decidió
# conservar el inglés en esos casos aunque diga otra palabra (un grito desgarrado
# apenas tiene idioma). No confundir con alemán real: "H-Hiiilfe! N-Notfall!"
# lleva "Notfall", que NO es un grito, así que la línea entera no es toda gritos
# y se dobla igual.
def _has_strong_elongation(token: str) -> bool:
    return bool(re.search(r"(.)\1{2,}", token))          # 3+ caracteres iguales


def _is_shout_token(raw: str, folded: str) -> bool:
    collapsed = _collapse_runs(folded)
    if (
        folded in EFFORTS
        or collapsed in EFFORTS
        or collapsed in ONOMATOPOEIA_STEMS
    ):
        return True
    # A very short elongated sound ("Rrrrr", "Nnnn", "Woooo") can be
    # non-lexical. A normal word does not become language-neutral merely
    # because it is shouted: Biiiitte, Angriiiff, Haaallo, Schwesteeer and
    # Chidori must still be dubbed.
    if len(collapsed) <= 2 and _has_strong_elongation(folded):
        return True
    return False


def is_shouted_or_onomatopoeia(text: str) -> bool:
    """La línea corta entera son gritos de palabra u onomatopeyas alargadas."""
    raws = re.findall(r"[^\W\d_]+", text or "", re.UNICODE)
    if not raws or len(raws) > 4:
        return False
    folded = [_fold(r) for r in raws]
    any_shout = False
    for raw, tok in zip(raws, folded):
        if not _is_shout_token(raw, tok):
            return False
        collapsed = _collapse_runs(tok)
        if collapsed in MILD_FILLERS:
            # "Hmmm...", "Mmm..." alargados: siguen siendo muletilla suave (P1),
            # el alargamiento por sí solo no los convierte en un grito de palabra.
            continue
        if _has_strong_elongation(tok) or (len(raw) >= 3 and raw.isupper()):
            any_shout = True
    return any_shout


# --- Punto 4 (limpiar falsos positivos del detector): números romanos ("III",
# "XIV") y emoticonos ("XDDD", "XP") no son habla; leerlos como palabra alemana
# es siempre un error. Se conserva el original.
ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
EMOTICON = re.compile(r"^(x+d+|x+p+|d+x+|:+[dpo]+|;+[dp]+|lol|lmao|rofl|xoxo)$", re.I)


def is_nonspeech_glyph(text: str) -> bool:
    raws = re.findall(r"[^\W\d_]+", text or "", re.UNICODE)
    if len(raws) != 1:
        return False
    raw = raws[0]
    if raw.isupper() and len(raw) >= 2 and ROMAN.match(raw):      # II, III, XIV
        return True
    return bool(EMOTICON.match(raw))                              # XDDD, XP, lol


def is_developer_or_unused_text(text: str) -> bool:
    """Recognize explicit unreachable/developer placeholders, not dialogue."""
    folded = _fold(text or "").strip()
    compact = re.sub(r"\s+", " ", folded)
    return (
        "未使用" in (text or "")
        or compact in {"unused", "dev: unused"}
        or compact.startswith("dev: *unused")
        or compact == "please report if this is displayed."
    )


def contains_cjk(text: str) -> bool:
    """German delivery text still written in Japanese/Chinese script."""
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text or ""))


def is_language_neutral_call(source: str, target: str) -> bool:
    # _merge_stutters: "W-Wow..." tokeniza como ['w','wow'] y el 'w' suelto
    # rompia el guard de EFFORTS de abajo (encontrado en revision manual,
    # Cmmu_002_090_C#22), igual que ya se corrigio en is_effort_or_scream.
    source_words = _merge_stutters(words(source))
    target_words = _merge_stutters(words(target))
    source_canonical = [_collapse_runs(token) for token in source_words]
    target_canonical = [_collapse_runs(token) for token in target_words]
    if not source_words or source_canonical != target_canonical or len(target_words) > 3:
        return False
    compact_source = "".join(source_canonical)
    compact_target = "".join(target_canonical)
    if compact_source != compact_target:
        return False
    if all(token in EFFORTS for token in target_canonical):
        return False   # "Oh...", "Ähm..." -- muletilla, no nombre/llamada
    # Persona!, Shinji!, Aragaki-san!, Senpai!, nombres o llamadas idénticas.
    # El "!" solo cuenta como señal de llamada para una palabra suelta: en
    # frases de 2-3 palabras ("Aye aye, Captain!") cualquier idiomatismo
    # calcado del ingles pasaba por "llamada" solo por llevar un signo de
    # exclamacion (correcciones_a_mano.md). Ahi se exige Title Case real.
    target_raw = re.findall(r"[\w-]+", target)
    if len(target_raw) == 1:
        return "!" in source or "!" in target or target_raw[0][:1].isupper()
    return all(part[:1].isupper() for part in target_raw)


def is_language_neutral_vocalization(source: str, target: str) -> bool:
    """Preserve identical EN/DE vocalizations instead of cloning them.

    This is intentionally checked separately from ``MILD_FILLERS``: a soft
    filler may still need German TTS when its target changes, but when the
    acoustic utterance is already identical (``Wow``, ``Ugh``, ``Hmm``) the
    actor's original performance is always preferable.
    """
    source_tokens = [_collapse_runs(token) for token in _merge_stutters(words(source))]
    target_tokens = [_collapse_runs(token) for token in _merge_stutters(words(target))]
    return bool(
        source_tokens
        and source_tokens == target_tokens
        and all(token in EFFORTS for token in target_tokens)
    )


def expanded_prompt(target: str) -> str:
    """Devuelve contexto auxiliar; solo se entrega el primer segmento."""
    clean = target.strip()
    stem = re.sub(r"(?:\.{2,}|[.!?…]+)\s*$", "", clean).strip()
    key = _fold(stem)
    if key in {"morgen", "guten morgen"}:
        tail = "Schön, dich zu sehen."
    elif key == "willkommen":
        tail = "Schön, dass du hier bist."
    elif key == "warte":
        tail = "Geh noch nicht weiter."
    elif clean.rstrip().endswith("?"):
        tail = "Sag mir bitte, was du meinst."
    elif clean.rstrip().endswith("!"):
        tail = "Hör mir bitte einen Moment zu."
    else:
        tail = "Ich wollte dir noch etwas sagen."
    return f"{stem} ... {tail}"


def classify_line(source: str, target: str) -> Decision:
    count = len(words(target))
    if is_developer_or_unused_text(target) or is_developer_or_unused_text(source):
        return Decision(KEEP_ORIGINAL, "developer_or_unused_text", count)
    if contains_cjk(target):
        return Decision(KEEP_ORIGINAL, "foreign_or_untranslated_text", count)
    if is_punctuation_only(source) and is_punctuation_only(target):
        return Decision(KEEP_ORIGINAL, "silence_or_punctuation", count)
    if STAGE_DIRECTION.match(target or "") or STAGE_DIRECTION.match(source or ""):
        return Decision(KEEP_ORIGINAL, "stage_direction", count)
    if (
        is_stage_direction_plus_effort(target)
        or is_stage_direction_plus_effort(source)
    ):
        return Decision(KEEP_ORIGINAL, "stage_direction_plus_effort", count)
    # Once a mixed ``*stage direction* + speech`` row has survived the
    # all-nonverbal guard above, only classify its pronounceable remainder.
    # Otherwise a hard ``*keuch*`` could make lexical ``Sorry``/``Hey`` keep
    # the entire English clip.
    target_spoken = INLINE_STAGE_DIRECTION.sub(" ", target or "").strip()
    if is_hard_shouted_vocalization(target_spoken):
        return Decision(KEEP_ORIGINAL, "hard_shouted_vocalization", count)
    # Subtitle coverage rule (2026-07-24): an identical spoken word still
    # carries the English actor/timbre, so it must be cloned. This includes
    # "Wow", station announcements and proper-name calls. Hard nonverbal
    # efforts remain protected by the effort/onomatopoeia checks below.
    # Se decide por el ALEMÁN (lo que se dobla), no por el inglés: si el
    # alemán tiene contenido real ("Wie...?"), se dobla aunque el inglés sea
    # una simple interjección ("Huh...?"). Antes se conservaba con OR(source,
    # target) y eso tapaba alemán real detrás de una muletilla inglesa.
    if is_effort_or_scream(target_spoken):
        return Decision(KEEP_ORIGINAL, "effort_or_scream", count)
    if is_shouted_negation_line(target) and is_shouted_negation_line(source):
        return Decision(KEEP_ORIGINAL, "shouted_negation", count)
    if is_nonspeech_glyph(target) or is_nonspeech_glyph(source):
        return Decision(KEEP_ORIGINAL, "nonspeech_glyph", count)
    if is_shouted_or_onomatopoeia(target):
        return Decision(KEEP_ORIGINAL, "shouted_word_or_onomatopoeia", count)
    if 1 <= count <= 3:
        # Frase corta: TTS normal (sin prompt ampliado ni corte -- era frágil).
        # El orquestador la genera plana y la pasa por revisión tipo "oder"
        # (¿se cortó la última palabra?, ¿suena mal?) y regenera si hace falta.
        return Decision(SHORT_TTS_QA, "short_spoken_line", count, synthesis_text=target)
    return Decision(TTS, "normal_spoken_line", count, synthesis_text=target)
