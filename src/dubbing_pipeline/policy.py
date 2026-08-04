"""Topology-independent line policy.

The policy is deterministic: an LLM may propose text or a correction, but it
does not decide whether an unproven audio unit is allowed into the release.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import Line

KEEP_ORIGINAL = "KEEP_ORIGINAL"
BLOCKED = "BLOCKED"
SHORT_TTS_QA = "SHORT_TTS_QA"
TTS = "TTS"

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_STAGE = re.compile(r"\*[^*]+\*")
_NEUTRAL = {
    "ah", "aah", "ahh", "agh", "argh", "augh", "eh", "gah", "gugh",
    "ha", "hah", "heh", "hm", "hmm", "hng", "hngh", "huh", "ngh", "oh",
    "ooh", "oof", "ow", "ugh", "uh", "uhh", "uff", "urgh", "um", "erm",
    "wow", "woah", "whoa", "huch", "ach", "tsk", "tch", "pff", "puh",
    "hmpf", "grr", "wuff", "wau", "woof", "arf", "miau", "meow",
}
_COMMON_WORDS = {"a", "an", "and", "are", "be", "but", "do", "for", "go", "hey", "i", "in", "is", "it", "me", "my", "no", "not", "of", "oh", "on", "or", "so", "that", "the", "this", "to", "we", "what", "why", "you"}


@dataclass(frozen=True)
class Decision:
    policy: str
    reason: str
    tts_text: str | None


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def words(text: str) -> list[str]:
    return [fold(item) for item in _WORD.findall(str(text or ""))]


def strip_stage_directions(text: str) -> str:
    return re.sub(r"\s+", " ", _STAGE.sub(" ", str(text or ""))).strip()


def append_ellipsis(text: str, enabled: bool = True, marker: str = "...") -> str:
    value = str(text or "").strip()
    if not enabled or not value or not words(value):
        return value
    marker = marker.strip() or "..."
    return value if value.endswith(marker) else f"{value}{marker}"


def is_neutral_effort(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and all(token in _NEUTRAL or re.sub(r"(.)\1+", r"\1", token) in _NEUTRAL for token in tokens)


def isolated_call(source: str, target: str) -> bool:
    source_tokens, target_tokens = words(source), words(target)
    if len(source_tokens) != 1 or len(target_tokens) != 1:
        return False
    if source_tokens[0] != target_tokens[0] or source_tokens[0] in _COMMON_WORDS:
        return False
    # A single identical non-common token is normally a name/skill call. The
    # project can disable this by setting force_keep_original=false and a
    # custom policy in its manifest.
    return True


def classify_line(line: Line, *, append_ellipsis_experiment: bool = False) -> Decision:
    if not line.subtitle_authorized:
        return Decision(KEEP_ORIGINAL, "NO_VISIBLE_SUBTITLE_CARD", None)
    if line.force_keep_original:
        return Decision(KEEP_ORIGINAL, line.preserve_reason or "EXPLICIT_POLICY", None)
    if line.topology == "EMBEDDED_FMV":
        missing = [key for key, value in {
            "movie_identity_verified": line.movie_identity_verified,
            "card_identity_verified": line.card_identity_verified,
            "card_timebase_verified": line.card_timebase_verified,
        }.items() if not value]
        if missing:
            return Decision(BLOCKED, "MISSING_FMV_EVIDENCE:" + ",".join(missing), None)
    if isolated_call(line.source_text, line.target_text):
        return Decision(KEEP_ORIGINAL, "ISOLATED_NAME_OR_SKILL_CALL", None)
    target = strip_stage_directions(line.synthesis_text_override or line.effective_target_text)
    if not target or not words(target):
        return Decision(KEEP_ORIGINAL, "NONLEXICAL_OR_EMPTY", None)
    if is_neutral_effort(line.source_text) and is_neutral_effort(target):
        return Decision(KEEP_ORIGINAL, "LANGUAGE_NEUTRAL_EFFORT", None)
    decision = SHORT_TTS_QA if len(words(target)) <= 2 else TTS
    return Decision(decision, "SUBTITLE_AUTHORIZED", append_ellipsis(target, append_ellipsis_experiment))
