"""Small deterministic Praat TextGrid parser for MFA diagnostics."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PhoneInterval:
    start: float
    end: float
    text: str
    tier: str = "phones"

@dataclass(frozen=True)
class TextGrid:
    xmin: float
    xmax: float
    intervals: tuple[PhoneInterval, ...]
    tier_names: tuple[str, ...] = ()

    def coverage(self, expected: str, *, expected_phones: list[str] | None = None) -> float:
        """Return content coverage, never just a length ratio.

        Prefer a word tier when one exists.  If only a phone tier is present,
        callers may supply the expected phone sequence; otherwise a one-to-one
        character comparison is used and cannot turn an unrelated same-length
        label into a perfect score.
        """
        usable = [item for item in self.intervals if item.text.strip() and item.text.strip().casefold() not in {"sil", "sp", "<sil>", "<unk>"}]
        word_rows = [item for item in usable if "word" in item.tier.casefold()]
        rows = word_rows or usable
        labels = [re.sub(r"[^\w']+", "", item.text.casefold()) for item in rows]
        labels = [item for item in labels if item]
        if not expected.strip() or not labels:
            return 0.0
        if word_rows:
            want = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", expected.casefold(), flags=re.UNICODE)
            got = labels
        elif expected_phones is not None:
            want = [str(item).casefold() for item in expected_phones if str(item).strip()]
            got = [item for item in labels if item]
        else:
            want = list(re.sub(r"[^\w']+", "", expected.casefold()))
            got = list("".join(labels))
        if not want:
            return 0.0
        matcher = SequenceMatcher(a=want, b=got, autojunk=False)
        matched = sum(size for _a, _b, size in matcher.get_matching_blocks())
        return min(1.0, matched / len(want))

def parse_textgrid(path: str | Path) -> TextGrid:
    text = Path(path).read_text(encoding="utf-8-sig")
    bounds = [float(x) for x in re.findall(r"(?:^|\n)\s*x(?:min|max)\s*=\s*([0-9.eE+-]+)", text)]
    if len(bounds) < 2: raise ValueError("TextGrid is missing xmin/xmax")
    intervals=[]; current_tier="phones"
    tier_names = list(re.finditer(r'name\s*=\s*"([^"]*)"', text))
    for block in re.split(r'(?=\bintervals\s*\[\d+\]\s*:)', text)[1:]:
        before = text[:text.find(block)]
        prior = [m for m in tier_names if m.start() < text.find(block)]
        current_tier = prior[-1].group(1) if prior else "phones"
        starts = re.findall(r'\bxmin\s*=\s*([0-9.eE+-]+)', block)
        ends = re.findall(r'\bxmax\s*=\s*([0-9.eE+-]+)', block)
        label = re.search(r'\btext\s*=\s*"(.*)"', block)
        if starts and ends and label:
            intervals.append(PhoneInterval(float(starts[-1]), float(ends[-1]), label.group(1).replace('\\"','"'), current_tier))
    if not intervals: raise ValueError("TextGrid contains no intervals")
    return TextGrid(bounds[0], bounds[1], tuple(intervals), tuple(dict.fromkeys(item.tier for item in intervals)))

__all__=["PhoneInterval","TextGrid","parse_textgrid"]
