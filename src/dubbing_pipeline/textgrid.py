"""Small deterministic Praat TextGrid parser for MFA diagnostics."""
from __future__ import annotations
import re
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

    def coverage(self, expected: str) -> float:
        heard = "".join(item.text for item in self.intervals if item.text.strip())
        want = "".join(str(expected).split())
        if not want: return 0.0
        return min(len(heard), len(want)) / len(want)

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
    return TextGrid(bounds[0], bounds[1], tuple(intervals))

__all__=["PhoneInterval","TextGrid","parse_textgrid"]
