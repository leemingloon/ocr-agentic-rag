"""
Hedged language: financial hedging lexicon and intensity scorer.
High hedge intensity widens the neutral band (require higher confidence for pos/neg).
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Financial hedging phrases — weak (slightly, modestly) vs strong (materially, significantly)
# Strong hedge: often negates or heavily qualifies the polarity
HEDGE_STRONG = [
    "materially below", "materially above", "materially worse", "materially better",
    "significantly missed", "significantly beat", "significantly below", "significantly above",
    "well below", "well above", "well short", "well ahead",
    "broadly in line", "largely in line", "roughly in line",
    "tepid", "muted", "subdued", "mixed", "lackluster",
]
# Weak hedge: softens but doesn't flip
HEDGE_WEAK = [
    "somewhat", "marginally", "slightly", "modestly", "largely", "broadly",
    "slightly below", "slightly above", "modestly ahead", "modestly below",
    "in line with", "broadly in line with", "largely in line with",
    "more or less", "rather", "fairly", "relatively",
]
# Compile for speed
_HEDGE_STRONG_RE = [re.compile(re.escape(p), re.I) for p in HEDGE_STRONG]
_HEDGE_WEAK_RE = [re.compile(re.escape(p), re.I) for p in HEDGE_WEAK]


class HedgeScorer:
    """
    Score hedge intensity in text. Returns (strong_count, weak_count) or a single float.
    Used to adjust confidence threshold: high intensity → require higher confidence for pos/neg.
    """

    def __init__(self):
        self.strong_patterns = _HEDGE_STRONG_RE
        self.weak_patterns = _HEDGE_WEAK_RE

    def score(self, text: str) -> Tuple[int, int]:
        """Return (strong_hedge_count, weak_hedge_count)."""
        if not text or not text.strip():
            return 0, 0
        strong = sum(1 for p in self.strong_patterns if p.search(text))
        weak = sum(1 for p in self.weak_patterns if p.search(text))
        return strong, weak

    def intensity(self, text: str) -> float:
        """
        Return a single intensity score in [0, 1+]. 0 = no hedge; >0.5 = strong hedge.
        Used to scale neutral band: intensity * scale → require that much more confidence.
        """
        s, w = self.score(text)
        return s * 0.5 + w * 0.15

    def has_hedge(self, text: str) -> bool:
        """True if any hedging phrase is present."""
        s, w = self.score(text)
        return s > 0 or w > 0
