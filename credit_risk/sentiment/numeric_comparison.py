"""
Numeric comparison context: detect "X vs expected Y", "X vs consensus Y", etc.,
compute percentage deviation, and signal negative when underperformance (e.g. < -5%).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Number: optional $, digits, optional unit (B/bn, M/mn, %)
_NUM = r"(?:\$?\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:%|billion|bn|B|million|mn|m|M|mln)?"
# Patterns: (actual_capture, expected_capture) — actual can be "X" or "metric of X"
PATTERNS = [
    re.compile(r"revenue\s+of\s+" + _NUM + r"\s+vs\.?\s+(?:expected|consensus|forecast)\s+" + _NUM, re.I),
    re.compile(r"earnings\s+of\s+" + _NUM + r"\s+vs\.?\s+(?:expected|consensus|forecast)\s+" + _NUM, re.I),
    re.compile(r"(.+?)\s+vs\.?\s+expected\s+" + _NUM, re.I),
    re.compile(r"(.+?)\s+vs\.?\s+consensus\s+" + _NUM, re.I),
    re.compile(r"(.+?)\s+vs\.?\s+forecast\s+" + _NUM, re.I),
    re.compile(r"(.+?)\s+vs\.?\s+prior\s+" + _NUM, re.I),
    re.compile(r"(.+?)\s+vs\.?\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|billion|bn|B|million|mn|m|M|mln)?", re.I),
]


def _parse_number(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*([%bBnMmKk])?", s)
    if not m:
        return None
    v = float(m.group(1))
    unit = (m.group(2) or "").upper()
    if unit in ("B", "BN"):
        v *= 1e9
    elif unit in ("M", "MN"):
        v *= 1e6
    elif unit == "K":
        v *= 1e3
    return v


def _normalize_to_comparable(actual_str: str, expected_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse actual and expected; if one is % and the other not, return (actual, expected) as same unit where possible."""
    a = _parse_number(actual_str)
    e = _parse_number(expected_str)
    return a, e


class NumericComparisonExtractor:
    """Extract actual vs expected from financial comparison phrases; compute % deviation."""

    def __init__(self, underperform_pct: float = 5.0):
        self.underperform_pct = underperform_pct

    def extract(self, text: str) -> Optional[dict]:
        """
        If text matches "X vs expected Y" (or consensus/forecast/prior), return
        { "actual": float, "expected": float, "pct_deviation": float, "is_underperform": bool }.
        """
        if not text or not text.strip():
            return None
        text = text.strip()
        for pat in PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            groups = m.groups()
            if len(groups) >= 2:
                actual_str = groups[0].strip()
                expected_str = groups[-1].strip()
                # If first group is not a number (e.g. "Revenue of $2.1B"), extract number from it
                num_in_first = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:%|billion|bn|B|million|mn|m|M|mln)?", actual_str, re.I)
                if num_in_first:
                    actual_str = num_in_first.group(0).replace("$", "").strip()
                actual, expected = _normalize_to_comparable(actual_str, expected_str)
                if actual is not None and expected is not None and expected != 0:
                    pct_dev = ((actual - expected) / abs(expected)) * 100.0
                    return {
                        "actual": actual,
                        "expected": expected,
                        "pct_deviation": pct_dev,
                        "is_underperform": pct_dev < -self.underperform_pct,
                    }
                # Same-unit percentage comparison
                if "%" in actual_str or "%" in expected_str:
                    a = _parse_number(actual_str.replace("%", "").strip())
                    e = _parse_number(expected_str.replace("%", "").strip())
                    if a is not None and e is not None and e != 0:
                        pct_dev = ((a - e) / abs(e)) * 100.0
                        return {
                            "actual": a,
                            "expected": e,
                            "pct_deviation": pct_dev,
                            "is_underperform": pct_dev < -self.underperform_pct,
                        }
        return None
