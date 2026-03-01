"""
Small KB of standard financial constants for numerical grounding.

When a query mentions a concept (e.g. effective tax rate, statutory rate) and no retrieved
chunk contains a matching number, we can inject a one-line hint so the model states the
assumption or uses a standard value (see RAG_ROADMAP numerical grounding).
"""

import re
from typing import List, Optional, Tuple

# Concept -> (regex to detect in query, one-line hint, optional canonical value for program)
FINANCIAL_CONSTANTS = [
    (
        r"statutory\s+(?:tax\s+)?rate|federal\s+statutory\s+rate|u\.?s\.?\s+statutory",
        "If the document does not state the statutory tax rate, common US federal rate is 21% (post-TCJA). State your assumption or say INSUFFICIENT_DATA.",
        "0.21",
    ),
    (
        r"effective\s+tax\s+rate|effective\s+rate\s+.*tax",
        "If the document does not state the effective tax rate, state your assumption (e.g. 21% statutory) or say INSUFFICIENT_DATA.",
        None,
    ),
    (
        r"risk[- ]free\s+rate|risk\s+free\s+rate",
        "If the document does not state the risk-free rate, common assumption is 2–5% (Treasury yield). State your assumption or say INSUFFICIENT_DATA.",
        None,
    ),
]


def detect_missing_constant(
    query: str,
    chunk_texts: List[str],
) -> Optional[Tuple[str, str]]:
    """
    If the query mentions a concept that often requires a constant and no chunk
    contains a plausible number for it, return (concept_key, hint_line) to inject.
    """
    if not query or not isinstance(query, str):
        return None
    q = query.strip().lower()
    combined_chunks = " ".join(chunk_texts or []).lower()
    for pattern, hint, _ in FINANCIAL_CONSTANTS:
        if not re.search(pattern, q, re.I):
            continue
        # Check if chunks already contain a percentage or rate-like number near tax/rate
        if re.search(r"statutory|effective\s+tax|tax\s+rate", combined_chunks, re.I):
            # Look for a number like 21, 21%, 0.21 in the same sentence
            if re.search(r"(?:statutory|effective|tax\s+rate).{0,40}\d{1,3}(?:\.\d+)?\s*%?", combined_chunks, re.I | re.DOTALL):
                return None  # Already have a number
        if re.search(r"risk[- ]free|risk\s+free", combined_chunks, re.I):
            if re.search(r"(?:risk[- ]free|treasury).{0,30}\d(?:\.\d+)?\s*%?", combined_chunks, re.I | re.DOTALL):
                return None
        return ("constant_hint", hint)
    return None
