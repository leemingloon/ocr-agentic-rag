"""
Preprocessing applied before index build (section tagging, unit parsing, dedup, provenance).

Used by scripts/build_finqa_embeddings_colab.py and scripts/build_tatqa_embeddings_colab.py
so that Colab index builds include these steps before embedding. Assembly-time use of
section/units is in the orchestrator (unit note in prompt; optional section filter later).
"""

import hashlib
import re
from typing import List, Optional, Dict, Any

# Section types for chunk metadata (multi-hop / retrieval preference)
SECTION_INCOME_STATEMENT = "income_statement"
SECTION_BALANCE_SHEET = "balance_sheet"
SECTION_NOTES = "notes"
SECTION_UNKNOWN = "unknown"


def infer_section_type(text: str, context_before: str = "") -> str:
    """
    Infer document section from chunk text and optional preceding context (e.g. pre_text).

    Heuristics: "statement of operations", "income statement", "balance sheet",
    "note 5", "note 6", "consolidated statements", "MD&A", etc.
    """
    combined = (context_before + "\n" + text).lower()
    if not combined.strip():
        return SECTION_UNKNOWN
    # Income statement / operations
    if re.search(
        r"statement(s)?\s+of\s+operations|consolidated\s+statement(s)?\s+of\s+operations|"
        r"income\s+statement|statements?\s+of\s+comprehensive\s+income|"
        r"results?\s+of\s+operations",
        combined,
    ):
        return SECTION_INCOME_STATEMENT
    # Balance sheet
    if re.search(
        r"balance\s+sheet|statement(s)?\s+of\s+financial\s+position|"
        r"consolidated\s+balance\s+sheet|statement(s)?\s+of\s+condition",
        combined,
    ):
        return SECTION_BALANCE_SHEET
    # Notes (footnotes)
    if re.search(
        r"\bnote\s+\d+\b|footnote|notes\s+to\s+(the\s+)?consolidated\s+financial\s+statements",
        combined,
    ):
        return SECTION_NOTES
    if re.search(r"md\s*&\s*a|management'?s\s+discussion", combined):
        return SECTION_NOTES  # treat MD&A as notes-like
    return SECTION_UNKNOWN


def detect_units(text: str) -> List[str]:
    """
    Detect unit hints in chunk text. Returns a list of tags: millions, thousands,
    per_share, quarterly, etc., for use in chunk metadata and assembly-time normalisation.
    """
    if not text or not isinstance(text, str):
        return []
    lower = text.lower()
    units = []
    if re.search(
        r"\$?\s*(?:in\s+)?(?:millions?|million\s+of\s+dollars?)|"
        r"\(?\s*in\s+millions?\s*\)?|amounts?\s+in\s+millions?",
        lower,
    ):
        units.append("millions")
    if re.search(
        r"(?:in\s+)?thousands?(?:\s+of\s+dollars?)?|"
        r"\(?\s*in\s+thousands?\s*\)?|amounts?\s+in\s+thousands?",
        lower,
    ):
        units.append("thousands")
    if re.search(
        r"per\s+share|per\s+common\s+share|diluted\s+earnings?\s+per\s+share|eps",
        lower,
    ):
        units.append("per_share")
    if re.search(
        r"quarterly|quarter\s+ended|three\s+months?\s+ended|six\s+months?\s+ended",
        lower,
    ):
        units.append("quarterly")
    if re.search(r"billions?|billion\s+of\s+dollars?", lower):
        units.append("billions")
    return units


def content_hash(text: str) -> str:
    """Stable hash of chunk content for deduplication."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def extract_section_header(context: str, max_len: int = 120) -> Optional[str]:
    """
    Extract a short section header from document context (e.g. "Note 5 - Debt") for chunk metadata.
    """
    if not context or not isinstance(context, str):
        return None
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    for line in lines[:5]:  # first few lines only
        if len(line) > max_len:
            continue
        if re.search(r"^note\s+\d+", line, re.I) or re.search(r"statement(s)?\s+of\s+(operations|financial)", line, re.I) or re.search(r"balance\s+sheet", line, re.I):
            return line[:max_len]
    return None


# Patterns for multi-level header detection (order matters for hierarchy)
_HEADER_PATTERNS = [
    (r"^part\s+[IVXLCDMivxlcdm]+\s*[\.\-:]?\s*.+", 1),   # Part I, Part II
    (r"^note\s+\d+\s*[\.\-:]?\s*.+", 2),                 # Note 5 - Debt
    (r"^statement(s)?\s+of\s+(operations|comprehensive\s+income|financial\s+position|condition|cash\s+flows)", 2),
    (r"^consolidated\s+balance\s+sheet", 2),
    (r"^consolidated\s+statement(s)?\s+of\s+operations", 2),
    (r"^balance\s+sheet", 2),
    (r"^income\s+statement", 2),
    (r"^#{1,6}\s+.+", 0),  # markdown # to ###### (level from length)
]


def extract_header_hierarchy(context: str, max_headers: int = 25) -> List[str]:
    """
    Extract ordered list of section headers from document context (multi-level).
    Gives the model the full outline so chunk context is clear even when the
    section title is far from the data in the source.
    """
    if not context or not isinstance(context, str):
        return []
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    out: List[str] = []
    for line in lines:
        if len(out) >= max_headers:
            break
        for pat, _ in _HEADER_PATTERNS:
            m = re.match(pat, line, re.I)
            if m:
                # Normalise: trim to 100 chars
                out.append(line[:100].strip())
                break
    return out


def extract_references_pages(text: str) -> List[int]:
    """
    Extract page numbers referenced in chunk text (e.g. "see page 5", "on page 3", "p. 12").
    Used for cross-page number references: a figure on page 1 referenced on page 3.
    """
    if not text or not isinstance(text, str):
        return []
    seen: set = set()
    out: List[int] = []
    # "page 5", "on page 3", "see page 12", "(p. 4)", "p. 10"
    for m in re.finditer(r"(?:^|\s)(?:see\s+)?(?:on\s+)?page\s+(\d+)|(?:^|\s)\(?p\.?\s*(\d+)\)?|(?:^|\s)p\.\s*(\d+)", text, re.I):
        g = m.group(1) or m.group(2) or m.group(3)
        if g:
            n = int(g)
            if n not in seen and 1 <= n <= 9999:
                seen.add(n)
                out.append(n)
    return sorted(out)


def add_section_and_units(
    chunks: List[Any],
    *,
    context_by_corpus: Optional[Dict[str, str]] = None,
) -> None:
    """
    Mutate each chunk's metadata with section_type and units. If context_by_corpus
    is provided, use it for section inference and for unit detection so document-level
    phrases (e.g. "in millions" in pre_text) apply to table rows that don't contain them.
    """
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        text = getattr(c, "text", "") or ""
        corpus_id = meta.get("corpus_id", "")
        context = (context_by_corpus or {}).get(corpus_id, "")
        meta["section_type"] = infer_section_type(text, context)
        # Multi-level headers: full document outline so model sees section context
        header_hierarchy = extract_header_hierarchy(context)
        if header_hierarchy:
            meta["header_hierarchy"] = header_hierarchy
            meta["section_header"] = header_hierarchy[-1]  # innermost section
        else:
            section_header = extract_section_header(context)
            if section_header:
                meta["section_header"] = section_header
        # Cross-page references: page numbers mentioned in this chunk (e.g. "see page 5")
        ref_pages = extract_references_pages(text)
        if ref_pages:
            meta["references_pages"] = ref_pages
        # Units: from chunk text and from document context so table rows get doc-level units
        from_chunk = detect_units(text)
        from_doc = detect_units(context) if context else []
        combined = list(dict.fromkeys(from_doc + from_chunk))  # doc first, then chunk, no duplicates
        if combined:
            meta["units"] = combined
        c.metadata = meta


def add_provenance(
    chunks: List[Any],
    *,
    page_by_corpus: Optional[Dict[str, int]] = None,
    table_id_prefix_by_corpus: Optional[Dict[str, str]] = None,
) -> None:
    """
    Add page_number and table_id to chunk metadata when available.
    page_by_corpus: corpus_id -> page number.
    table_id_prefix_by_corpus: corpus_id -> prefix (e.g. "GS/2014/page_134"); we append _row0 etc. from chunk metadata if present.
    """
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        cid = meta.get("corpus_id", "")
        if page_by_corpus and cid in page_by_corpus:
            meta["page_number"] = page_by_corpus[cid]
        prefix = (table_id_prefix_by_corpus or {}).get(cid) or cid
        if meta.get("chunk_type") == "table" or "table" in (meta.get("chunk_type") or ""):
            row_idx = meta.get("row_index")
            meta["table_id"] = f"{prefix}_table" if row_idx is None else f"{prefix}_row{row_idx}"
        elif meta.get("row_index") is not None:
            meta["table_id"] = f"{prefix}_row{meta['row_index']}"
        c.metadata = meta


def deduplicate_chunks(
    chunks: List[Any],
    *,
    keep: str = "first",
) -> List[Any]:
    """
    Deduplicate by content hash. keep="first" keeps the first occurrence and sets
    duplicate_count on that chunk. Removes later duplicates from the list so the
    index is built over unique content; retriever can prefer low duplicate_count later.
    """
    seen: Dict[str, int] = {}
    out: List[Any] = []
    for c in chunks:
        text = getattr(c, "text", "") or ""
        h = content_hash(text)
        meta = getattr(c, "metadata", None) or {}
        if h not in seen:
            seen[h] = len(out)
            meta["content_hash"] = h
            meta["duplicate_count"] = 1
            c.metadata = meta
            out.append(c)
        else:
            idx = seen[h]
            first = out[idx]
            first_meta = getattr(first, "metadata", None) or {}
            first_meta["duplicate_count"] = first_meta.get("duplicate_count", 1) + 1
            first.metadata = first_meta
    return out


def preprocess_chunks_for_index(
    chunks: List[Any],
    *,
    context_by_corpus: Optional[Dict[str, str]] = None,
    page_by_corpus: Optional[Dict[str, int]] = None,
    table_id_prefix_by_corpus: Optional[Dict[str, str]] = None,
    dedup: bool = True,
) -> List[Any]:
    """
    Full preprocessing pipeline: section + units, provenance, content hash, dedup.
    Returns the list of chunks to pass to retriever.build_index() (may be smaller if dedup=True).
    """
    if not chunks:
        return []
    add_section_and_units(chunks, context_by_corpus=context_by_corpus)
    add_provenance(
        chunks,
        page_by_corpus=page_by_corpus,
        table_id_prefix_by_corpus=table_id_prefix_by_corpus,
    )
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        meta["content_hash"] = content_hash(getattr(c, "text", "") or "")
        c.metadata = meta
    if dedup:
        chunks = deduplicate_chunks(chunks, keep="first")
    return chunks
