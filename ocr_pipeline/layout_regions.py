"""
Layout-aware regions for OCR: dataset-specific vertical zones (header / body / footer).

Used to split full-page OCR text by line order so entity extraction can prefer
header for company/address/date (SROIE) and footer for total (SROIE), or
header vs body for FUNSD form fields.
"""
from __future__ import annotations

from typing import Any

# Region definitions: (y_frac_start, y_frac_end) = fraction of lines from top (0) to bottom (1).
# Lines are assigned by order: first N% of lines = header, last M% = footer, etc.
SROIE_REGIONS = {
    "header": (0.0, 0.40),   # company, address, date typically in top 40% of lines
    "body": (0.40, 0.65),    # line items
    "footer": (0.65, 1.0),   # total, cash, change
}
FUNSD_REGIONS = {
    "header": (0.0, 0.25),   # title / form header
    "body": (0.25, 1.0),     # questions and answers
}
DATASET_REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "SROIE": SROIE_REGIONS,
    "FUNSD": FUNSD_REGIONS,
}


def get_region_slices(dataset_name: str) -> dict[str, tuple[float, float]]:
    """Return region name -> (start_frac, end_frac) for the dataset. Frac = fraction of line count (0=top, 1=bottom)."""
    return DATASET_REGIONS.get(str(dataset_name).upper(), {}).copy()


def split_text_into_region_lines(text: str, dataset_name: str) -> dict[str, str]:
    """
    Split full OCR text into region texts by line order.
    Lines are split by newline; then lines are assigned to regions by their index (first N% = header, etc.).
    Returns dict region_name -> single string (lines joined by space for that region).
    """
    if not text or not isinstance(text, str):
        return {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n == 0:
        return {}
    regions = get_region_slices(dataset_name)
    if not regions:
        return {"full": text.strip()}
    out: dict[str, str] = {}
    for region_name, (start_frac, end_frac) in regions.items():
        i0 = int(n * start_frac)
        i1 = int(n * end_frac)
        if i0 >= i1:
            segment = []
        else:
            segment = lines[i0:i1]
        out[region_name] = " ".join(segment) if segment else ""
    return out


def get_region_text_for_entity(dataset_name: str, entity_key: str, region_texts: dict[str, str]) -> str:
    """
    Return the region text(s) most relevant for extracting a given entity.
    SROIE: company, address, date -> header; total -> footer.
    FUNSD: not entity-based; returns body (or full).
    """
    d = str(dataset_name).upper()
    if d == "SROIE":
        if entity_key in ("company", "address", "date"):
            return region_texts.get("header", "") or region_texts.get("full", "")
        if entity_key == "total":
            return region_texts.get("footer", "") or region_texts.get("full", "")
        return region_texts.get("full", "")
    if d == "FUNSD":
        return region_texts.get("body", "") or region_texts.get("full", "")
    return region_texts.get("full", "")


def assign_bbox_to_region(
    bbox: tuple[int, int, int, int] | list[int],
    image_height: int,
    dataset_name: str,
) -> str:
    """
    Assign a single bbox to a layout region by vertical position (crcresearch/FUNSD: spatial layout).
    bbox can be (x, y, w, h) or [x1, y1, x2, y2]. Uses centre y to decide header vs body.
    image_height: full image height in pixels.
    Returns region name: "header" or "body" for FUNSD; "header" / "body" / "footer" for SROIE.
    """
    if image_height <= 0:
        return "body"
    if len(bbox) >= 4:
        # (x, y, w, h) vs [x1, y1, x2, y2]: use y2 > y1 as sign of (x1,y1,x2,y2)
        if bbox[3] > bbox[1]:
            y_centre = (bbox[1] + bbox[3]) // 2
        else:
            y_centre = bbox[1] + (bbox[3] or 0) // 2
    else:
        return "body"
    y_frac = y_centre / image_height
    regions = get_region_slices(dataset_name)
    if not regions:
        return "body"
    for name, (start, end) in regions.items():
        if start <= y_frac < end:
            return name
    return "body"


def split_words_by_region(
    words: list[str],
    bboxes: list[list[int] | tuple[int, ...]],
    image_height: int,
    dataset_name: str,
) -> dict[str, list[str]]:
    """
    Group words into region lists by bbox y-position (FUNSD/SROIE best practice: spatial layout).
    words and bboxes must align 1:1. Returns dict region_name -> list of words in that region.
    """
    if not words or not bboxes or image_height <= 0:
        return {}
    n = min(len(words), len(bboxes))
    by_region: dict[str, list[str]] = {}
    for i in range(n):
        r = assign_bbox_to_region(
            bboxes[i] if isinstance(bboxes[i], (list, tuple)) else [0, 0, 0, 0],
            image_height,
            dataset_name,
        )
        by_region.setdefault(r, []).append(str(words[i]).strip())
    return by_region
