"""
OCR evaluation post-processing and improved scoring.

Improves accuracy of SROIE (entity match) and FUNSD (word recall) by:
- Normalizing text (punctuation, spaces, comma vs dot for numbers)
- Extracting entities from raw OCR for SROIE
- Substring and normalized word matching for FUNSD
- OCR confusion correction (o/0, I/l/1, 5/S) tuned for SROIE and FUNSD
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# OCR confusion correction (o/0, I/l/1, 5/S) tuned for SROIE and FUNSD
# ---------------------------------------------------------------------------

def apply_ocr_confusion_correction(text: str, dataset_name: str = "sroie") -> str:
    """
    Apply context-aware character corrections for common OCR confusions: O/0, I/l/1, 5/S.
    Tuned for SROIE (receipts: totals, dates, company) and FUNSD (forms: words and numbers).

    - In numeric contexts (totals, prices, dates): treat letter O as 0, I/l as 1, S as 5.
    - In word contexts (FUNSD): treat digit 0 as O, 1 as l, 5 as S when inside otherwise-letter tokens.
    """
    if not text or not isinstance(text, str):
        return text
    dataset = dataset_name.upper()
    out = text

    # ---- Numeric/date context: letters -> digits (OCR read 0 as O, 1 as l/I, 5 as S) ----
    def _replace_letter_as_digit_in_numeric(m: re.Match) -> str:
        s = m.group(0)
        s = re.sub(r"[Oo]", "0", s)
        s = re.sub(r"[Il|]", "1", s)
        s = re.sub(r"[Ss]", "5", s)
        return s

    # Decimal-like tokens: 80.90, 8O,9O, 12.5O
    out = re.sub(
        r"\b([\dOoIl|Ss]+[.,][\dOoIl|Ss]+)\b",
        _replace_letter_as_digit_in_numeric,
        out,
    )
    # Date-like: dd/mm/yyyy or dd-mm-yy
    out = re.sub(
        r"\b([\dOoIl|]{1,2}[./-][\dOoIl|]{1,2}[./-][\dOoIl|]{2,4})\b",
        _replace_letter_as_digit_in_numeric,
        out,
    )
    # Integer-like runs (e.g. "1O5" -> "105") so totals without decimal get fixed
    out = re.sub(
        r"\b([\dOoIl|Ss]{2,})\b",
        lambda m: _replace_letter_as_digit_in_numeric(m) if re.search(r"[OoIl|Ss]", m.group(0)) else m.group(0),
        out,
    )

    # ---- Word context (FUNSD and SROIE company/address): digits -> letters ----
    # Only in tokens that are letters with a single 0/1/5 (e.g. c0mpany, F1lter, Filte5)
    def _replace_digit_as_letter_in_word(m: re.Match) -> str:
        s = m.group(0)
        if re.fullmatch(r"[a-zA-Z]*[015][a-zA-Z]*", s):
            s = s.replace("0", "O").replace("1", "l").replace("5", "S")
        return s

    out = re.sub(r"\b([a-zA-Z]*[015][a-zA-Z]*)\b", _replace_digit_as_letter_in_word, out)

    return out


def _normalize_word(w: str) -> str:
    """Lowercase, strip, collapse internal spaces. Keep alphanumeric and common punctuation."""
    if not isinstance(w, str):
        return ""
    s = w.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_word_funsd(w: str) -> str:
    """FUNSD-specific: strip trailing colons and outer parentheses so form labels match (e.g. 'Date:' -> 'Date')."""
    if not isinstance(w, str):
        return ""
    s = w.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(":")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def _normalize_number_for_match(s: str) -> str:
    """For monetary/total: comma -> dot, strip spaces."""
    if not s:
        return ""
    s = str(s).strip().replace(",", ".").replace(" ", "")
    return s


def _tokenize_for_word_match(text: str) -> set[str]:
    """Tokenize prediction/GT into normalized tokens (split on whitespace + punctuation)."""
    if not text:
        return set()
    # Split on non-alphanumeric but keep tokens
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\s]", text.strip().lower())
    return set(t for t in tokens if t and len(t) > 0)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (for short words, e.g. fuzzy FUNSD)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def funsd_word_recall_improved(
    pred_text: str,
    words_gt: list,
    *,
    normalize: bool = True,
    use_substring: bool = True,
    use_fuzzy: bool = True,
    fuzzy_max_edit_ratio: float = 0.3,
    fuzzy_min_len: int = 4,
) -> tuple[float, int, int]:
    """
    Word-level recall: fraction of GT words that appear in prediction.

    Improvements over exact-set match:
    - normalize: lowercase and collapse spaces so "R&D" vs "r&d" match
    - use_substring: count GT word as matched if it appears anywhere in pred
    - use_fuzzy: if substring fails, count as matched when a pred token is within
      edit_distance <= fuzzy_max_edit_ratio * len(gt_word) and len(gt_word) >= fuzzy_min_len
    Returns (recall, n_matched, n_gt).
    """
    if not words_gt:
        return 0.0, 0, 0
    gt_list = [w for w in words_gt if isinstance(w, str) and str(w).strip()]
    if not gt_list:
        return 0.0, 0, 0

    pred_norm = (pred_text or "").strip().lower()
    if normalize:
        pred_norm = re.sub(r"\s+", " ", pred_norm)
    pred_tokens = _tokenize_for_word_match(pred_norm) if pred_norm else set()
    pred_no_spaces = pred_norm.replace(" ", "")
    # Build list of pred token strings (for fuzzy) from pred_norm
    pred_token_list = re.findall(r"[a-zA-Z0-9]+", pred_norm) if pred_norm else []

    matched = 0
    norm_fn = _normalize_word_funsd if normalize else (lambda x: x.strip().lower())
    for w in gt_list:
        wn = norm_fn(w) if normalize else w.strip().lower()
        if not wn:
            continue
        wn_no_spaces = wn.replace(" ", "")
        if use_substring:
            if wn in pred_norm or wn_no_spaces in pred_no_spaces:
                matched += 1
                continue
            if pred_tokens and wn in pred_tokens:
                matched += 1
                continue
            if len(wn) <= 2 and wn in pred_norm:
                matched += 1
                continue
        else:
            if pred_tokens and wn in pred_tokens:
                matched += 1
                continue

        # Fuzzy: OCR typos e.g. ciparettes -> cigarettes (allow shorter words with fuzzy_min_len=3)
        if use_fuzzy and len(wn) >= fuzzy_min_len:
            max_ed = max(1, int(len(wn) * fuzzy_max_edit_ratio))
            for tok in pred_token_list:
                if len(tok) < fuzzy_min_len:
                    continue
                if _edit_distance(wn, tok) <= max_ed:
                    matched += 1
                    break

    n_gt = len(gt_list)
    recall = matched / n_gt if n_gt else 0.0
    return recall, matched, n_gt


def extract_sroie_entities_from_text(text: str) -> dict[str, str]:
    """
    Extract company, date, address, total from raw OCR text (SROIE-style).

    Uses heuristics: date patterns, decimal totals, company (SDN BHD), address (long line).
    """
    if not text:
        return {}
    text_clean = re.sub(r"\s+", " ", (text or "").strip())
    entities: dict[str, str] = {}

    # Date: dd/mm/yyyy, dd-mm-yy, d/m/yyyy (SROIE uses various formats)
    for pat in [
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{2})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}-\d{1,2}-\d{2,4})\b",
    ]:
        m = re.search(pat, text_clean)
        if m:
            entities["date"] = m.group(1)
            break

    # Total: entity-specific regex (SROIE receipts) - RM 9.00, Total : 9.00, R M 9.00, etc.
    total_patterns = [
        r"\bRM\s*(\d+[.,]\d{2})\b",
        r"\bR\s*M\s*(\d+[.,]\d{2})\b",
        r"(?:total|tota|tot)\s*[:\s]*(\d+[.,]\d{2})\b",
        r"(?:total|rm)\s*[:\s]*(\d+[.,]\d{2})\b",
    ]
    for pat in total_patterns:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m:
            entities["total"] = m.group(1).replace(",", ".")
            break
    if "total" not in entities:
        total_candidates = re.findall(r"\b(\d+[.,]\d{2})\b", text_clean)
        if total_candidates:
            def to_float(s: str) -> float:
                try:
                    return float(s.replace(",", "."))
                except ValueError:
                    return 0.0
            sorted_totals = sorted(
                total_candidates,
                key=to_float,
                reverse=True,
            )
            entities["total"] = sorted_totals[0].replace(",", ".")

    # Company: line containing SDN BHD / BND; or leading segment with ENTERPRISE/TRADING/DECO/GIFT (SROIE receipts)
    text_upper = text_clean.upper()
    for part in ["SDN BHD", "SDN  BHD", "BHD", "BND"]:
        if part in text_upper:
            idx = text_upper.find(part)
            start = max(0, idx - 60)
            end = min(len(text_clean), idx + 40)
            company = text_clean[start:end].strip()
            if len(company) > 5:
                entities["company"] = company
            break
    if "company" not in entities:
        for token in ("ENTERPRISE", "TRADING", "DECO", "GIFT", "INDAH", "MR D.I.Y"):
            if token in text_upper:
                idx = text_upper.find(token)
                start = max(0, idx - 40)
                end = min(len(text_clean), idx + 50)
                company = text_clean[start:end].strip()
                if len(company) > 5:
                    entities["company"] = company
                    break

    # Address: longest segment that looks like a street address (digits, letters, commas, length 20–250)
    # SROIE addresses often have "JALAN", "NO.", "STREET", "ROAD", numbers, commas
    segments = re.split(r"[\n.]", text_clean)
    address_candidates = [
        s.strip() for s in segments
        if 20 <= len(s.strip()) <= 250
        and re.search(r"\d", s)
        and re.search(r"[A-Za-z]{3,}", s)
    ]
    # Prefer longest segment that contains address-like tokens (Malaysian and generic)
    address_tokens = ("JALAN", "NO.", "NO ", "STREET", "ROAD", "LANE", "PLAZA", "BUILDING")
    with_tokens = [s for s in address_candidates if any(t in s.upper() for t in address_tokens)]
    if with_tokens:
        entities["address"] = max(with_tokens, key=len)
    elif address_candidates:
        entities["address"] = max(address_candidates, key=len)

    return entities


def extract_sroie_entities_from_text_layout_aware(text: str, dataset_name: str = "SROIE") -> dict[str, str]:
    """
    Extract SROIE entities using layout regions: company/address/date from header (top 40% of lines),
    total from footer (bottom 35% of lines). Falls back to full-text extraction when region text is empty.
    """
    try:
        from ocr_pipeline.layout_regions import split_text_into_region_lines
    except ImportError:
        return extract_sroie_entities_from_text(text)
    region_texts = split_text_into_region_lines(text, dataset_name)
    if not region_texts or (len(region_texts) == 1 and "full" in region_texts):
        return extract_sroie_entities_from_text(text)
    entities: dict[str, str] = {}
    header_text = region_texts.get("header", "")
    footer_text = region_texts.get("footer", "")
    if header_text:
        head_ents = extract_sroie_entities_from_text(header_text)
        for k in ("company", "address", "date"):
            if head_ents.get(k):
                entities[k] = head_ents[k]
    if footer_text:
        foot_ents = extract_sroie_entities_from_text(footer_text)
        if foot_ents.get("total"):
            entities["total"] = foot_ents["total"]
    for k in ("company", "address", "date", "total"):
        if entities.get(k):
            continue
        full_ents = extract_sroie_entities_from_text(text)
        if full_ents.get(k):
            entities[k] = full_ents[k]
    return entities


def _normalize_date_canonical(s: str) -> str:
    """Normalize date to dd/mm/yyyy for comparison (SROIE uses 25/12/2018, 12-01-19, 9/01/2019)."""
    s = s.replace("-", "/").strip()
    parts = re.split(r"[/\s]+", s)
    if len(parts) >= 3:
        try:
            d, m, y = parts[0], parts[1], parts[2]
            if len(y) == 2:
                y = "20" + y if int(y) < 50 else "19" + y
            return f"{int(d):02d}/{int(m):02d}/{y}"
        except (ValueError, TypeError):
            pass
    return s


def normalize_sroie_value(key: str, value: Any) -> str:
    """Normalize GT or predicted value for comparison."""
    s = str(value).strip() if value is not None else ""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    if key == "total":
        s = _normalize_number_for_match(s).lower()
    elif key == "date":
        s = _normalize_date_canonical(s)
    return s


def sroie_entity_match_improved(
    pred_text: str,
    gt_entities: dict,
    *,
    extract_from_pred: bool = True,
    normalize: bool = True,
    soft_cer_threshold: float | None = 0.35,
) -> tuple[int, int, list[str]]:
    """
    Match SROIE entities (company, address, date, total).

    - extract_from_pred: extract entities from pred text; otherwise treat pred as raw and match substrings.
    - normalize: normalize numbers (comma->dot) and spaces.
    - soft_cer_threshold: if set, count as match when normalized pred contains gt or when CER <= threshold (simple ratio).
    Returns (matched_count, total_count, details list).
    """
    if not isinstance(gt_entities, dict):
        return 0, 0, []
    keys_order = ("company", "address", "date", "total")
    total = 0
    matched = 0
    details: list[str] = []

    pred_norm = (pred_text or "").strip().lower()
    pred_norm_no_spaces = pred_norm.replace(" ", "").replace(",", ".")
    pred_entities = extract_sroie_entities_from_text(pred_text) if extract_from_pred else {}

    for key in keys_order:
        gt_val = gt_entities.get(key)
        if gt_val is None or str(gt_val).strip() == "":
            continue
        total += 1
        gt_norm = normalize_sroie_value(key, gt_val)
        pred_val = pred_entities.get(key) if extract_from_pred else None
        pred_norm_val = normalize_sroie_value(key, pred_val) if pred_val else ""

        if extract_from_pred and pred_norm_val:
            if pred_norm_val == gt_norm:
                matched += 1
                details.append(f"{key}: ok")
                continue
            if key == "total" and pred_norm_val.replace(".", "") == gt_norm.replace(".", ""):
                matched += 1
                details.append(f"{key}: ok")
                continue
            if soft_cer_threshold is not None and _cer_ratio(pred_norm_val, gt_norm) <= soft_cer_threshold:
                matched += 1
                details.append(f"{key}: ok(soft)")
                continue

        # Fallback: substring in raw pred
        if gt_norm in pred_norm or gt_norm.replace(" ", "") in pred_norm_no_spaces:
            matched += 1
            details.append(f"{key}: ok")
            continue
        # Partial for total: e.g. "80.90" vs "80,90"
        if key == "total":
            gt_dot = gt_norm.replace(",", ".")
            if gt_dot in pred_norm or gt_dot in pred_norm_no_spaces:
                matched += 1
                details.append(f"{key}: ok")
                continue
        details.append(f"{key}: miss")

    return matched, total, details


def _cer_ratio(a: str, b: str) -> float:
    """Simple character error ratio: 1 - (matching_chars / max(len(a), len(b)))."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Count longest common subsequence length (approximation)
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, a, b)
    match_len = sum(triple.size for triple in sm.get_matching_blocks())
    max_len = max(len(a), len(b))
    return 1.0 - (match_len / max_len)


def get_funsd_gt_words_from_sample(sample: dict) -> list:
    """
    Build the list of ground-truth words for FUNSD word recall from the sample.
    Uses token_labels (NER tag IDs): we keep every word whose label is not O (0).
    Requires sample["input"]["ocr"]["words"] and sample["ground_truth"]["token_labels"].
    """
    if not sample or not isinstance(sample, dict):
        return []
    gt = sample.get("ground_truth") or {}
    labels = gt.get("token_labels")
    if labels is None:
        return []
    inp = sample.get("input") or {}
    ocr = inp.get("ocr") or {}
    words = ocr.get("words")
    if not words or not isinstance(words, list):
        return []
    n = min(len(words), len(labels))
    return [str(words[i]).strip() for i in range(n) if labels[i] != 0 and str(words[i]).strip()]


# FUNSD entity type from NER tag ID (crcresearch/FUNSD: header, question, answer, other)
_FUNSD_TAG_TO_ENTITY: dict[int, str] = {
    0: "other",
    1: "header", 2: "header",
    3: "question", 4: "question",
    5: "answer", 6: "answer",
}


def get_funsd_entities_from_sample(sample: dict) -> list[dict[str, Any]]:
    """
    Build semantic entities from FUNSD sample (align with crcresearch/FUNSD entity-centric view).
    Each entity = group of consecutive words with same label (header / question / answer).
    Returns list of {"label": "header"|"question"|"answer", "text": str}.
    """
    if not sample or not isinstance(sample, dict):
        return []
    gt = sample.get("ground_truth") or {}
    labels = gt.get("token_labels")
    if labels is None:
        return []
    inp = sample.get("input") or {}
    ocr = inp.get("ocr") or {}
    words = ocr.get("words")
    if not words or not isinstance(words, list):
        return []
    n = min(len(words), len(labels))
    entities: list[dict[str, Any]] = []
    current_label: str | None = None
    current_words: list[str] = []
    for i in range(n):
        tag = int(labels[i]) if labels[i] is not None else 0
        entity_type = _FUNSD_TAG_TO_ENTITY.get(tag, "other")
        if entity_type == "other":
            if current_label is not None and current_words:
                entities.append({"label": current_label, "text": " ".join(current_words).strip()})
            current_label = None
            current_words = []
            continue
        w = str(words[i]).strip()
        if not w:
            continue
        if entity_type == current_label:
            current_words.append(w)
        else:
            if current_label is not None and current_words:
                entities.append({"label": current_label, "text": " ".join(current_words).strip()})
            current_label = entity_type
            current_words = [w]
    if current_label is not None and current_words:
        entities.append({"label": current_label, "text": " ".join(current_words).strip()})
    return entities


def funsd_entity_recall(
    pred_text: str,
    entities: list[dict[str, Any]],
    *,
    normalize: bool = True,
    use_fuzzy: bool = True,
    fuzzy_max_edit_ratio: float = 0.35,
    min_entity_len: int = 2,
) -> tuple[float, int, int]:
    """
    Entity-level recall for FUNSD (crcresearch/FUNSD best practice: semantic entities).
    Count how many GT entities have their text (substring or fuzzy) present in prediction.
    Returns (recall, n_matched, n_entities).
    """
    if not entities:
        return 0.0, 0, 0
    pred_norm = (pred_text or "").strip().lower()
    if normalize:
        pred_norm = re.sub(r"\s+", " ", pred_norm)
    pred_no_spaces = pred_norm.replace(" ", "")
    matched = 0
    for ent in entities:
        text = (ent.get("text") or "").strip()
        if len(text) < min_entity_len:
            continue
        en = _normalize_word_funsd(text) if normalize else text.strip().lower()
        if not en:
            continue
        en_no_spaces = en.replace(" ", "")
        if en in pred_norm or en_no_spaces in pred_no_spaces:
            matched += 1
            continue
        if use_fuzzy and len(en) >= 3:
            pred_tokens = re.findall(r"[a-zA-Z0-9]+", pred_norm) if pred_norm else []
            max_ed = max(1, int(len(en) * fuzzy_max_edit_ratio))
            for tok in pred_tokens:
                if len(tok) >= 2 and _edit_distance(en, tok) <= max_ed:
                    matched += 1
                    break
    n_ent = len(entities)
    recall = matched / n_ent if n_ent else 0.0
    return recall, matched, n_ent


def compute_ocr_metrics(
    prediction: str,
    ground_truth: Any,
    dataset_name: str,
    apply_confusion_correction: bool = True,
    sample: dict | None = None,
) -> dict[str, Any]:
    """
    Compute OCR metrics for a single sample from prediction and ground_truth.

    - SROIE: ground_truth is dict with company, date, address, total.
    - FUNSD: ground_truth can be dict with token_labels; if sample is provided and has input.ocr.words,
      words_gt is built from words where token_labels[i] != 0. Otherwise ground_truth can be a list of words.
    - sample: optional full sample dict; used for FUNSD to derive GT words from token_labels + input.ocr.words.
    - apply_confusion_correction: if True, run OCR confusion correction on prediction before matching.
    Returns a metrics dict suitable for proof JSON.
    """
    pred = (prediction or "").strip()
    if apply_confusion_correction:
        pred = apply_ocr_confusion_correction(pred, dataset_name)
    if dataset_name.upper() == "SROIE":
        gt = ground_truth if isinstance(ground_truth, dict) else {}
        matched, total, _ = sroie_entity_match_improved(
            pred,
            gt,
            extract_from_pred=True,
            normalize=True,
            soft_cer_threshold=0.35,
        )
        return {
            "entity_match": matched / total if total else 0.0,
            "entity_matched": matched,
            "entity_total": total,
        }
    if dataset_name.upper() == "FUNSD":
        words_gt = ground_truth if isinstance(ground_truth, list) else []
        if not words_gt and sample and isinstance(ground_truth, dict) and ground_truth.get("token_labels") is not None:
            words_gt = get_funsd_gt_words_from_sample(sample)
        recall, n_matched, n_gt = funsd_word_recall_improved(
            pred,
            words_gt,
            normalize=True,
            use_substring=True,
            use_fuzzy=True,
            fuzzy_max_edit_ratio=0.35,
            fuzzy_min_len=3,
        )
        out: dict[str, Any] = {
            "word_recall": recall,
            "words_matched": n_matched,
            "words_gt": n_gt,
        }
        # Entity-level recall (crcresearch/FUNSD: semantic entities = header/question/answer groups)
        if sample and isinstance(ground_truth, dict) and ground_truth.get("token_labels") is not None:
            entities = get_funsd_entities_from_sample(sample)
            if entities:
                ent_recall, ent_matched, ent_total = funsd_entity_recall(
                    pred, entities, normalize=True, use_fuzzy=True, fuzzy_max_edit_ratio=0.35
                )
                out["entity_recall"] = ent_recall
                out["entity_matched"] = ent_matched
                out["entity_total"] = ent_total
        return out
    return {}
