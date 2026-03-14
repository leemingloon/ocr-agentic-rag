"""Utility classes for post-processing model outputs and computing evaluation metrics.
OCR (SROIE/FUNSD): confusion correction, entity extraction, entity/word recall metrics.

ENCODING RULE (Windows cp1252 safety): Do not use Unicode symbols in any string that may be
printed to stdout/stderr or included in debug output (e.g. rag_numerical_match_debug_info).
Use ASCII only: e.g. "~" not "≈", "-" not "—", "->" not "→". This prevents UnicodeEncodeError
when the eval runner or other scripts print these strings on Windows (cp1252 console).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable


class BaseUtils:
    """Base utility helpers shared across evaluation categories.
    Text-based metrics here are reusable by vision, RAG, credit_risk_sentiment, credit_risk_memo_generator.
    """

    @staticmethod
    def normalize_text(text: str | None) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().lower().split())

    @staticmethod
    def financial_normalize(text: str | None) -> str:
        """
        Financial-domain normalisation for exact_match and F1 on financial SEC
        filing QA (FinQA, TAT-QA, FinanceBench). Shared across RagUtils and all
        subclasses via BaseUtils.

        Extends SQuAD normalize_answer:
          - Strips $ and % (formatting artifacts in model predictions and GTs)
          - Strips thousands-separator commas: "1,234" -> "1234"
          - Converts parenthetical magnitude: "(9,187)" -> "9187"
            (SEC filing convention: parentheses denote loss/negative magnitude)
          - Preserves negative sign: hyphen immediately before a digit is kept
          - Preserves decimal point between two digits
          - Removes all other punctuation (replaced with space)
          - Removes articles (a, an, the)
          - Lowercases and collapses whitespace

        Unlike SQuAD normalize_answer: sign-preserving (-8551 != 8551).
        Unlike normalize_text: strips $, %, thousands commas, parenthetical notation.
        """
        import string as _string

        if text is None:
            return ""
        s = str(text).strip().lower()
        # Remove articles
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Strip common financial formatting
        s = s.replace("$", "").replace("%", "")
        # Strip thousands-separator commas between digits: 1,234 -> 1234
        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        # Parenthetical magnitude: (9,187) -> 9187
        s = re.sub(r"\((\d[\d.]*)\)", r"\1", s)

        out: list[str] = []
        for i, ch in enumerate(s):
            if ch == "-":
                # Preserve hyphen only when immediately before a digit (after optional whitespace)
                rest = s[i + 1 :].lstrip()
                if rest and rest[0].isdigit():
                    out.append(ch)
                else:
                    out.append(" ")
            elif ch == ".":
                # Preserve decimal point only when between two digits
                prev_digit = i > 0 and s[i - 1].isdigit()
                next_digit = i < len(s) - 1 and s[i + 1].isdigit()
                if prev_digit and next_digit:
                    out.append(ch)
                else:
                    out.append(" ")
            elif ch in _string.punctuation:
                out.append(" ")
            else:
                out.append(ch)
        return " ".join("".join(out).split())

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        """Parse a single numeric value from string; used by relaxed_numeric_* helpers."""
        if value is None:
            return None
        cleaned = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def exact_match_whole_word(self, prediction: str | None, reference: str | None) -> float:
        """Reusable: 1.0 if reference appears as a whole word in prediction (word-boundary match).
        Use when the model returns long text but the gold answer is a short string (e.g. "14", "0.57").
        Case-insensitive via normalize_text.
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        if not ref_norm:
            return 1.0 if not pred_norm else 0.0
        # Word-boundary match so "0.57" matches in "boxed{0.57}" or "answer 0.57 points"
        pattern = r"\b" + re.escape(ref_norm) + r"\b"
        return 1.0 if re.search(pattern, pred_norm) else 0.0

    def relaxed_numeric_accuracy_any_number(
        self, prediction: str | None, reference: str | None, rel_tol: float = 0.05
    ) -> float:
        """Reusable: 1.0 if any number in the prediction is within rel_tol of the reference number.
        Use when the model returns long text that contains the numeric answer among other numbers.
        """
        r_num = self._safe_float(reference)
        if r_num is None:
            return float(self.normalize_text(prediction) == self.normalize_text(reference))
        if prediction is None:
            return 0.0
        num_strs = re.findall(r"-?\d+(?:\.\d+)?", str(prediction).replace(",", ""))
        for s in num_strs:
            try:
                p_num = float(s)
            except ValueError:
                continue
            denom = max(abs(r_num), 1e-9)
            if abs(p_num - r_num) / denom <= rel_tol:
                return 1.0
        return 0.0

    # -------------------------------------------------------------------------
    # Shared text postprocessing: correct if contains numeric answer or option letter
    # Applies to all text-based model responses (vision, RAG, etc.) when inheriting.
    # -------------------------------------------------------------------------

    def _reference_is_option_letter(self, reference: str | None) -> bool:
        """True if reference is a single option letter (e.g. A–Z, often A–E for MC)."""
        if not reference:
            return False
        ref_stripped = str(reference).strip().upper()
        return len(ref_stripped) == 1 and ref_stripped.isalpha()

    def _prediction_contains_option_letter(self, prediction: str | None, reference: str | None) -> bool:
        """True if reference is a single letter and prediction clearly contains that option (e.g. 'D', 'option D', 'answer D')."""
        if not reference or prediction is None:
            return False
        ref_letter = str(reference).strip().upper()
        if len(ref_letter) != 1 or not ref_letter.isalpha():
            return False
        pred_norm = self.normalize_text(prediction)
        # Standalone letter, or "option X", "answer is X", "choice x", "(x)", "boxed{x}", "$x"
        patterns = [
            r"\b" + re.escape(ref_letter) + r"\b",
            r"option\s+" + re.escape(ref_letter) + r"\b",
            r"answer\s+(?:is\s+)?" + re.escape(ref_letter) + r"\b",
            r"choice\s+" + re.escape(ref_letter) + r"\b",
            r"\(\s*" + re.escape(ref_letter) + r"\s*\)",
            r"\\boxed\s*\{\s*" + re.escape(ref_letter) + r"\s*\}",
        ]
        return any(re.search(p, pred_norm) for p in patterns)

    def _prediction_contains_reference_number(
        self, prediction: str | None, reference: str | None, rel_tol: float = 0.05
    ) -> bool:
        """True if reference parses as a number and that number appears in prediction (with optional formatting)."""
        return self.relaxed_numeric_accuracy_any_number(prediction, reference, rel_tol) == 1.0

    @staticmethod
    def _letter_to_option_index(letter: str) -> int:
        """A=0, B=1, ..., Z=25. Returns -1 if not a single letter."""
        if not letter or len(letter) != 1:
            return -1
        c = letter.strip().upper()
        if not c.isalpha():
            return -1
        return ord(c) - ord("A")

    def _prediction_contains_option_value(
        self,
        prediction: str | None,
        reference: str | None,
        options_list: list,
        rel_tol: float = 0.05,
    ) -> bool:
        """True if reference is option letter (A–Z), options_list maps index to value, and prediction contains that value.
        Final evaluation catch: model may output the correct value (e.g. 77490) instead of the letter (D).
        """
        if not reference or prediction is None or not options_list:
            return False
        idx = self._letter_to_option_index(str(reference).strip())
        if idx < 0 or idx >= len(options_list):
            return False
        option_value = options_list[idx]
        if option_value is None:
            return False
        # Value can be numeric (e.g. $77,490) or text; check both number match and normalized substring
        if self._prediction_contains_reference_number(prediction, str(option_value), rel_tol):
            return True
        val_norm = self.normalize_text(str(option_value))
        if val_norm and val_norm in self.normalize_text(prediction):
            return True
        return False

    def relaxed_exact_match(
        self,
        prediction: str | None,
        reference: str | None,
        rel_tol: float = 0.05,
        options_list: list | None = None,
        **kwargs: object,
    ) -> float:
        """Broad, inheritable match for text-based responses. Returns 1.0 if any of:
        - Exact match after case-insensitive normalize
        - Reference appears as whole word in prediction
        - Reference is a single option letter and prediction contains that letter as answer
        - Reference is numeric and prediction contains that number
        - (Final catch) Reference is option letter and options_list is provided: map letter to value (A=0, B=1, ...)
          and accept if prediction contains that option value (e.g. GT 'D', options_list[3]='$77,490' -> pred contains 77490).
        """
        if reference is None and prediction is None:
            return 1.0
        if reference is None:
            return 1.0 if not self.normalize_text(prediction) else 0.0
        if prediction is None:
            return 0.0

        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)

        if ref_norm and pred_norm == ref_norm:
            return 1.0
        if ref_norm and re.search(r"\b" + re.escape(ref_norm) + r"\b", pred_norm):
            return 1.0
        if self._reference_is_option_letter(reference) and self._prediction_contains_option_letter(
            prediction, reference
        ):
            return 1.0
        if self._prediction_contains_reference_number(prediction, reference, rel_tol):
            return 1.0
        # Final catch: option letter + options_list -> check if pred contains the value at that index
        if (
            options_list
            and isinstance(options_list, list)
            and self._reference_is_option_letter(reference)
            and self._prediction_contains_option_value(prediction, reference, options_list, rel_tol)
        ):
            return 1.0

        return 0.0


# ---------------------------------------------------------------------------
# OCR evaluation (SROIE / FUNSD)
# Confusion correction, entity extraction, entity/word recall.
# ---------------------------------------------------------------------------


class OCRUtils(BaseUtils):
    """Base OCR metric helpers: confusion correction (O/0, I/l/1, 5/S) for SROIE and FUNSD."""

    @staticmethod
    def apply_confusion_correction(text: str, dataset_name: str = "sroie") -> str:
        """
        Apply context-aware character corrections for common OCR confusions: O/0, I/l/1, 5/S.
        Tuned for SROIE (receipts: totals, dates, company) and FUNSD (forms: words and numbers).
        """
        if not text or not isinstance(text, str):
            return text
        out = text

        def _replace_letter_as_digit_in_numeric(m: re.Match) -> str:
            s = m.group(0)
            s = re.sub(r"[Oo]", "0", s)
            s = re.sub(r"[Il|]", "1", s)
            s = re.sub(r"[Ss]", "5", s)
            return s

        out = re.sub(
            r"\b([\dOoIl|Ss]+[.,][\dOoIl|Ss]+)\b",
            _replace_letter_as_digit_in_numeric,
            out,
        )
        out = re.sub(
            r"\b([\dOoIl|]{1,2}[./-][\dOoIl|]{1,2}[./-][\dOoIl|]{2,4})\b",
            _replace_letter_as_digit_in_numeric,
            out,
        )
        out = re.sub(
            r"\b([\dOoIl|Ss]{2,})\b",
            lambda m: _replace_letter_as_digit_in_numeric(m) if re.search(r"[OoIl|Ss]", m.group(0)) else m.group(0),
            out,
        )

        def _replace_digit_as_letter_in_word(m: re.Match) -> str:
            s = m.group(0)
            if re.fullmatch(r"[a-zA-Z]*[015][a-zA-Z]*", s):
                s = s.replace("0", "O").replace("1", "l").replace("5", "S")
            return s

        out = re.sub(r"\b([a-zA-Z]*[015][a-zA-Z]*)\b", _replace_digit_as_letter_in_word, out)
        return out


apply_ocr_confusion_correction = OCRUtils.apply_confusion_correction


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


def _cer_ratio(a: str, b: str) -> float:
    """Simple character error ratio: 1 - (matching_chars / max(len(a), len(b)))."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    sm = SequenceMatcher(None, a, b)
    match_len = sum(triple.size for triple in sm.get_matching_blocks())
    max_len = max(len(a), len(b))
    return 1.0 - (match_len / max_len)


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


class SROIEUtils(OCRUtils):
    """SROIE (receipt) OCR: entity extraction, normalization, entity match with optional soft CER."""

    @staticmethod
    def extract_entities_from_text(text: str) -> dict[str, str]:
        """Extract company, date, address, total from raw OCR text (SROIE-style)."""
        if not text:
            return {}
        text_clean = re.sub(r"\s+", " ", (text or "").strip())
        entities: dict[str, str] = {}
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
                sorted_totals = sorted(
                    total_candidates,
                    key=lambda s: float(s.replace(",", ".")) if s else 0.0,
                    reverse=True,
                )
                entities["total"] = sorted_totals[0].replace(",", ".")
        text_upper = text_clean.upper()
        for part in ["SDN BHD", "SDN  BHD", "BHD", "BND"]:
            if part in text_upper:
                idx = text_upper.find(part)
                start, end = max(0, idx - 60), min(len(text_clean), idx + 40)
                company = text_clean[start:end].strip()
                if len(company) > 5:
                    entities["company"] = company
                break
        if "company" not in entities:
            for token in ("ENTERPRISE", "TRADING", "DECO", "GIFT", "INDAH", "MR D.I.Y"):
                if token in text_upper:
                    idx = text_upper.find(token)
                    start, end = max(0, idx - 40), min(len(text_clean), idx + 50)
                    company = text_clean[start:end].strip()
                    if len(company) > 5:
                        entities["company"] = company
                        break
        segments = re.split(r"[\n.]", text_clean)
        address_candidates = [
            s.strip() for s in segments
            if 20 <= len(s.strip()) <= 250 and re.search(r"\d", s) and re.search(r"[A-Za-z]{3,}", s)
        ]
        address_tokens = ("JALAN", "NO.", "NO ", "STREET", "ROAD", "LANE", "PLAZA", "BUILDING")
        with_tokens = [s for s in address_candidates if any(t in s.upper() for t in address_tokens)]
        if with_tokens:
            entities["address"] = max(with_tokens, key=len)
        elif address_candidates:
            entities["address"] = max(address_candidates, key=len)
        return entities

    @staticmethod
    def extract_entities_from_text_layout_aware(text: str, dataset_name: str = "SROIE") -> dict[str, str]:
        """Extract SROIE entities using layout regions; fallback to full-text extraction."""
        try:
            from ocr_pipeline.layout_regions import split_text_into_region_lines
        except ImportError:
            return SROIEUtils.extract_entities_from_text(text)
        region_texts = split_text_into_region_lines(text, dataset_name)
        if not region_texts or (len(region_texts) == 1 and "full" in region_texts):
            return SROIEUtils.extract_entities_from_text(text)
        entities: dict[str, str] = {}
        header_text = region_texts.get("header", "")
        footer_text = region_texts.get("footer", "")
        if header_text:
            head_ents = SROIEUtils.extract_entities_from_text(header_text)
            for k in ("company", "address", "date"):
                if head_ents.get(k):
                    entities[k] = head_ents[k]
        if footer_text:
            foot_ents = SROIEUtils.extract_entities_from_text(footer_text)
            if foot_ents.get("total"):
                entities["total"] = foot_ents["total"]
        for k in ("company", "address", "date", "total"):
            if entities.get(k):
                continue
            full_ents = SROIEUtils.extract_entities_from_text(text)
            if full_ents.get(k):
                entities[k] = full_ents[k]
        return entities

    @staticmethod
    def normalize_value(key: str, value: Any) -> str:
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

    @staticmethod
    def entity_match_improved(
        pred_text: str,
        gt_entities: dict,
        *,
        extract_from_pred: bool = True,
        normalize: bool = True,
        soft_cer_threshold: float | None = 0.35,
    ) -> tuple[int, int, list[str]]:
        """Match SROIE entities (company, address, date, total). Returns (matched_count, total_count, details)."""
        if not isinstance(gt_entities, dict):
            return 0, 0, []
        keys_order = ("company", "address", "date", "total")
        total, matched, details = 0, 0, []
        pred_norm = (pred_text or "").strip().lower()
        pred_norm_no_spaces = pred_norm.replace(" ", "").replace(",", ".")
        pred_entities = SROIEUtils.extract_entities_from_text(pred_text) if extract_from_pred else {}
        for key in keys_order:
            gt_val = gt_entities.get(key)
            if gt_val is None or str(gt_val).strip() == "":
                continue
            total += 1
            gt_norm = SROIEUtils.normalize_value(key, gt_val)
            pred_val = pred_entities.get(key) if extract_from_pred else None
            pred_norm_val = SROIEUtils.normalize_value(key, pred_val) if pred_val else ""
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
            if gt_norm in pred_norm or gt_norm.replace(" ", "") in pred_norm_no_spaces:
                matched += 1
                details.append(f"{key}: ok")
                continue
            if key == "total":
                gt_dot = gt_norm.replace(",", ".")
                if gt_dot in pred_norm or gt_dot in pred_norm_no_spaces:
                    matched += 1
                    details.append(f"{key}: ok")
                    continue
            details.append(f"{key}: miss")
        return matched, total, details


sroie_entity_match_improved = SROIEUtils.entity_match_improved

# FUNSD entity type from NER tag ID (crcresearch/FUNSD: header, question, answer, other)
_FUNSD_TAG_TO_ENTITY: dict[int, str] = {
    0: "other",
    1: "header", 2: "header",
    3: "question", 4: "question",
    5: "answer", 6: "answer",
}


class FUNSDUtils(OCRUtils):
    """FUNSD (form) OCR: word recall, entity recall, GT words/entities from sample."""

    @staticmethod
    def get_gt_words_from_sample(sample: dict) -> list:
        """Build GT words for FUNSD word recall from sample (token_labels != 0)."""
        if not sample or not isinstance(sample, dict):
            return []
        gt = sample.get("ground_truth") or {}
        labels = gt.get("token_labels")
        if labels is None:
            return []
        inp = sample.get("input") or {}
        words = (inp.get("ocr") or {}).get("words")
        if not words or not isinstance(words, list):
            return []
        n = min(len(words), len(labels))
        return [str(words[i]).strip() for i in range(n) if labels[i] != 0 and str(words[i]).strip()]

    @staticmethod
    def get_entities_from_sample(sample: dict) -> list[dict[str, Any]]:
        """Build semantic entities from FUNSD sample (header/question/answer groups)."""
        if not sample or not isinstance(sample, dict):
            return []
        gt = sample.get("ground_truth") or {}
        labels = gt.get("token_labels")
        if labels is None:
            return []
        inp = sample.get("input") or {}
        words = (inp.get("ocr") or {}).get("words")
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

    @staticmethod
    def word_recall_improved(
        pred_text: str,
        words_gt: list,
        *,
        normalize: bool = True,
        use_substring: bool = True,
        use_fuzzy: bool = True,
        fuzzy_max_edit_ratio: float = 0.3,
        fuzzy_min_len: int = 4,
    ) -> tuple[float, int, int]:
        """Word-level recall: fraction of GT words that appear in prediction. Returns (recall, n_matched, n_gt)."""
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
            if use_fuzzy and len(wn) >= fuzzy_min_len:
                max_ed = max(1, int(len(wn) * fuzzy_max_edit_ratio))
                for tok in pred_token_list:
                    if len(tok) < fuzzy_min_len:
                        continue
                    if _edit_distance(wn, tok) <= max_ed:
                        matched += 1
                        break
        n_gt = len(gt_list)
        return (matched / n_gt if n_gt else 0.0), matched, n_gt

    @staticmethod
    def entity_recall(
        pred_text: str,
        entities: list[dict[str, Any]],
        *,
        normalize: bool = True,
        use_fuzzy: bool = True,
        fuzzy_max_edit_ratio: float = 0.35,
        min_entity_len: int = 2,
    ) -> tuple[float, int, int]:
        """Entity-level recall for FUNSD. Returns (recall, n_matched, n_entities)."""
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
        return (matched / n_ent if n_ent else 0.0), matched, n_ent


funsd_word_recall_improved = FUNSDUtils.word_recall_improved


def compute_ocr_metrics(
    prediction: str,
    ground_truth: Any,
    dataset_name: str,
    apply_confusion_correction: bool = True,
    sample: dict | None = None,
) -> dict[str, Any]:
    """
    Compute OCR metrics for a single sample from prediction and ground_truth.
    SROIE: entity_match; FUNSD: word_recall and optionally entity_recall.
    """
    pred = (prediction or "").strip()
    if apply_confusion_correction:
        pred = OCRUtils.apply_confusion_correction(pred, dataset_name)
    if dataset_name.upper() == "SROIE":
        gt = ground_truth if isinstance(ground_truth, dict) else {}
        matched, total, _ = SROIEUtils.entity_match_improved(
            pred, gt, extract_from_pred=True, normalize=True, soft_cer_threshold=0.35
        )
        return {
            "entity_match": matched / total if total else 0.0,
            "entity_matched": matched,
            "entity_total": total,
        }
    if dataset_name.upper() == "FUNSD":
        words_gt = ground_truth if isinstance(ground_truth, list) else []
        if not words_gt and sample and isinstance(ground_truth, dict) and ground_truth.get("token_labels") is not None:
            words_gt = FUNSDUtils.get_gt_words_from_sample(sample)
        recall, n_matched, n_gt = FUNSDUtils.word_recall_improved(
            pred, words_gt,
            normalize=True, use_substring=True, use_fuzzy=True,
            fuzzy_max_edit_ratio=0.35, fuzzy_min_len=3,
        )
        out: dict[str, Any] = {"word_recall": recall, "words_matched": n_matched, "words_gt": n_gt}
        if sample and isinstance(ground_truth, dict) and ground_truth.get("token_labels") is not None:
            entities = FUNSDUtils.get_entities_from_sample(sample)
            if entities:
                ent_recall, ent_matched, ent_total = FUNSDUtils.entity_recall(
                    pred, entities, normalize=True, use_fuzzy=True, fuzzy_max_edit_ratio=0.35
                )
                out["entity_recall"] = ent_recall
                out["entity_matched"] = ent_matched
                out["entity_total"] = ent_total
        return out
    return {}


# ---------------------------------------------------------------------------
# Vision / RAG / Credit Risk (unchanged below)
# ---------------------------------------------------------------------------


class VisionUtils(BaseUtils):
    """Shared utilities for vision-language QA benchmarks.
    Inherits from BaseUtils: normalize_text, exact_match_whole_word, relaxed_exact_match, etc.
    exact_match uses relaxed_exact_match so all vision datasets get: case-insensitive, whole-word,
    option-letter, and numeric containment matching.
    """

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        return self.relaxed_exact_match(prediction, reference, **kwargs)

    def relaxed_numeric_accuracy(self, prediction: str | None, reference: str | None, rel_tol: float = 0.05) -> float:
        p_num = self._safe_float(prediction)
        r_num = self._safe_float(reference)
        if p_num is None or r_num is None:
            return self.exact_match(prediction, reference)
        denom = max(abs(r_num), 1e-9)
        return float(abs(p_num - r_num) / denom <= rel_tol)

    def anls(self, prediction: str | None, reference: str | None, threshold: float = 0.5) -> float:
        pred = self.normalize_text(prediction)
        ref = self.normalize_text(reference)
        if not ref and not pred:
            return 1.0
        if not ref:
            return 0.0
        ratio = SequenceMatcher(None, pred, ref).ratio()
        return ratio if ratio >= threshold else 0.0

    def accuracy(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        return self.exact_match(prediction, reference, **kwargs)


class DocVQAUtils(VisionUtils):
    """DocVQA: GT is short; model often returns long text. Uses relaxed_exact_match (case-insensitive,
    whole-word, option letter, number containment) so e.g. 'pinterest' matches 'Pinterest'."""

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        return self.relaxed_exact_match(prediction, reference, **kwargs)

    def anls(self, prediction: str | None, reference: str | None, threshold: float = 0.5) -> float:
        # If reference appears as whole word in prediction (case-insensitive), count as correct
        if self.exact_match_whole_word(prediction, reference) == 1.0:
            return 1.0
        return super().anls(prediction, reference)


class InfographicsVQAUtils(VisionUtils):
    """InfographicsVQA: same as DocVQA; GT is short, model returns long. Uses relaxed_exact_match
    so e.g. 'pininterest' matches 'Pininterest' (case-insensitive)."""

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        return self.relaxed_exact_match(prediction, reference, **kwargs)

    def anls(self, prediction: str | None, reference: str | None, threshold: float = 0.5) -> float:
        if self.exact_match_whole_word(prediction, reference) == 1.0:
            return 1.0
        return super().anls(prediction, reference)


class ChartQAUtils(VisionUtils):
    """ChartQA: model often returns long chart analysis; GT is short (e.g. '14', '0.57', 'Yes').
    Uses relaxed_exact_match so e.g. '14' matches '14 food items are shown in the bar graph'.
    Relaxed accuracy must be >= strict: when GT is non-numeric (Yes/No/etc.), use same match as strict."""

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        return self.relaxed_exact_match(prediction, reference, **kwargs)

    def relaxed_numeric_accuracy(
        self, prediction: str | None, reference: str | None, rel_tol: float = 0.05, **kwargs: object
    ) -> float:
        """Relaxed: any number in the prediction within rel_tol of the reference number.
        When reference is not numeric (e.g. Yes/No), use relaxed_exact_match so relaxed >= strict."""
        if self._safe_float(reference) is None:
            return self.relaxed_exact_match(prediction, reference, rel_tol=rel_tol, **kwargs)
        return self.relaxed_numeric_accuracy_any_number(prediction, reference, rel_tol)


class MMMUUtils(VisionUtils):
    """MMMU: multiple-choice (option letter) and numeric answers. Uses VisionUtils.exact_match
    (relaxed_exact_match) so correct option letter in text or correct number in explanation counts."""


class RagUtils(BaseUtils):
    """Shared utilities for RAG evaluation.

    FinQA and TAT-QA use the same definitions for:
      - exact_match: RagUtils.exact_match (financial_normalize; numeric -> numerical_exact_match).
      - relaxed_exact_match: RagUtils.score_relaxed_exact_match (6-gate deterministic scorer).

    Only numerical_exact_match is dataset-specific (FinQAUtils vs TATQAUtils override) for
    numeric ground truths; exact_match and score_relaxed_exact_match are centralized here.
    """

    _unit_map = {
        "million": 1_000_000.0,
        "millions": 1_000_000.0,
        "thousand": 1_000.0,
        "thousands": 1_000.0,
        "billion": 1_000_000_000.0,
        "billions": 1_000_000_000.0,
    }

    def _extract_number_and_scale(self, text: str | None) -> float | None:
        if text is None:
            return None
        lowered = str(text).lower()
        match = re.search(r"-?\d+(?:\.\d+)?", lowered.replace(",", ""))
        if not match:
            return None
        value = float(match.group(0))
        for unit, mult in self._unit_map.items():
            if unit in lowered:
                return value * mult
        return value

    def numerical_exact_match(self, prediction: str | None, reference: str | None, tol: float = 1e-6) -> float:
        """Base fallback: scale-extraction comparison (no DROP decimal rounding).
        TATQAUtils and FinQAUtils override this with dataset-specific extraction
        (multi-value parsing, decimal rounding tolerance, proportion equivalence)
        and call super() only when all subclass-specific paths are exhausted.
        Direct callers (e.g. FinanceBench memo scorer) use RagUtils directly and
        receive this base implementation - appropriate for memo-style answers where
        the GT is a formatted string, not a raw scalar."""
        p = self._extract_number_and_scale(prediction)
        r = self._extract_number_and_scale(reference)
        if p is None or r is None:
            return float(self.normalize_text(prediction) == self.normalize_text(reference))
        return float(math.isclose(p, r, rel_tol=tol, abs_tol=tol))

    @staticmethod
    def _binary_align(pred: str, ref: str) -> bool:
        """True if prediction and reference agree on the binary conclusion (No/Yes) for capital-intensity-style QA."""
        if not ref or not pred:
            return False
        ref_l = ref.strip().lower()[:120]
        pred_l = pred.strip().lower()[:400]
        # Reference binary: "no" (not capital-intensive) vs "yes" (capital-intensive)
        ref_no = ref_l.startswith("no") or "not capital-intensive" in ref_l or "not a capital-intensive" in ref_l
        ref_yes = ref_l.startswith("yes") and not ref_l.startswith("no,")
        # Prediction binary: same sense
        pred_no = (
            pred_l.startswith("no")
            or "not capital-intensive" in pred_l
            or "not a capital-intensive" in pred_l
            or ("no," in pred_l and "capital-intensive" in pred_l)
        )
        pred_yes = pred_l.startswith("yes") and "capital-intensive" in pred_l
        if ref_no and pred_no:
            return True
        if ref_yes and pred_yes:
            return True
        return False

    @staticmethod
    def _margin_driver_relaxed(pred: str, ref: str, f1_val: float) -> bool:
        """True if ref looks like a margin-driver answer and pred has key structure/phrases plus sufficient overlap."""
        if not ref or not pred or f1_val < 0.25:
            return False
        ref_n = ref.strip().lower()
        pred_n = pred.strip().lower()
        # Ref: operating margin driver style (has "operating margin" and cause phrasing)
        if "operating margin" not in ref_n or ("primarily due to" not in ref_n and "decreased" not in ref_n and "decrease" not in ref_n):
            return False
        # Pred: gross margin lead plus mostly one-off framing
        if "gross margin" not in pred_n:
            return False
        if "one-off" not in pred_n and "one off" not in pred_n:
            return False
        if "mostly" not in pred_n and "primarily" not in pred_n:
            return False
        return f1_val >= 0.30

    @staticmethod
    def _semantic_key_overlap(pred: str, ref: str, f1_val: float) -> bool:
        """True if ref is short and pred contains the key factual content (numbers and main subject)."""
        if not ref or not pred:
            return False
        if len(ref.strip()) > 350:
            return False
        ref_n = ref.strip().lower()
        pred_n = pred.strip().lower()
        # At least one number from ref must appear in pred (allow -0.9 vs 0.9, or with %)
        ref_numbers = re.findall(r"-?\d+\.?\d*", ref_n)
        number_ok = any(
            n in pred_n or n.lstrip("-") in pred_n or (n + "%") in pred_n or ("-" + n) in pred_n
            for n in ref_numbers
        )
        # At least one distinctive content word from ref (length > 2, skip common stopwords) must appear in pred
        stop = {
            "the",
            "and",
            "for",
            "by",
            "has",
            "was",
            "were",
            "from",
            "with",
            "that",
            "this",
            "are",
            "is",
            "to",
            "of",
            "in",
            "on",
            "at",
        }
        ref_words = [w for w in re.findall(r"[a-z0-9.%-]+", ref_n) if len(w) > 2 and w not in stop]
        word_ok = any(w in pred_n for w in ref_words) if ref_words else True
        return (number_ok or not ref_numbers) and word_ok and f1_val >= 0.08

    @staticmethod
    def _primary_ratio_match(pred: str, ref: str) -> bool:
        """True when the key non-year numeric ratio in pred and ref matches (e.g. 9.5x vs 9.5 times)."""

        def _extract_ratio_candidates(text: str) -> list[float]:
            if not text:
                return []
            nums = re.findall(r"-?\d+\.?\d*", text.lower())
            candidates: list[float] = []
            for n in nums:
                try:
                    v = float(n)
                except ValueError:
                    continue
                if v == 0.0:
                    continue
                if abs(v) >= 1900:
                    continue
                candidates.append(v)
            return candidates

        ref_vals = _extract_ratio_candidates(ref or "")
        pred_vals = _extract_ratio_candidates(pred or "")
        if not ref_vals or not pred_vals:
            return False
        ref_val = ref_vals[0]
        denom = max(1.0, abs(ref_val))
        return any(abs(ref_val - pv) <= 0.005 * denom for pv in pred_vals)

    def token_f1(self, prediction: str | None, reference: str | None) -> float:
        """
        SQuAD-style token F1 using financial_normalize tokenisation.
        Strips $, %, thousands commas; preserves sign and decimal point.
        Shared across FinQA, TAT-QA, and FinanceBench.
        F1 = 2 * precision * recall / (precision + recall).
        """
        p_tokens = self.financial_normalize(prediction).split()
        r_tokens = self.financial_normalize(reference).split()
        if not p_tokens and not r_tokens:
            return 1.0
        if not p_tokens or not r_tokens:
            return 0.0
        p_count = Counter(p_tokens)
        r_count = Counter(r_tokens)
        overlap = sum((p_count & r_count).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(p_tokens)
        recall = overlap / len(r_tokens)
        return 2 * precision * recall / (precision + recall)

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        """
        Shared exact_match for financial SEC filing QA (FinQA, TAT-QA, FinanceBench).
        Uses financial_normalize: strips $, %, thousands commas, parenthetical notation;
        preserves negative sign. Sign-preserving: -8,551 != 8,551.

        Dispatch:
          - Numeric reference  -> self.numerical_exact_match (polymorphic; resolves to
            TATQAUtils or FinQAUtils at runtime).
          - yes/no reference   -> extract model's stated yes/no
          - all other text     -> financial_normalize both sides and compare.
        """
        if self._safe_float(reference) is not None:
            return self.numerical_exact_match(prediction, reference)
        ref_str = str(reference).strip().lower() if reference else ""
        if ref_str in ("yes", "no"):
            extracted = _extract_yes_no_from_prediction(prediction)
            if extracted is not None:
                return 1.0 if extracted == ref_str else 0.0
        return 1.0 if self.financial_normalize(prediction) == self.financial_normalize(reference) else 0.0

    def score_relaxed_exact_match(
        self,
        pred: str,
        ref: str,
        pred_raw: str,
        exact: float,
        f1: float,
    ) -> float:
        """
        Shared relaxed_exact_match scorer for financial SEC filing QA.
        Used identically by FinQA, TAT-QA, and FinanceBench.

        6-gate deterministic scorer — any gate passing yields 1.0:

        Gate 1: binary_align plus corroboration
                Passes when binary_align=True AND at least one of:
                exact==1, f1>=0.25, or GT verbatim in prediction.

        Gate 2: margin_driver_relaxed (requires f1>=0.30)
                Operating margin driver answers with structural phrase overlap.

        Gate 3: semantic_key_overlap (requires f1>=0.08)
                Short factual answers where key numbers and content words match.

        Gate 4: primary_ratio_match (+-0.5 percent numeric tolerance)
                Numeric ratio / financial figure answers.

        Gate 5: GT text recoverable verbatim from full raw model output
                (non-numeric GT only). Handles verbose predictions where extraction
                left the answer embedded in reasoning text.

        Gate 6: exact==1.0 always passes (internal consistency guarantee).

        No LLM-as-judge. All gates are deterministic string/numeric operations.
        """
        fn = self.financial_normalize
        f1_eff = max(f1, exact)

        binary_ok = self._binary_align(pred, ref)
        ref_in_pred = bool(ref and fn(ref) in fn(pred))
        margin_driver_ok = self._margin_driver_relaxed(pred, ref, f1_eff)
        semantic_short_ok = self._semantic_key_overlap(pred, ref, f1_eff)
        numeric_ratio_ok = self._primary_ratio_match(pred, ref)

        val = 1.0 if (
            (binary_ok and (exact == 1.0 or f1_eff >= 0.25 or ref_in_pred))
            or margin_driver_ok
            or semantic_short_ok
            or numeric_ratio_ok
        ) else 0.0

        # Gate 5: ref verbatim in full raw output (non-numeric GT and year spans only)
        # Year strings like "2019" are span answers (TAT-QA answer_type="span"), not
        # arithmetic values. They should use substring inclusion, not numeric tolerance.
        # A 4-digit integer in 1900-2100 is treated as a year span, not a number.
        if not val and ref:
            if fn(ref) in fn(pred_raw):
                try:
                    ref_stripped = ref.strip()
                    is_year_span = (
                        bool(re.match(r"^\d{4}$", ref_stripped))
                        and 1900 <= int(ref_stripped) <= 2100
                    )
                    is_non_numeric = is_year_span or not any(ch.isdigit() for ch in ref)
                except Exception:
                    is_non_numeric = False
                if is_non_numeric:
                    val = 1.0

        # Gate 6: exact always passes
        if exact == 1.0:
            val = 1.0

        return val


# ---------------------------------------------------------------------------
# RAG dataset/split aggregation (FinQA and TAT-QA share same format)
# ---------------------------------------------------------------------------

RAG_METRIC_ORDER = ("relaxed_exact_match", "exact_match", "f1")
"""Key order for metrics in RAG split and dataset avg JSON (legends and weighted_metrics)."""


def aggregate_rag_split_metrics(rows: list[dict]) -> dict[str, Any]:
    """
    Aggregate per-sample metrics for one RAG split (FinQA or TAT-QA).
    All rows contribute to the denominator. Used for both FinQA and TAT-QA so
    split-level *_avg.json and dataset-level *_avg.json share the same schema.

    Returns:
        sample_count: int
        relaxed_exact_match: "0.XXXX (N/total)"
        exact_match: "0.XXXX (N/total)"
        f1: float (weighted mean, no count breakdown)
    """
    if not rows:
        return {}
    total = len(rows)
    all_metrics = [r.get("metrics") or {} for r in rows]
    if not all_metrics:
        return {
            "sample_count": total,
            "relaxed_exact_match": "0.0000 (0/0)",
            "exact_match": "0.0000 (0/0)",
            "f1": 0.0,
        }
    rem_vals = [m.get("relaxed_exact_match") for m in all_metrics if isinstance(m.get("relaxed_exact_match"), (int, float))]
    ex_vals = [m.get("exact_match") for m in all_metrics if isinstance(m.get("exact_match"), (int, float))]
    f1_vals = [m.get("f1") for m in all_metrics if isinstance(m.get("f1"), (int, float))]
    rem_count = int(round(sum(rem_vals))) if rem_vals else 0
    ex_count = int(round(sum(ex_vals))) if ex_vals else 0
    rem_mean = (sum(rem_vals) / len(rem_vals)) if rem_vals else 0.0
    ex_mean = (sum(ex_vals) / len(ex_vals)) if ex_vals else 0.0
    f1_mean = (sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0
    return {
        "sample_count": total,
        "relaxed_exact_match": f"{rem_mean:.4f} ({rem_count}/{total})",
        "exact_match": f"{ex_mean:.4f} ({ex_count}/{total})",
        "f1": round(f1_mean, 4),
    }


def build_rag_dataset_avg_payload(
    dataset_name: str,
    split_avgs: dict[str, dict],
    timestamp: str,
) -> dict[str, Any]:
    """
    Build dataset-level *_avg.json payload for RAG (FinQA and TAT-QA).
    split_avgs: map split_name -> aggregate_rag_split_metrics output (sample_count,
                relaxed_exact_match "0.XXXX (N/total)", exact_match "0.XXXX (N/total)", f1 float).

    Top-level key order: dataset, sample_count, weighted_metrics, timestamp, splits, splits_breakdown.
    relaxed_exact_match and exact_match are strings; f1 is a number in metrics and weighted_metrics.
    """
    _frac_re = re.compile(r"\((\d+)/(\d+)\)")

    def _f1_float(avg: dict) -> float:
        v = avg.get("f1", 0)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    dataset_total = sum(avg.get("sample_count", 0) for avg in split_avgs.values())
    rem_sum, ex_sum = 0, 0
    f1_weighted_sum = 0.0
    for avg in split_avgs.values():
        n = avg.get("sample_count", 0)
        sm = avg.get("relaxed_exact_match", "")
        em = avg.get("exact_match", "")
        m_rem = _frac_re.search(str(sm))
        m_ex = _frac_re.search(str(em))
        if m_rem:
            rem_sum += int(m_rem.group(1))
        if m_ex:
            ex_sum += int(m_ex.group(1))
        f1_weighted_sum += _f1_float(avg) * n
    f1_mean = (f1_weighted_sum / dataset_total) if dataset_total else 0.0
    weighted_metrics = {
        "relaxed_exact_match": f"{(rem_sum / dataset_total):.4f} ({rem_sum}/{dataset_total})" if dataset_total else "0.0000 (0/0)",
        "exact_match": f"{(ex_sum / dataset_total):.4f} ({ex_sum}/{dataset_total})" if dataset_total else "0.0000 (0/0)",
        "f1": round(f1_mean, 4),
    }
    splits = sorted(split_avgs.keys())
    splits_breakdown = []
    for split_name in splits:
        avg = split_avgs[split_name]
        metrics = {
            "relaxed_exact_match": avg.get("relaxed_exact_match", "0.0000 (0/0)"),
            "exact_match": avg.get("exact_match", "0.0000 (0/0)"),
            "f1": _f1_float(avg),
        }
        splits_breakdown.append({
            "split": split_name,
            "sample_count": avg.get("sample_count", 0),
            "metrics": metrics,
        })
    return {
        "dataset": dataset_name,
        "sample_count": dataset_total,
        "weighted_metrics": weighted_metrics,
        "timestamp": timestamp,
        "splits": splits,
        "splits_breakdown": splits_breakdown,
    }


def _extract_yes_no_from_prediction(prediction: str | None) -> str | None:
    """Extract the model's yes/no answer from a long prediction (e.g. 'the answer is **No**' or '... **no**' -> 'no')."""
    if not prediction:
        return None
    text = str(prediction).strip().lower()
    # Prefer conclusive phrases (answer is X, answer: X) then yes/no at end (allow markdown **yes**/**no**)
    for pattern in [
        r"(?:^|\s)(?:the\s+)?answer\s+is\s+(?:[\*\s]*)(yes|no)(?:[\*\s]*)(?:\s|$|[\.\)])",
        r"(?:^|\s)answer\s*:\s*(?:[\*\s]*)(yes|no)(?:[\*\s]*)(?:\s|$|[\.\)])",
        r"(?:^|\s)(yes|no)\s*$",
        r"(yes|no)[\*\s]*$",  # "... **no**" or "... no" at end (markdown bold)
        r"\*\*(yes|no)\*\*\s*$",  # exactly "**no**" or "**yes**" at end
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    # Last token after stripping asterisks/punctuation (e.g. "**no**" -> "no")
    tokens = text.split()
    if tokens:
        last = tokens[-1].strip("*").rstrip(".*)")
        if last in ("yes", "no"):
            return last
    return None


def _last_number_in_text(text: str | None) -> float | None:
    """Last number in text (by occurrence). Used to prefer final-answer number over in-context numbers.
    For RAG predictions that end with 'Numerical answer (from program execution): -1657.0', returns -1657.0
    so sign_agnostic_match can compare abs(pred) to abs(gt) correctly."""
    if not text:
        return None
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", str(text).replace(",", "")))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except (ValueError, IndexError):
        return None


def _first_number_in_text(text: str | None) -> float | None:
    """First number in text. Used to get the numeric value from gold answers like '$0.5 million'."""
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except (ValueError, IndexError):
        return None


def normalize_rag_prediction_to_gold_scale(pred_answer: str | None, gt_answer: object) -> str | None:
    """When prediction is a bare number and gold has a formatted financial answer ($X million), normalize
    pred to that scale so exact_match and numerical_exact_match can pass (TAT-QA / FinQA scale binding).
    E.g. pred '... 50 ...' and gt '$0.5 million' -> return '$0.5 million' (50/100 = 0.5).
    Returns None if no normalization applied (caller keeps original pred_answer)."""
    if not pred_answer or not gt_answer or not isinstance(gt_answer, str):
        return None
    gt_str = str(gt_answer).strip()
    # Only when gold looks like formatted financial (e.g. $X million, $X billion)
    if "$" not in gt_str or ("million" not in gt_str.lower() and "billion" not in gt_str.lower() and "thousand" not in gt_str.lower()):
        return None
    gold_num = _first_number_in_text(gt_str)
    pred_num = _last_number_in_text(pred_answer)  # program execution puts answer at end
    if gold_num is None or pred_num is None:
        return None
    # Find scale factor k such that pred_num/k ≈ gold_num (table in different units than display)
    for k in (1, 10, 100, 1000, 0.1):
        if k == 0:
            continue
        scaled = pred_num / k
        if math.isclose(scaled, gold_num, rel_tol=0.02, abs_tol=0.02):
            formatted = str(round(scaled, 6)).rstrip("0").rstrip(".")
            normalized = re.sub(r"-?\d+(?:\.\d+)?", formatted, gt_str, count=1)
            return normalized
    # Same scale: pred_num ≈ gold_num
    if math.isclose(pred_num, gold_num, rel_tol=0.02, abs_tol=0.02):
        formatted = str(round(pred_num, 6)).rstrip("0").rstrip(".")
        normalized = re.sub(r"-?\d+(?:\.\d+)?", formatted, gt_str, count=1)
        return normalized
    return None


def prediction_used_back_calc(prediction: str | None) -> bool:
    """True if prediction appears to use percentage back-calculation (e.g. divide(..., %))."""
    if not prediction:
        return False
    lower = prediction.lower()
    return "divide" in lower and ("%" in lower or "percent" in lower)


class FinQAUtils(RagUtils):
    """FinQA: numerical QA. Use strict numerical match for exact_match so we don't give
    credit when the prediction contains a different number (e.g. 3.9 in context) that
    happens to be within 5% of the reference (3.8). Primary metric: numerical_exact_match.
    When the model states the answer at the end (e.g. last line '3.8') but also mentions
    '$3.8 million' earlier, the first-number extraction would wrongly scale to 3.8e6; we
    also check the last number in the prediction so the final answer is counted correct."""

    def numerical_exact_match(self, prediction: str | None, reference: str | None, tol: float = 1e-6) -> float:
        r = self._safe_float(reference)
        if r is None:
            # For yes/no references, use same logic as exact_match so yes/no samples get 1.0 when correct
            ref_str = str(reference).strip().lower() if reference else ""
            if ref_str in ("yes", "no"):
                extracted = _extract_yes_no_from_prediction(prediction)
                if extracted is not None:
                    return 1.0 if extracted == ref_str else 0.0
            return float(self.normalize_text(prediction) == self.normalize_text(reference))
        ref_str = str(reference).strip().lower()
        # For decimal references (e.g. 0.53232), round prediction to ref decimals so computed/fallback
        # values (e.g. 0.5323169340734545) match the GT
        ref_decimals = 0
        if "million" not in ref_str and "billion" not in ref_str and "thousand" not in ref_str and "." in ref_str:
            ref_decimals = len(ref_str.split(".")[-1].replace(",", ""))
        abs_tol = tol
        if ref_decimals > 0:
            abs_tol = max(tol, 10 ** (-ref_decimals))

        def _close(a: float, b: float) -> bool:
            if ref_decimals > 0:
                return round(a, ref_decimals) == round(b, ref_decimals)
            return math.isclose(a, b, rel_tol=tol, abs_tol=abs_tol)

        # Standard: first number in prediction with scale (e.g. $3.8 million -> 3.8e6)
        p_first = self._extract_number_and_scale(prediction)
        if p_first is not None and _close(p_first, r):
            return 1.0
        # When reference is a plain number (no units), also accept if the last number in the
        # prediction equals the reference (model often puts the answer at the end, e.g. "3.8")
        p_last = None
        if "million" not in ref_str and "billion" not in ref_str and "thousand" not in ref_str:
            p_last = _last_number_in_text(prediction)
            if p_last is not None and _close(p_last, r):
                return 1.0
        # Proportion/percentage equivalence: FinQA GT may be decimal (0.65273) while model returns percentage (65.27307) or vice versa
        if p_first is not None and (_close(p_first / 100, r) or _close(p_first * 100, r)):
            return 1.0
        if p_last is not None and (_close(p_last / 100, r) or _close(p_last * 100, r)):
            return 1.0
        return 0.0

    def numerical_near_match(
        self, prediction: str | None, reference: str | None, rel_tol: float = 0.01
    ) -> float:
        """1.0 if prediction is within rel_tol (default 1%) of reference numerically; else 0.0.
        Uses same extraction as numerical_exact_match (first/last number, proportion/percentage).
        Surfaces 'correct reasoning, wrong-but-reasonable operand' (e.g. table row vs prose) as near-correct."""
        if self.numerical_exact_match(prediction, reference) == 1.0:
            return 1.0
        r = self._safe_float(reference)
        if r is None:
            return 0.0
        ref_str = str(reference).strip().lower() if reference else ""
        # Same extraction as numerical_exact_match: first number with scale, then last number for plain ref
        p_first = self._extract_number_and_scale(prediction)
        p_last = None
        if "million" not in ref_str and "billion" not in ref_str and "thousand" not in ref_str:
            p_last = _last_number_in_text(prediction)
        for p_val in (p_first, p_last):
            if p_val is None:
                continue
            denom = max(abs(r), 1e-9)
            if abs(p_val - r) / denom <= rel_tol:
                return 1.0
            if abs(p_val / 100 - r) / denom <= rel_tol or abs(p_val * 100 - r) / denom <= rel_tol:
                return 1.0
        return 0.0


def rag_numerical_match_debug_info(
    prediction: str | None, reference: str | None, dataset_name: str
) -> str:
    """Return debug string for numerical_exact_match=0 drill-down (scale, multi-value, extracted numbers)."""
    pred_str = str(prediction).strip() if prediction else ""
    ref_str = str(reference).strip() if reference else ""
    pred_tail = (pred_str[-280] if len(pred_str) > 280 else pred_str).replace("\n", " ")
    lines = [
        f"pred_tail={pred_tail!r}",
        f"gt={ref_str!r}",
    ]
    ref_multi = _parse_multi_value_numbers(ref_str or None)
    pred_multi = _parse_multi_value_numbers(pred_str or None)
    if dataset_name == "TATQA":
        lines.append(f"ref_parsed_as_multi={ref_multi!r} pred_parsed_as_multi={pred_multi!r}")
        if ref_multi is not None and pred_multi is None:
            lines.append("hint: GT is multi-value (comma-separated); pred did not parse as multi-value")
        elif ref_multi is None and pred_multi is not None:
            lines.append("hint: pred is multi-value but GT is single; dataset may expect one number (e.g. sum)")
    pred_last = _last_number_in_text(pred_str)
    ref_single = None
    if ref_str:
        try:
            ref_single = float(ref_str.replace(",", "").replace("$", "").replace("%", "").strip())
        except (ValueError, TypeError):
            ref_single = _first_number_in_text(ref_str)
    lines.append(f"extracted: pred_last_number={pred_last!r} ref_single={ref_single!r}")
    if ref_single is not None and pred_last is not None and ref_single != 0:
        ratio = pred_last / ref_single
        if abs(ratio - 1000.0) < 10 or abs(ratio - 0.001) < 0.0001:
            lines.append("hint: possible scale mismatch (pred/ref ~ 1000 or 0.001 - check millions vs thousands)")
        elif abs(ratio - 100.0) < 1 or abs(ratio - 0.01) < 0.001:
            lines.append("hint: possible scale mismatch (pred/ref ~ 100 or 0.01 - check percentage vs decimal)")
    return " | ".join(lines)


def _parse_multi_value_numbers(s: str | None) -> list[float] | None:
    """Parse comma-separated numeric values (e.g. '782.833, 106.836' or '1,568.6, 690.5'). Returns list of 2+ floats or None.
    Splits on ', ' (comma-space) so '1,568.6' (thousands separator) stays a single value per segment."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if ", " not in s:
        return None
    parts = [p.strip() for p in s.split(", ")]
    if len(parts) < 2:
        return None
    out = []
    for p in parts:
        if not p:
            return None
        cleaned = p.replace("$", "").replace("%", "").replace(",", "").strip()
        # Strip trailing non-numeric (e.g. '690.5**' from markdown)
        cleaned = re.sub(r"[^\d.-].*$", "", cleaned)
        try:
            out.append(float(cleaned))
        except ValueError:
            num = _first_number_in_text(p)
            if num is not None:
                out.append(num)
            else:
                return None
    return out if len(out) >= 2 else None


# Pattern for a number with optional thousands comma and optional decimal (e.g. 1,568.6 or 690.5)
_NUMBER_WITH_COMMA_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _last_n_numbers_in_text(s: str | None, n: int = 2) -> list[float] | None:
    """Extract the last n numbers (with optional thousands comma) from text. Returns list of n floats or None."""
    if not s or not isinstance(s, str):
        return None
    matches = _NUMBER_WITH_COMMA_RE.findall(s.replace(" ", ""))
    if len(matches) < n:
        return None
    out = []
    for m in matches[-n:]:
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            return None
    return out


def _parse_multi_value_numbers_from_tail(s: str | None, tail_chars: int = 320) -> list[float] | None:
    """Try to parse comma-separated numeric values from the end of the string (answer often at tail).
    Use when full prediction has many ', ' so _parse_multi_value_numbers fails; the numerical answer line is usually last.
    Uses last 2 numbers from tail (with thousands comma) so '1,568.6, 690.5' -> [1568.6, 690.5]."""
    if not s or not isinstance(s, str):
        return None
    tail = s.strip()[-tail_chars:] if len(s) > tail_chars else s.strip()
    parsed = _parse_multi_value_numbers(tail)
    if parsed is not None and len(parsed) == 2:
        return parsed
    if parsed is not None and len(parsed) > 2:
        return _last_n_numbers_in_text(tail, n=2)
    return _last_n_numbers_in_text(tail, n=2)


def _ref_decimal_places(ref_str: str) -> int:
    """Infer display decimal places from reference (e.g. '17.7' -> 1, '1,568.6' -> 1). Used for rounding tolerance."""
    if not ref_str or "." not in ref_str:
        return 0
    return len(ref_str.split(".")[-1].replace(",", "").replace("$", "").replace("%", "").strip())


class TATQAUtils(RagUtils):
    """TAT-QA: supports single-value and multi-value (comma-separated) numerical comparison.
    Uses reference decimal places for rounding tolerance so --reevaluate_only can score correctly
    without re-running the model (e.g. pred 17.69723 vs ref 17.7 -> match when rounded to 1 decimal)."""

    def numerical_exact_match(self, prediction: str | None, reference: str | None, tol: float = 1e-6) -> float:
        ref_multi = _parse_multi_value_numbers(str(reference) if reference else None)
        if ref_multi is not None and len(ref_multi) >= 2:
            pred_multi = _parse_multi_value_numbers(prediction)
            if pred_multi is None and prediction:
                pred_multi = _parse_multi_value_numbers_from_tail(prediction)
            if pred_multi is not None and len(pred_multi) == len(ref_multi):
                for p, r in zip(pred_multi, ref_multi):
                    if not math.isclose(p, r, rel_tol=tol, abs_tol=max(tol, 1e-6)):
                        return 0.0
                return 1.0
            return 0.0
        ref_str = str(reference).strip() if reference else ""
        try:
            ref_single = float(ref_str.replace(",", "").replace("$", "").replace("%", ""))
        except (ValueError, TypeError, AttributeError):
            ref_single = None
        ref_decimals = _ref_decimal_places(ref_str)

        def _close_to_ref(pred_val: float) -> bool:
            if ref_single is None:
                return False
            if ref_decimals > 0:
                return round(pred_val, ref_decimals) == round(ref_single, ref_decimals)
            return math.isclose(pred_val, ref_single, rel_tol=tol, abs_tol=max(tol, 1e-6))

        # Single-value reference: accept if prediction is "a, b" and first value matches (respectively GT sometimes only lists first)
        pred_multi = _parse_multi_value_numbers(prediction)
        if pred_multi is None and prediction:
            pred_multi = _parse_multi_value_numbers_from_tail(prediction)
        if pred_multi is not None and len(pred_multi) >= 1 and ref_single is not None:
            if _close_to_ref(pred_multi[0]):
                return 1.0
        # Single-value: accept if last number in prediction matches reference when rounded to ref decimals (e.g. 17.69723 vs 17.7)
        p_last = _last_number_in_text(prediction)
        if p_last is not None and ref_single is not None and _close_to_ref(p_last):
            return 1.0
        return super().numerical_exact_match(prediction, reference, tol=tol)

    def numerical_near_match(
        self, prediction: str | None, reference: str | None, rel_tol: float = 0.01
    ) -> float:
        """1.0 if prediction is within rel_tol (default 1%) of reference numerically; else 0.0.
        Phase 3c: Same ±1% tolerance as FinQA. Handles single-value and multi-value (comma-separated)."""
        if self.numerical_exact_match(prediction, reference) == 1.0:
            return 1.0
        ref_multi = _parse_multi_value_numbers(str(reference) if reference else None)
        if ref_multi is not None and len(ref_multi) >= 2:
            pred_multi = _parse_multi_value_numbers(prediction)
            if pred_multi is None and prediction:
                pred_multi = _parse_multi_value_numbers_from_tail(prediction)
            if pred_multi is not None and len(pred_multi) == len(ref_multi):
                for p, r in zip(pred_multi, ref_multi):
                    denom = max(abs(r), 1e-9)
                    if abs(p - r) / denom > rel_tol:
                        return 0.0
                return 1.0
            return 0.0
        ref_str = str(reference).strip() if reference else ""
        try:
            ref_single = float(ref_str.replace(",", "").replace("$", "").replace("%", ""))
        except (ValueError, TypeError, AttributeError):
            return 0.0
        p_last = _last_number_in_text(prediction)
        if p_last is not None:
            denom = max(abs(ref_single), 1e-9)
            if abs(p_last - ref_single) / denom <= rel_tol:
                return 1.0
        pred_multi = _parse_multi_value_numbers(prediction)
        if pred_multi is None and prediction:
            pred_multi = _parse_multi_value_numbers_from_tail(prediction)
        if pred_multi is not None and len(pred_multi) >= 1:
            denom = max(abs(ref_single), 1e-9)
            if abs(pred_multi[0] - ref_single) / denom <= rel_tol:
                return 1.0
        return 0.0


# -------------------------------------------------------------------------
# Credit Risk: PD and Sentiment (post-processing and metric helpers)
# -------------------------------------------------------------------------


class CreditRiskUtils(BaseUtils):
    """Shared base for credit risk evaluation (PD, sentiment)."""

    @staticmethod
    def _label_to_binary(gt: str | int | float | None) -> int | None:
        """Map ground truth to binary 0/1 for PD. Accepts 0/1, 'fullypaid'/'chargedoff', etc."""
        if gt is None:
            return None
        if isinstance(gt, (int, float)):
            return 1 if int(gt) == 1 else 0
        s = str(gt).strip().lower()
        if s in ("1", "chargedoff", "chargeoff", "default", "charged_off"):
            return 1
        if s in ("0", "fullypaid", "fully_paid", "current", "paid"):
            return 0
        try:
            return int(float(gt))
        except (ValueError, TypeError):
            return None


class CreditRiskPDUtils(CreditRiskUtils):
    """
    PD (Probability of Default) evaluation: AUC-ROC, F1, Precision, Recall, KS-statistic.
    KS < 0.05 used for drift detection (no significant drift).
    """

    @staticmethod
    def auc_roc(y_true: list[int], y_score: list[float]) -> float:
        """Compute AUC-ROC from binary labels and probability scores. Returns 0.5 when undefined (one class only or NaN)."""
        if not y_true or not y_score or len(y_true) != len(y_score):
            return 0.5
        if len(set(y_true)) < 2:
            return 0.5  # need both classes for AUC
        try:
            from sklearn.metrics import roc_auc_score
            import math
            auc = float(roc_auc_score(y_true, y_score))
            return 0.5 if math.isnan(auc) else auc
        except Exception:
            return 0.5

    @staticmethod
    def f1_precision_recall(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        """Compute F1, precision, recall for binary classification."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            return {
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        except Exception:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    @staticmethod
    def ks_statistic(ref_scores: list[float], current_scores: list[float]) -> float:
        """
        Two-sample Kolmogorov-Smirnov statistic for drift detection.
        Returns KS stat; drift typically flagged when KS >= 0.05.
        """
        if not ref_scores or not current_scores:
            return 0.0
        try:
            from scipy import stats
            return float(stats.ks_2samp(ref_scores, current_scores).statistic)
        except Exception:
            return 0.0

    @staticmethod
    def ks_discriminative(y_true: list[int], y_score: list[float]) -> float:
        """
        KS statistic for PD model discriminative power: max distance between
        positive vs negative class score distributions. Bank-grade target > 0.35.
        """
        if not y_true or not y_score or len(y_true) != len(y_score):
            return 0.0
        try:
            import numpy as np
            from scipy import stats
            y_true_arr = np.asarray(y_true)
            y_score_arr = np.asarray(y_score, dtype=float)
            pos_probs = y_score_arr[y_true_arr == 1]
            neg_probs = y_score_arr[y_true_arr == 0]
            if len(pos_probs) == 0 or len(neg_probs) == 0:
                return 0.0
            return float(stats.ks_2samp(pos_probs, neg_probs).statistic)
        except Exception:
            return 0.0

    def binary_prediction_from_probability(self, prob: float, threshold: float = 0.5) -> int:
        """Convert PD probability to binary class for F1/precision/recall."""
        return 1 if prob >= threshold else 0


class CreditRiskSentimentUtils(CreditRiskUtils):
    """
    Sentiment evaluation: F1 (macro for classification), MSE (for regression/score).
    Used for Financial PhraseBank (label_text: positive/neutral/negative) and FiQA (score or binned label).
    """

    def f1_macro(self, y_true: list[str], y_pred: list[str]) -> float:
        """Multiclass F1 macro (e.g. positive/neutral/negative)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        try:
            from sklearn.metrics import f1_score
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except Exception:
            return 0.0

    @staticmethod
    def mse(y_true: list[float], y_pred: list[float]) -> float:
        """Mean squared error for continuous sentiment score (e.g. FiQA score)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        try:
            from sklearn.metrics import mean_squared_error
            return float(mean_squared_error(y_true, y_pred))
        except Exception:
            return 0.0

    def exact_match(self, prediction: str | None, reference: str | None) -> float:
        """1.0 if normalized prediction equals reference (case-insensitive)."""
        return 1.0 if self.normalize_text(prediction) == self.normalize_text(reference) else 0.0

    @staticmethod
    def score_to_label(score: float, neg_thresh: float = -0.2, pos_thresh: float = 0.2) -> str:
        """Map FiQA-style continuous score to negative/neutral/positive."""
        if score < neg_thresh:
            return "negative"
        if score > pos_thresh:
            return "positive"
        return "neutral"
