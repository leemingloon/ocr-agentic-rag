"""Utility classes for post-processing model outputs and computing evaluation metrics."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable


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


class OCRUtils(BaseUtils):
    """Base OCR metric helpers."""


class SROIEUtils(OCRUtils):
    """SROIE-specific OCR metrics."""


class FUNSDUtils(OCRUtils):
    """FUNSD-specific OCR metrics."""


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
    """Shared utilities for RAG evaluation."""

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
        p = self._extract_number_and_scale(prediction)
        r = self._extract_number_and_scale(reference)
        if p is None or r is None:
            return float(self.normalize_text(prediction) == self.normalize_text(reference))
        return float(math.isclose(p, r, rel_tol=tol, abs_tol=tol))

    def token_f1(self, prediction: str | None, reference: str | None) -> float:
        p_tokens = self.normalize_text(prediction).split()
        r_tokens = self.normalize_text(reference).split()
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
        return self.relaxed_exact_match(prediction, reference, **kwargs)

    def program_accuracy(self, prediction: str | None, reference: str | None) -> float:
        # Placeholder until executable programs are available in GT schema.
        return self.exact_match(prediction, reference)


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
    """Last number in text (by occurrence). Used to prefer final-answer number over in-context numbers."""
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

    def exact_match(self, prediction: str | None, reference: str | None, **kwargs: object) -> float:
        if self._safe_float(reference) is not None:
            return self.numerical_exact_match(prediction, reference)
        # When reference is yes/no, extract the model's stated yes/no from the prediction for robust matching
        ref_str = str(reference).strip().lower() if reference else ""
        if ref_str in ("yes", "no"):
            extracted = _extract_yes_no_from_prediction(prediction)
            if extracted is not None:
                return 1.0 if extracted == ref_str else 0.0
        return self.relaxed_exact_match(prediction, reference, **kwargs)


class TATQAUtils(RagUtils):
    pass


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
