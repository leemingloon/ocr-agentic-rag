"""Utility classes for post-processing model outputs and computing evaluation metrics."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable


class BaseUtils:
    """Base utility helpers shared across evaluation categories."""

    @staticmethod
    def normalize_text(text: str | None) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().lower().split())


class OCRUtils(BaseUtils):
    """Base OCR metric helpers."""


class SROIEUtils(OCRUtils):
    """SROIE-specific OCR metrics."""


class FUNSDUtils(OCRUtils):
    """FUNSD-specific OCR metrics."""


class VisionUtils(BaseUtils):
    """Shared utilities for vision-language QA benchmarks."""

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None:
            return None
        cleaned = str(value).strip().replace(",", "")
        cleaned = cleaned.replace("$", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def exact_match(self, prediction: str | None, reference: str | None) -> float:
        return float(self.normalize_text(prediction) == self.normalize_text(reference))

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

    def accuracy(self, prediction: str | None, reference: str | None) -> float:
        return self.exact_match(prediction, reference)


class DocVQAUtils(VisionUtils):
    pass


class InfographicsVQAUtils(VisionUtils):
    pass


class OmniDocBenchUtils(VisionUtils):
    pass


class ChartQAUtils(VisionUtils):
    pass


class MMMUUtils(VisionUtils):
    pass


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

    def exact_match(self, prediction: str | None, reference: str | None) -> float:
        return float(self.normalize_text(prediction) == self.normalize_text(reference))

    def program_accuracy(self, prediction: str | None, reference: str | None) -> float:
        # Placeholder until executable programs are available in GT schema.
        return self.exact_match(prediction, reference)


class FinQAUtils(RagUtils):
    pass


class TATQAUtils(RagUtils):
    pass
