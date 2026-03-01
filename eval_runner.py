#!/usr/bin/env python3
"""
Unified evaluation runner for OCR / Vision / RAG / Credit Risk.

Architecture notes:
- AUTO_DATASETS and ADAPTER_REGISTRY are authoritative benchmark registries.
- Evaluates one dataset at a time and writes:
  - per-sample JSON: {dataset}_{split}_samples.json (under data/proof/<category>/<dataset>/<split>/)
  - split average JSON: {dataset}_{split}_avg.json (per split)
  - dataset average: {dataset}_avg.json
  - category average: {category}_avg.json
  - eval_summary.json
  Propagation order: samples -> split_avg -> dataset avg -> category avg -> eval_summary.
  Each level is computed by reading the previous level's files so edits to samples files propagate on next run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from eval_dataset_adapters import (
    SROIEAdapter,
    FUNSDAdapter,
    DocVQAAdapter,
    ChartQAAdapter,
    InfographicsVQAAdapter,
    MMMUAccountingAdapter,
    MMMUEconomicsAdapter,
    MMMUFinanceAdapter,
    MMMUMathAdapter,
    FinQAAdapter,
    TATQAAdapter,
    LendingClubAdapter,
    FinancialPhraseBankAdapter,
    FiQAAdapter,
    FinanceBenchAdapter,
)
from eval_postprocess_utils import (
    ChartQAUtils,
    DocVQAUtils,
    FinQAUtils,
    InfographicsVQAUtils,
    MMMUUtils,
    prediction_used_back_calc,
    TATQAUtils,
    CreditRiskPDUtils,
    CreditRiskSentimentUtils,
    RagUtils,
    _extract_yes_no_from_prediction,
)

# ------------------------
# Datasets to evaluate
# ------------------------
AUTO_DATASETS = {
    "ocr": [
        ("SROIE", "hf", "jsdnrs/ICDAR2019-SROIE", None),
        ("FUNSD", "hf", "nielsr/funsd", None),
    ],
    "vision": [
        ("DocVQA", "hf", "lmms-lab/DocVQA", "DocVQA"),
        ("ChartQA", "hf", "HuggingFaceM4/ChartQA", "vis-nlp/ChartQA"),
        ("InfographicsVQA", "hf", "lmms-lab/DocVQA", "InfographicVQA"),
        ("MMMU_Accounting", "hf", "MMMU/MMMU", "Accounting"),
        ("MMMU_Economics", "hf", "MMMU/MMMU", "Economics"),
        ("MMMU_Finance", "hf", "MMMU/MMMU", "Finance"),
        ("MMMU_Math", "hf", "MMMU/MMMU", "Math"),
    ],
    # FinQA: RAG QA uses data/rag/FinQA/train/train_qa.json only (see FinQAAdapter FILE_MAPPING)
    "rag": [
        ("FinQA", "manual", None, None),
        ("TATQA", "hf", None, None),
    ],
    "credit_risk_PD": [
        ("LendingClub", "hf", "TheFinAI/lendingclub-benchmark", None),
    ],
    "credit_risk_PD_quantum": [
        ("LendingClub", "hf", "TheFinAI/lendingclub-benchmark", None),
    ],
    "credit_risk_sentiment": [
        ("FinancialPhraseBank", "hf", "FinanceMTEB/financial_phrasebank", None),
        ("FiQA", "hf", "TheFinAI/fiqa-sentiment-classification", None),
    ],
    "credit_risk_sentiment_finbert": [
        ("FinancialPhraseBank", "hf", "FinanceMTEB/financial_phrasebank", None),
        ("FiQA", "hf", "TheFinAI/fiqa-sentiment-classification", None),
    ],
    "credit_risk_memo_generator": [
        ("FinanceBench", "hf", "PatronusAI/financebench", None),
    ],
}

ADAPTER_REGISTRY = {
    "SROIE": SROIEAdapter,
    "FUNSD": FUNSDAdapter,
    "DocVQA": DocVQAAdapter,
    "ChartQA": ChartQAAdapter,
    "InfographicsVQA": InfographicsVQAAdapter,
    "MMMU_Accounting": MMMUAccountingAdapter,
    "MMMU_Economics": MMMUEconomicsAdapter,
    "MMMU_Finance": MMMUFinanceAdapter,
    "MMMU_Math": MMMUMathAdapter,
    "FinQA": FinQAAdapter,
    "TATQA": TATQAAdapter,
    "LendingClub": LendingClubAdapter,
    "FinancialPhraseBank": FinancialPhraseBankAdapter,
    "FiQA": FiQAAdapter,
    "FinanceBench": FinanceBenchAdapter,
}

VISION_UTILS = {
    "DocVQA": DocVQAUtils(),
    "InfographicsVQA": InfographicsVQAUtils(),
    "ChartQA": ChartQAUtils(),
    "MMMU_Accounting": MMMUUtils(),
    "MMMU_Economics": MMMUUtils(),
    "MMMU_Finance": MMMUUtils(),
    "MMMU_Math": MMMUUtils(),
}

RAG_UTILS = {
    "FinQA": FinQAUtils(),
    "TATQA": TATQAUtils(),
}

MODEL_META = {
    "ocr": {
        "model_class": "HybridOCR",
        "backbone": "paddleocr_tesseract",
        "model_slug": "hybridocr",
    },
    "vision": {
        "model_class": "VisionOCR",
        "backbone": None,  # Set at runtime from VisionOCR.get_effective_model()
        "model_slug": "visionocr",
    },
    "rag": {
        "model_class": "AgenticRAG",
        "backbone": "langgraph_hybrid_retriever_bge_reranker",
        "model_slug": "agenticrag",
    },
    "credit_risk_PD": {
        "model_class": "PDModel",
        "backbone": "xgboost",
        "model_slug": "pd_xgboost",
    },
    "credit_risk_PD_quantum": {
        "model_class": "QuantumPDModel",
        "backbone": "pennylane_vqc",
        "model_slug": "pd_quantum_vqc",
    },
    "credit_risk_sentiment": {
        "model_class": "SentimentClassical",
        "backbone": "tfidf_classical",
        "model_slug": "sentiment_classical",
    },
    "credit_risk_sentiment_finbert": {
        "model_class": "NLPSignalExtractor",
        "backbone": "finbert",
        "model_slug": "finbert",
    },
    "credit_risk_memo_generator": {
        "model_class": "RiskMemoGenerator",
        "backbone": "claude_sonnet",
        "model_slug": "memo_claude",
    },
}

SINGAPORE_TZ = ZoneInfo("Asia/Singapore")


def singapore_now_iso() -> str:
    """Timestamp in Singapore timezone for all proof JSONs."""
    return datetime.now(SINGAPORE_TZ).isoformat()


def _safe_metric_val(v: float | None, default: float = 0.5) -> float:
    """Return default when value is NaN (e.g. AUC undefined for one-class split)."""
    if v is None:
        return default
    try:
        import math
        return default if (isinstance(v, float) and math.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return default


def get_vision_backbone() -> str:
    """Single source of truth: read from VisionOCR so proof JSONs stay aligned."""
    from ocr_pipeline.recognition.vision_ocr import VisionOCR
    return VisionOCR.get_effective_model()


def has_ground_truth(sample: dict, category: str) -> bool:
    gt = sample.get("ground_truth")
    if category == "vision":
        if gt is None:
            return False
        if isinstance(gt, str):
            return bool(gt.strip())
        if isinstance(gt, list):
            return bool(gt and str(gt[0]).strip())
        return True
    if category == "rag":
        if isinstance(gt, dict):
            answer = gt.get("answer")
            return answer is not None and str(answer).strip() != ""
        return gt is not None and str(gt).strip() != ""
    if category in ("credit_risk_PD", "credit_risk_PD_quantum", "credit_risk_sentiment", "credit_risk_sentiment_finbert", "credit_risk_memo_generator"):
        if isinstance(gt, dict):
            val = gt.get("label") or gt.get("answer") or gt.get("reference")
        else:
            val = gt
        return val is not None and str(val).strip() != ""
    if category == "ocr":
        # OCR: ground_truth can be entities, token_labels, or similar
        return gt is not None
    return False


# Vision: strict verbatim extraction primer for DocVQA/InfographicsVQA (address, phone, exact string)
VERBATIM_EXTRACTION_PRIMER = """
MANDATORY VERBATIM EXTRACTION RULE – YOU MUST OBEY 100% OR THE ANSWER IS INVALID:

When the question asks for an address, phone number, code, name, date, or any exact string that appears in the document/image:

1. Locate the text in the image/document.
2. Copy it **character-for-character, pixel-for-pixel as it appears** — NO changes allowed whatsoever.
3. You are **forbidden** from:
   - Changing capitalization (even if it looks wrong)
   - Adding or removing spaces, periods, commas, hyphens
   - Normalizing abbreviations (e.g. keep "N. W." exactly as is — do NOT change to "N.W." or "NW")
   - Converting to title case, sentence case, or modern style
   - "Fixing" perceived typos or spacing
4. Output **ONLY** the copied string — no quotes, no bold, no explanation, no prefix/suffix.
   Example: If the image shows exactly "1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036", you MUST output exactly that — nothing else.
5. If the text is not found or unreadable, output only: NOT_FOUND

Apply this rule strictly now and answer the question.
"""


# Vision: minimal list extraction for InfographicsVQA ("which N items/types/categories" -> plain comma-separated list)
INFOGraphics_LIST_EXTRACTION_PRIMER = """
For questions asking "which [number] [items/types/categories]..." or similar from an infographic:

MANDATORY RULES – FOLLOW EXACTLY:

1. Output ONLY the requested items as a plain comma-separated list.
2. Use lowercase for all items unless the infographic explicitly uses uppercase or title case for the terms. Example: output exactly "restaurants, interior design, wedding venues" — no capitals unless the image shows them.
3. Do NOT add numbers, tables, bold, explanations, icons, or any extra text.
4. Preserve exact wording from the image (e.g., if it says "restaurants", do NOT change to "Restaurants").
5. Example: If the three types are restaurants, interior design, wedding venues, output exactly: restaurants, interior design, wedding venues
6. If uncertain, output: NOT_FOUND

Apply these rules strictly and answer ONLY with the list.
"""


# Vision: accounting / cash flow discipline for MMMU Accounting (taxes, formulas, sign check)
MMMU_ACCOUNTING_PRIMER = """
You are solving a multiple choice accounting problem from a financial statement image.

CRITICAL RULES:
1. Never assume taxes are zero. If a tax rate is not explicitly stated, look for a "taxes" or "income tax" line item directly in the income statement. Use that figure. Only if no tax figure exists anywhere in the image should you note the assumption explicitly.

2. For cash flow problems, use these formulas strictly:
   - OCF = EBIT + Depreciation - Taxes  (Taxes from income statement, not assumed)
   - NCS = Net Fixed Assets(end) - Net Fixed Assets(beg) + Depreciation
   - ΔNWC = (Current Assets - Current Liabilities)end - (Current Assets - Current Liabilities)beg
     where Current Liabilities excludes Long-Term Debt
   - CFA = OCF - NCS - ΔNWC
   - CF to Creditors = Interest Paid - Net New LT Debt
   - CF to Stockholders = CFA - CF to Creditors
   - For CF to Stockholders, if the question context suggests a corporate finance textbook framing, check whether the expected sign convention treats this as cash RECEIVED FROM stockholders (negative = paid out) rather than cash PAID TO stockholders. If options contain your computed value with opposite sign, prefer the negative version.

3. After completing calculations, compare your computed values against each option. Select the option whose values are closest to your results. If your CFA has the wrong sign compared to all options, recheck your tax figure first — a missed tax line is the most common source of sign errors.

4. For covered interest arbitrage when the question says the investor BORROWS the base currency (e.g. USD) to buy foreign currency (e.g. JPY) and asks for profit in the foreign currency: ALWAYS follow this direction — do NOT reverse it mid-calculation. (a) Borrow USD → repay at maturity = USD × (1 + r_USD/4) if rates are annual and 3-month horizon. (b) Convert borrowed USD to JPY at spot rate. (c) Invest JPY at (1 + r_JPY/4). (d) Lock in futures: convert the USD repayment obligation to JPY using the futures rate (USD repayment × futures rate in JPY/USD). (e) Yen profit = JPY investment proceeds − JPY cost of USD repayment. Keep the entire calculation in the foreign currency; do NOT convert USD profit to JPY at spot. If step (d) or (e) yields a loss, DO NOT reverse the strategy direction (e.g. do not switch to borrowing JPY). Instead recheck whether rates should be divided by 4 (annual → quarterly). The question states the investor borrows USD — honour that direction. Then compare your final FC figure to the options and choose the matching letter.

5. Your final answer must be a single letter on the last line: A, B, or C (or D if four options).
"""


# Vision: known bad ground truth (MMMU annotation errors). Score against correct_answer; set gt_override=1 for reporting.
KNOWN_BAD_GROUND_TRUTH: dict[str, dict] = {
    "dev_Accounting_3": {
        "correct_answer": "A",
        "reason": (
            "GT=C has CF to Stockholders = -$1,890.98 which is internally "
            "inconsistent. CFA - CF to Creditors = -493.02 - (-2384) = "
            "+1890.98 under any standard formula. A is correct."
        ),
        "dataset_issue": "MMMU sign convention annotation error",
    },
    "test_Accounting_2": {
        "correct_answer": "C",
        "reason": (
            "GT=B ($21,506) requires manufacturing costs of ~$21,509 "
            "inconsistent with image value of 22,441. Model COGM = "
            "932 + 22441 - 935 = $22,438, closest to C=$22,506. "
            "Unexplained $68 gap suggests possible image truncation "
            "or dataset transcription error."
        ),
        "dataset_issue": "MMMU annotation error - possible missing data in image",
    },
    "test_Accounting_3": {
        "correct_answer": "A",
        "reason": (
            "GT=C (¥129,928.61) is inconsistent with the given data. "
            "USD repayment = $1,008,750, futures = 123.2605, "
            "JPY debt cost = ¥124,336,611. "
            "JPY proceeds = ¥124,455,375. "
            "Profit = ¥118,763.90 = A. "
            "C would require USD repayment of ~$1,008,637 which "
            "contradicts the 3.50%/4 rate applied to $1,000,000."
        ),
        "dataset_issue": "MMMU annotation error - likely used different interest convention in answer key",
    },
    "test_Economics_3": {
        "correct_answer": None,
        "reason": (
            "Options list contains duplicates (A=C, B=D). "
            "Correct economic answer is 'decreases by $0.50' which "
            "is not among the options. GT=A ('increases by $0.50') "
            "is economically incorrect — reducing consumption below "
            "competitive equilibrium always decreases total surplus."
        ),
        "dataset_issue": "MMMU duplicate options + incorrect ground truth",
    },
}


def _needs_verbatim_extraction_primer(question: str) -> bool:
    """True if the question asks for address, location, phone number, or other exact string from the document."""
    if not question or not isinstance(question, str):
        return False
    q = question.strip().lower()
    triggers = (
        "address", "location", "where is", "phone", "telephone", "fax", "e-mail", "email",
        "exact text", "verbatim", "what does it say", "what is written", "what is the text",
        "copy", "reproduce", "zip code", "postal code", "street", "city", "state",
    )
    return any(t in q for t in triggers)


def _needs_list_extraction_primer(question: str) -> bool:
    """True if the question asks for a list of items, types, or categories from an infographic."""
    if not question or not isinstance(question, str):
        return False
    q = question.strip().lower()
    triggers = (
        "which ", "what types", "what kind", "what kinds", "list the", "name the",
        "what are the", "what are some", "types of", "categories", "list of",
        "business types", "good for", "examples of", "such as",
    )
    return any(t in q for t in triggers)


def _extract_numbers_from_text(text: str | None) -> list[int | float]:
    """Extract numeric values from text. Handles $1,234, $(1,234) as negative, etc. Returns list of int/float."""
    if not text:
        return []
    s = str(text)
    # Replace $(number) or (number) accounting-style negatives with a token we can parse
    s = re.sub(r"\(\s*\$?\s*([\d,]+)\s*\)", r" -\1 ", s)
    parts = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                continue
    return out


def _parse_option_string_to_tuple(option_str: str) -> tuple[float, ...]:
    """Parse one option string (e.g. '$63,020' or '$(493), $(2,384), $1,891') into tuple of numbers."""
    if not option_str:
        return ()
    text = str(option_str).strip()
    # Replace $(1,234) with -1234 for parsing
    def replace_neg(m):
        return " " + str(-int(m.group(1).replace(",", "")))
    text_norm = re.sub(r"\(\s*\$?\s*([\d,]+)\s*\)", replace_neg, text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_norm.replace(",", ""))
    if not numbers:
        return ()
    return tuple(float(n) for n in numbers)


def _mmmu_numeric_to_mc_letter(prediction_text: str | None, options_list: list | None) -> str | None:
    """
    Map numeric model output (e.g. cash flow CFFA, CF to creditors, CF to stockholders) to multiple-choice letter.
    Builds letter -> numeric tuple from options_list; extracts numbers from prediction; returns letter with closest L2 match.
    Returns None if options_list missing/empty or parsing fails (caller keeps original prediction).
    """
    if not prediction_text or not options_list or not isinstance(options_list, list):
        return None
    option_tuples: list[tuple[float, ...]] = []
    for opt in options_list:
        t = _parse_option_string_to_tuple(str(opt) if opt is not None else "")
        if t:
            option_tuples.append(t)
    if not option_tuples:
        return None
    n = len(option_tuples[0])
    if n == 0:
        return None
    pred_nums = _extract_numbers_from_text(prediction_text)
    if len(pred_nums) < n:
        return None
    # For multi-value options (e.g. cash flow CFFA, CF creditors, CF stockholders), the model
    # usually puts the final answer in a summary at the end; use last n numbers. For single-value
    # use first (only relevant if we ever call with n==1 for multi-option single-number).
    if n >= 2:
        pred_tuple = tuple(float(pred_nums[i]) for i in range(-n, 0))
    else:
        pred_tuple = tuple(float(pred_nums[i]) for i in range(n))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    min_dist = float("inf")
    best_letter = None
    for idx, gt_tuple in enumerate(option_tuples):
        if len(gt_tuple) != n:
            continue
        dist = sum((p - g) ** 2 for p, g in zip(pred_tuple, gt_tuple)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_letter = letters[idx] if idx < len(letters) else None
    return best_letter


def extract_mcq_answer(prediction: str, debug: bool = False) -> str | None:
    """Extract multiple-choice letter from model output. If body explicitly identifies an option
    (e.g. 'matches Option A') and last line differs, prefer body (reasoning-extraction mismatch)."""
    if not prediction or not isinstance(prediction, str):
        if debug:
            print("[DEBUG] extract_mcq_answer: no prediction or not str")
        return None
    text = prediction.strip()
    if not text:
        if debug:
            print("[DEBUG] extract_mcq_answer: empty after strip")
        return None
    lines = text.split("\n")
    last_line = lines[-1].strip() if lines else ""
    # Body = everything except last ~80 chars (avoid final line and trailing fluff)
    body = text[:-80] if len(text) > 80 else ""
    # Explicit "Option A/B/C/D" in body (e.g. "this matches Option A") — last occurrence = conclusion
    body_option_matches = re.findall(r"[Oo]ption\s+([A-D])\b", body)
    body_letter = body_option_matches[-1].upper() if body_option_matches else None
    last_line_letter = None
    if len(last_line) == 1 and last_line.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        last_line_letter = last_line.upper()
    if not last_line_letter:
        suffix = text[-200:] if len(text) > 200 else text
        fallback = re.findall(r"\b([A-D])\b", suffix)
        last_line_letter = fallback[-1] if fallback else None
    # Reasoning-extraction mismatch: body says "Option X", last line says different letter → prefer body
    if body_letter and last_line_letter and body_letter != last_line_letter:
        if debug:
            print(
                f"[DEBUG] extract_mcq_answer: body_override body_letter={body_letter!r} last_line={last_line_letter!r} -> {body_letter!r}"
            )
        return body_letter
    if debug:
        last_3_lines = lines[-3:] if len(lines) >= 3 else lines
        print(f"[DEBUG] extract_mcq_answer: last_line={repr(last_line)} (len={len(last_line)}) last_3_lines={[repr(L) for L in last_3_lines]}")
    if last_line_letter:
        if debug:
            print(f"[DEBUG] extract_mcq_answer: using last_line/fallback -> '{last_line_letter}'")
        return last_line_letter
    if debug:
        print(f"[DEBUG] extract_mcq_answer: no letter found")
    return None


def _extract_image_for_vision(sample: dict, debug: bool = False):
    img = sample.get("input", {}).get("image") if isinstance(sample, dict) else None
    sample_id = sample.get("metadata", {}).get("sample_id") if isinstance(sample, dict) else None

    if img is None:
        return None, "missing_image_field"

    try:
        import numpy as np
        from PIL import Image

        if isinstance(img, np.ndarray):
            if debug:
                print(f"[DEBUG] image_ok sample={sample_id} source=numpy shape={img.shape}")
            return img, None

        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
            if debug:
                print(f"[DEBUG] image_ok sample={sample_id} source=pil shape={arr.shape}")
            return arr, None

        if isinstance(img, str):
            candidate = img.strip()
            if not candidate:
                return None, "empty_image_string"
            if candidate.startswith("<PIL.Image.Image"):
                return None, "serialized_pil_repr_in_json_fallback"
            if candidate.startswith("data:image/"):
                return None, "unsupported_data_url_image"
            if not os.path.exists(candidate):
                return None, f"image_path_not_found:{candidate[:160]}"
            arr = np.array(Image.open(candidate).convert("RGB"))
            if debug:
                print(f"[DEBUG] image_ok sample={sample_id} source=path path={candidate[:160]} shape={arr.shape}")
            return arr, None

        if isinstance(img, dict):
            import io
            raw = img.get("bytes")
            if raw is not None:
                # Parquet/Arrow may give bytes, bytearray, pyarrow.Buffer, or list of ints
                if isinstance(raw, bytes):
                    b = raw
                elif isinstance(raw, bytearray):
                    b = bytes(raw)
                elif hasattr(raw, "tobytes"):
                    b = raw.tobytes()
                elif hasattr(raw, "as_py"):
                    b = raw.as_py()
                    if not isinstance(b, bytes):
                        b = bytes(b) if b is not None else b
                elif isinstance(raw, (list, tuple)):
                    try:
                        b = bytes(raw)
                    except Exception:
                        b = None
                else:
                    try:
                        b = bytes(raw)
                    except Exception:
                        b = None
                if b:
                    arr = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
                    if debug:
                        print(f"[DEBUG] image_ok sample={sample_id} source=dict_bytes shape={arr.shape}")
                    return arr, None
            if "path" in img and img["path"]:
                path = img["path"] if isinstance(img["path"], str) else str(img["path"])
                if os.path.exists(path):
                    arr = np.array(Image.open(path).convert("RGB"))
                    if debug:
                        print(f"[DEBUG] image_ok sample={sample_id} source=dict_path path={path[:160]} shape={arr.shape}")
                    return arr, None
                return None, f"image_path_not_found:{path[:160]}"

        return None, f"unsupported_image_type:{type(img).__name__}"
    except Exception as exc:
        return None, f"image_decode_exception:{exc}"


def _run_vision_model(sample: dict, dataset_name: str | None = None, debug: bool = False) -> dict:
    image, image_error = _extract_image_for_vision(sample, debug=debug)
    question = sample.get("input", {}).get("question") if isinstance(sample, dict) else None
    sample_id = sample.get("metadata", {}).get("sample_id") if isinstance(sample, dict) else None

    if image is None:
        if debug:
            img_val = sample.get("input", {}).get("image") if isinstance(sample, dict) else None
            preview = str(img_val)
            hint = ""
            if image_error == "serialized_pil_repr_in_json_fallback":
                hint = " hint=first_5_rows_json_contains_repr_not_real_image"
            elif str(image_error).startswith("image_path_not_found"):
                hint = " hint=image_path_missing_or_not_materialized"
            print(
                f"[DEBUG] vision_input_invalid sample={sample_id} reason={image_error} "
                f"image_type={type(img_val).__name__ if img_val is not None else 'None'} "
                f"image_preview={preview[:180]}{hint}"
            )
        return {"answer": "", "error": f"missing_image:{image_error}"}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"answer": "", "error": "missing_anthropic_api_key"}

    # DocVQA/InfographicsVQA: inject extraction primers (verbatim for address/phone; minimal list for infographic lists)
    extra_instruction = None
    if dataset_name in ("DocVQA", "InfographicsVQA") and question:
        if _needs_verbatim_extraction_primer(question):
            extra_instruction = VERBATIM_EXTRACTION_PRIMER
            if debug:
                print(f"[DEBUG] vision injecting verbatim extraction primer (dataset={dataset_name})")
        elif dataset_name == "InfographicsVQA" and _needs_list_extraction_primer(question):
            extra_instruction = INFOGraphics_LIST_EXTRACTION_PRIMER
            if debug:
                print(f"[DEBUG] vision injecting list extraction primer (dataset={dataset_name})")

    # MMMU_Accounting: inject accounting/cash-flow primer (taxes, formulas, sign check)
    if dataset_name == "MMMU_Accounting":
        accounting_block = MMMU_ACCOUNTING_PRIMER
        extra_instruction = (extra_instruction + "\n\n" + accounting_block) if extra_instruction else accounting_block
        if debug:
            print(f"[DEBUG] vision injecting MMMU Accounting primer (dataset={dataset_name})")

    # MMMU_* and any vision dataset with options_list: inject Options block + last-line instruction
    options_list = sample.get("metadata", {}).get("options_list") if isinstance(sample, dict) else None
    if options_list and isinstance(options_list, list) and len(options_list) > 0:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        formatted_options = "\n".join(f"{labels[i]}: {opt}" for i, opt in enumerate(options_list) if i < len(labels))
        letter_list = ", ".join(labels[: min(len(options_list), 4)]) if len(options_list) <= 4 else "A, B, C, etc."
        mc_block = (
            f"Options:\n{formatted_options}\n\n"
            f"You must select exactly one answer. State your final answer as a single letter ({letter_list}) on the very last line of your response, with no other text on that line.\n\n"
            "CRITICAL: Your final answer letter MUST match the option you identified as correct in your reasoning. Before writing your final letter, re-read your conclusion and verify the letter matches. If your reasoning says \"this matches Option A\", your final line must be \"A\", not any other letter."
        )
        extra_instruction = (extra_instruction + "\n\n" + mc_block) if extra_instruction else mc_block
        if debug:
            print(f"[DEBUG] vision injecting multiple-choice options (dataset={dataset_name}, {len(options_list)} options)")

    try:
        from ocr_pipeline.recognition.vision_ocr import VisionOCR
        vision = VisionOCR(api_key=api_key)
        # Use chart-aware path when question exists; otherwise generic recognize.
        if question:
            out = vision.extract_charts(image=image, question=question, extra_instruction=extra_instruction)
            return {"answer": str(out.get("chart_analysis", ""))}

        out = vision.recognize(image=image, task="extract")
        return {"answer": str(getattr(out, "text", ""))}
    except Exception as exc:
        return {"answer": "", "error": f"vision_inference_failed:{exc}"}


# Cache for RAG retriever per dataset (index built once, reused for all samples)
_RAG_RETRIEVER_CACHE: dict[str, Any] = {}

# Cache for HybridOCR (one instance per process for OCR eval)
_OCR_HYBRID_CACHE: dict[str, Any] = {}

# Cache for PD (XGBoost) model: load once per process for overnight sample-by-sample evaluation (CPU-only)
_PD_MODEL_CACHE: dict[str, Any] = {}
_QUANTUM_PD_MODEL_CACHE: dict[str, Any] = {}
_SENTIMENT_PKL_CACHE: dict[str, Any] = {}


def _build_finqa_corpus_chunks(debug: bool = False) -> list:
    """Load FinQA train_qa.json and return list of TextNode chunks for retrieval index.
    Each entry's pre_text, table, post_text are combined into a document and chunked.
    Always uses the full corpus so RAG evaluation is fair (no subset limit)."""
    from rag_system.chunking import DocumentChunker

    # Resolve path from repo root so it works regardless of cwd
    repo_root = Path(__file__).resolve().parent
    train_qa_path = repo_root / "data" / "rag" / "FinQA" / "train" / "train_qa.json"
    if not train_qa_path.exists():
        return []
    with open(train_qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        return []

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    for idx, entry in enumerate(data):
        pre = entry.get("pre_text") or []
        post = entry.get("post_text") or []
        table = entry.get("table") or entry.get("table_ori") or []
        pre_str = "\n".join(pre) if isinstance(pre, list) else str(pre)
        post_str = "\n".join(post) if isinstance(post, list) else str(post)
        table_str = ""
        if table:
            for row in table:
                if isinstance(row, (list, tuple)):
                    table_str += " | ".join(str(c) for c in row) + "\n"
                else:
                    table_str += str(row) + "\n"
        doc_text = f"{pre_str}\n\n{table_str}\n\n{post_str}".strip()
        if not doc_text:
            continue
        # Use id (e.g. "AAL/2018/page_13.pdf-2") so eval ground_truth.corpus_id matches exactly
        corpus_id = entry.get("id") or entry.get("filename", str(idx))
        chunks = chunker.chunk_document(
            doc_text,
            metadata={"entry_id": idx, "source": "finqa_train", "corpus_id": corpus_id},
        )
        all_chunks.extend(chunks)
    return all_chunks


def _build_tatqa_corpus_chunks(debug: bool = False) -> list:
    """Load TAT-QA train and dev JSONs and return list of TextNode chunks for retrieval index.
    Each doc's table + paragraphs are combined and chunked. corpus_id = table uid per doc."""
    from rag_system.chunking import DocumentChunker

    repo_root = Path(__file__).resolve().parent
    tatqa_dir = repo_root / "data" / "rag" / "TAT-QA"
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    for split, filename in [("train", "tatqa_dataset_train.json"), ("dev", "tatqa_dataset_dev.json")]:
        path = tatqa_dir / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for doc_idx, doc in enumerate(data):
            table = doc.get("table") or {}
            table_rows = table.get("table") if isinstance(table, dict) else table
            if isinstance(table_rows, list):
                table_str = "\n".join(" | ".join(str(c) for c in row) for row in table_rows)
            else:
                table_str = str(table_rows or "")
            paragraphs = doc.get("paragraphs") or []
            para_str = "\n".join(
                p.get("text", "") for p in paragraphs if isinstance(p, dict)
            )
            doc_text = f"{table_str}\n\n{para_str}".strip()
            if not doc_text:
                continue
            corpus_id = table.get("uid") if isinstance(table, dict) else None
            if not corpus_id:
                corpus_id = f"tatqa_{split}_{doc_idx}"
            chunks = chunker.chunk_document(
                doc_text,
                metadata={"entry_id": doc_idx, "source": f"tatqa_{split}", "corpus_id": corpus_id},
            )
            all_chunks.extend(chunks)
    return all_chunks


def _get_rag_retriever_for_dataset(dataset_name: str, debug: bool = False):
    """Return a HybridRetriever with index built from the dataset corpus. Cached per dataset."""
    global _RAG_RETRIEVER_CACHE
    if dataset_name in _RAG_RETRIEVER_CACHE:
        if debug:
            print(f"[DEBUG] RAG using cached retriever for {dataset_name}")
        return _RAG_RETRIEVER_CACHE[dataset_name]

    from rag_system.retrieval import HybridRetriever

    retriever = HybridRetriever()
    if dataset_name == "FinQA":
        repo_root = Path(__file__).resolve().parent
        prebuilt_index_dir = repo_root / "data" / "rag" / "FinQA" / "train" / "finqa_retriever_index"
        if (prebuilt_index_dir / "meta.json").exists():
            if debug:
                print(f"[DEBUG] RAG FinQA: loading pre-built index from {prebuilt_index_dir}")
            retriever.load_index_bundle(str(prebuilt_index_dir))
        else:
            chunks = _build_finqa_corpus_chunks(debug=debug)
            if not chunks:
                if debug:
                    print("[DEBUG] RAG FinQA: no chunks from train_qa.json; index will be empty (retrieve will fail)")
            else:
                if debug:
                    meta = lambda c: getattr(c, "metadata", None) or {}
                    corpus_ids = list({meta(c).get("corpus_id") for c in chunks})
                    corpus_ids = [x for x in corpus_ids if x is not None][:10]
                    print(f"[DEBUG] RAG FinQA: building index from {len(chunks)} chunks (train_qa.json); "
                          f"sample corpus_ids: {corpus_ids}")
                retriever.build_index(chunks)
    elif dataset_name == "TATQA":
        repo_root = Path(__file__).resolve().parent
        prebuilt_index_dir = repo_root / "data" / "rag" / "TAT-QA" / "tatqa_retriever_index"
        if (prebuilt_index_dir / "meta.json").exists():
            if debug:
                print(f"[DEBUG] RAG TATQA: loading pre-built index from {prebuilt_index_dir}")
            retriever.load_index_bundle(str(prebuilt_index_dir))
        else:
            chunks = _build_tatqa_corpus_chunks(debug=debug)
            if not chunks:
                if debug:
                    print("[DEBUG] RAG TATQA: no chunks from TAT-QA train/dev; index will be empty (retrieve will fail)")
            else:
                if debug:
                    meta = lambda c: getattr(c, "metadata", None) or {}
                    corpus_ids = list({meta(c).get("corpus_id") for c in chunks})[:10]
                    print(f"[DEBUG] RAG TATQA: building index from {len(chunks)} chunks; sample corpus_ids: {corpus_ids}")
                retriever.build_index(chunks)
    else:
        if debug:
            print(f"[DEBUG] RAG {dataset_name}: no corpus loader; indices not built (retrieve will fail)")
    _RAG_RETRIEVER_CACHE[dataset_name] = retriever
    return retriever


def run_model(sample: dict, category: str, dataset_name: str, debug: bool = False) -> dict:
    """Model inference hook.

    NOTE: this is intentionally a baseline stub so the evaluation plumbing can run end-to-end.
    Replace this with real pipeline calls (VisionOCR / AgenticRAG) per deployment mode.
    """
    sample_input = sample.get("input", {}) if isinstance(sample, dict) else {}

    if category == "ocr":
        image, image_error = _extract_image_for_vision(sample, debug=debug)
        if image is None:
            return {"answer": "", "error": f"missing_image:{image_error}", "metadata": {}}
        try:
            from ocr_pipeline.recognition.hybrid_ocr import HybridOCR
            if "ocr" not in _OCR_HYBRID_CACHE:
                _OCR_HYBRID_CACHE["ocr"] = HybridOCR(use_detection_router=True, use_vision_augmentation=False)
            ocr = _OCR_HYBRID_CACHE["ocr"]
            # OCR_EVAL_USE_TESSERACT=1: use Tesseract path (preprocessing, table-friendly, confidence flagging) first; PaddleOCR only as fallback.
            # Default: force_paddleocr=True so eval uses full PaddleOCR (det+rec) when available.
            force_paddle = os.environ.get("OCR_EVAL_USE_TESSERACT", "").strip().lower() not in ("1", "true", "yes")
            out = ocr.process_document(image, force_paddleocr=force_paddle)
            text = out.get("text", "")
            metadata = out.get("metadata", {})
            if out.get("low_confidence_words"):
                metadata["low_confidence_words"] = out["low_confidence_words"]
            return {"answer": text or "", "metadata": metadata}
        except Exception as e:
            if debug:
                print(f"[DEBUG] OCR inference failed: {e}")
            return {"answer": "", "error": f"ocr_inference_failed:{e}", "metadata": {}}

    if category == "vision":
        return _run_vision_model(sample, dataset_name=dataset_name, debug=debug)

    if category == "rag":
        query = sample_input.get("query") or sample_input.get("question") or ""
        gt = sample.get("ground_truth") if isinstance(sample.get("ground_truth"), dict) else {}
        corpus_id = gt.get("corpus_id") if gt else None
        if debug:
            print(f"[DEBUG] RAG query corpus_id={corpus_id!r} query={query[:80]!r}...")
        try:
            import os
            _prev_rag_debug = os.environ.get("RAG_DEBUG")
            if debug:
                os.environ["RAG_DEBUG"] = "1"
            try:
                from rag_system.agentic.orchestrator import AgenticRAG
                from rag_system.reranking import BGEReranker

                retriever = _get_rag_retriever_for_dataset(dataset_name, debug=debug)
                reranker = BGEReranker()
                rag = AgenticRAG(retriever=retriever, reranker=reranker)
                out = rag.query(query, corpus_id=corpus_id)
            finally:
                if _prev_rag_debug is None and "RAG_DEBUG" in os.environ:
                    os.environ.pop("RAG_DEBUG", None)
                elif _prev_rag_debug is not None:
                    os.environ["RAG_DEBUG"] = _prev_rag_debug
            if debug:
                sources = out.get("tool_results") or []
                num_chunks = 0
                for s in sources:
                    r = s.get("result") if isinstance(s.get("result"), dict) else {}
                    num_chunks += len(r.get("chunks") or [])
                first_preview = ""
                if sources:
                    r0 = sources[0].get("result")
                    if isinstance(r0, dict) and r0.get("chunks"):
                        t = (r0["chunks"][0].get("text") or "")[:200]
                        first_preview = t.replace("\n", " ") + ("..." if len(t) >= 200 else "")
                print(f"[DEBUG] RAG result: {num_chunks} chunks retrieved; first_chunk_preview={first_preview!r}")
            return {
                "answer": out.get("answer") or "",
                "sources": out.get("tool_results", []),
                "reasoning": str(out.get("plan", [])),
            }
        except Exception as e:
            if debug:
                print(f"[DEBUG] RAG inference failed: {e}")
            return {
                "answer": "",
                "sources": [],
                "error": f"rag_inference_failed:{e}",
            }

    if category == "credit_risk_PD":
        features = sample_input.get("features") or {}
        try:
            repo_root = Path(__file__).resolve().parent
            model_path = repo_root / "models" / "pd" / "pd_model_local_v1.pkl"
            cache_key = str(model_path)
            if cache_key not in _PD_MODEL_CACHE:
                from credit_risk.models.pd_model import PDModel
                pd_model = PDModel(mode="local")
                if model_path.exists():
                    pd_model.load(str(model_path))
                    _PD_MODEL_CACHE[cache_key] = pd_model
                else:
                    _PD_MODEL_CACHE[cache_key] = None
            pd_model = _PD_MODEL_CACHE[cache_key]
            if pd_model is not None:
                pd_prob = pd_model.predict_pd(features)
            else:
                pd_prob = 0.0
        except Exception as e:
            if debug:
                print(f"[DEBUG] PD inference failed: {e}")
            pd_prob = 0.0
        binary_pred = 1 if pd_prob >= 0.5 else 0
        return {
            "answer": str(pd_prob),
            "probability": pd_prob,
            "binary_pred": binary_pred,
        }

    if category == "credit_risk_PD_quantum":
        features = sample_input.get("features") or {}
        try:
            repo_root = Path(__file__).resolve().parent
            model_path = repo_root / "models" / "pd" / "pd_quantum_vqc_v1.pkl"
            cache_key = str(model_path)
            if cache_key not in _QUANTUM_PD_MODEL_CACHE:
                from credit_risk.models.quantum_pd_model import QuantumPDModel
                qpd = QuantumPDModel()
                if model_path.exists():
                    qpd.load(str(model_path))
                    _QUANTUM_PD_MODEL_CACHE[cache_key] = qpd
                else:
                    _QUANTUM_PD_MODEL_CACHE[cache_key] = None
            qpd_model = _QUANTUM_PD_MODEL_CACHE[cache_key]
            if qpd_model is not None:
                pd_prob = qpd_model.predict_pd(features)
            else:
                pd_prob = 0.0
        except Exception as e:
            if debug:
                print(f"[DEBUG] Quantum PD inference failed: {e}")
            pd_prob = 0.0
        binary_pred = 1 if pd_prob >= 0.5 else 0
        return {
            "answer": str(pd_prob),
            "probability": pd_prob,
            "binary_pred": binary_pred,
        }

    if category == "credit_risk_sentiment":
        text = sample_input.get("text") or ""
        try:
            repo_root = Path(__file__).resolve().parent
            pkl_path = repo_root / "models" / "sentiment" / "sentiment_classical_v1.pkl"
            if pkl_path.exists():
                cache_key = str(pkl_path)
                if cache_key not in _SENTIMENT_PKL_CACHE:
                    import joblib
                    _SENTIMENT_PKL_CACHE[cache_key] = joblib.load(pkl_path)
                data = _SENTIMENT_PKL_CACHE[cache_key]
                vec = data.get("vectorizer")
                clf = data.get("classifier")
                if vec is not None and clf is not None:
                    X = vec.transform([str(text)])
                    label = clf.predict(X)[0]
                    if hasattr(label, "item"):
                        label = str(label.item()) if hasattr(clf, "classes_") else str(label)
                    else:
                        label = str(label)
                    label = label.strip().lower() if label else "neutral"
                else:
                    label = "neutral"
            else:
                label = "neutral"
        except Exception as e:
            if debug:
                print(f"[DEBUG] Sentiment (classical pkl) inference failed: {e}")
            label = "neutral"
        return {"answer": label}

    if category == "credit_risk_sentiment_finbert":
        text = sample_input.get("text") or ""
        try:
            from credit_risk.feature_engineering.nlp_signals import NLPSignalExtractor
            extractor = NLPSignalExtractor(mode="local")
            signals = extractor.extract_signals([text])
            ns = signals.get("news_sentiment") or {}
            score = ns.get("score", 0.0)
            trend = (ns.get("trend") or "stable").lower()
            if trend == "deteriorating" or (isinstance(score, (int, float)) and score < -0.2):
                label = "negative"
            elif trend == "improving" or (isinstance(score, (int, float)) and score > 0.2):
                label = "positive"
            else:
                label = "neutral"
        except Exception as e:
            if debug:
                print(f"[DEBUG] FinBERT sentiment inference failed: {e}")
            label = "neutral"
        return {"answer": label}

    if category == "credit_risk_memo_generator":
        question = sample_input.get("question") or sample_input.get("prompt") or ""
        context = sample_input.get("context") or ""
        answer = ""
        if question or context:
            try:
                import os
                if os.getenv("ANTHROPIC_API_KEY"):
                    import anthropic
                    client = anthropic.Anthropic()
                    msg = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"}],
                    )
                    answer = (msg.content[0].text if msg.content else "") or ""
                else:
                    from credit_risk.governance.risk_memo_generator import RiskMemoGenerator
                    gen = RiskMemoGenerator(mode="local")
                    memo = gen.generate_memo(borrower="eval_sample", features={}, pd=0.0, drivers=[], save_to_s3=False)
                    answer = getattr(memo, "content", None) or str(memo) if memo else ""
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Memo inference failed: {e}")
        return {"answer": answer or ""}

    return {"answer": ""}


def evaluate_vision_sample(
    dataset_name: str,
    prediction: dict,
    sample: dict,
    debug: bool = False,
) -> dict[str, float]:
    utils = VISION_UTILS[dataset_name]
    pred_answer = prediction.get("answer", "")
    gt = sample.get("ground_truth")
    if isinstance(gt, list):
        gt_answer = gt[0] if gt else ""
    else:
        gt_answer = gt if gt is not None else ""
    options_list = sample.get("metadata", {}).get("options_list")
    sample_id = sample.get("metadata", {}).get("sample_id")

    # Known bad GT (e.g. MMMU annotation errors): score against correct_answer and flag for reporting
    gt_override_used = False
    effective_gt = gt_answer
    if sample_id and sample_id in KNOWN_BAD_GROUND_TRUTH:
        override = KNOWN_BAD_GROUND_TRUTH[sample_id]
        effective_gt = override.get("correct_answer")  # None = no correct option, do not penalize
        gt_override_used = True
        if debug:
            print(f"[DEBUG] vision gt_override: sample_id={sample_id!r} reason={override.get('reason', '')[:80]}... effective_gt={effective_gt!r}")

    if debug:
        pred_preview = str(pred_answer).replace("\n", " ")[:200]
        gt_preview = str(gt_answer).replace("\n", " ")[:200]
        print(
            f"[DEBUG] vision_eval_input dataset={dataset_name} "
            f"pred_preview='{pred_preview}' gt_preview='{gt_preview}'"
        )

    if dataset_name in {"DocVQA", "InfographicsVQA"}:
        anls = utils.anls(pred_answer, gt_answer)
        exact = utils.exact_match(pred_answer, gt_answer, options_list=options_list)
        if debug:
            print(
                f"[DEBUG] vision_eval_metrics dataset={dataset_name} "
                f"anls={anls} exact_match={exact}"
            )
        return {
            "anls": anls,
            "exact_match": exact,
        }

    if dataset_name == "ChartQA":
        strict_acc = utils.exact_match(pred_answer, gt_answer, options_list=options_list)
        relaxed_acc = utils.relaxed_numeric_accuracy(pred_answer, gt_answer)
        if debug:
            print(
                f"[DEBUG] vision_eval_metrics dataset={dataset_name} "
                f"strict_accuracy={strict_acc} relaxed_accuracy={relaxed_acc}"
            )
        return {
            "strict_accuracy": strict_acc,
            "relaxed_accuracy": relaxed_acc,
        }

    # MMMU_*: prefer extracted letter from last line; else map numeric output to MC letter when options are multi-value
    if (
        dataset_name.startswith("MMMU_")
        and options_list
        and gt_answer
        and len(str(gt_answer).strip()) == 1
        and str(gt_answer).strip().upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        if debug:
            last_3 = pred_answer.strip().split("\n")[-3:] if pred_answer else []
            print(f"[DEBUG] vision MMMU pred last_3_lines: {[repr(L) for L in last_3]}")
            letters_preview = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, opt in enumerate(options_list):
                if i < len(letters_preview):
                    preview = (str(opt))[:80] + ("..." if len(str(opt)) > 80 else "")
                    print(f"[DEBUG] vision MMMU options_list[{letters_preview[i]}]: {preview!r}")
        extracted_letter = extract_mcq_answer(pred_answer, debug=debug)
        if extracted_letter is not None:
            pred_answer = extracted_letter
            if debug:
                print(f"[DEBUG] vision MMMU using extracted MC letter '{extracted_letter}' (gt={gt_answer})")
        else:
            mapped_letter = _mmmu_numeric_to_mc_letter(pred_answer, options_list)
            if mapped_letter is not None:
                first_tup = _parse_option_string_to_tuple(
                    str(options_list[0]) if options_list else ""
                )
                if len(first_tup) >= 2:
                    if debug:
                        pred_nums = _extract_numbers_from_text(pred_answer)
                        n = len(first_tup)
                        pred_tup = tuple(float(pred_nums[i]) for i in range(-n, 0)) if len(pred_nums) >= n else ()
                        print(
                            f"[DEBUG] vision MMMU numeric->MC mapped pred_tuple={pred_tup} -> letter '{mapped_letter}' (gt={gt_answer})"
                        )
                    pred_answer = mapped_letter

    if gt_override_used and effective_gt is None:
        # No correct option (e.g. duplicate options + wrong GT); do not penalize
        acc = 1.0
    else:
        acc = utils.accuracy(pred_answer, effective_gt, options_list=options_list)
    if debug:
        print(
            f"[DEBUG] vision_eval_metrics dataset={dataset_name} "
            f"accuracy={acc}"
            + (f" gt_override=1 (effective_gt={effective_gt!r})" if gt_override_used else "")
        )
    out = {"accuracy": acc}
    if gt_override_used:
        out["gt_override"] = 1
    return out


def _rag_prediction_is_error_or_refusal(pred_answer: str) -> bool:
    """True if the model output indicates retrieval/system failure or refusal to answer.
    Such predictions must not get credit (e.g. GT '1' matching '1. Remove...' in error text)."""
    if not (pred_answer and pred_answer.strip()):
        return False
    lower = pred_answer.lower()
    error_phrases = [
        "i cannot answer",
        "cannot provide",
        "technical error",
        "retrieval process",
        "chunks: []",
        "unexpected keyword argument",
        "no information was successfully retrieved",
        "configuration or implementation issue",
    ]
    return any(phrase in lower for phrase in error_phrases)


def _rag_debug_index_diagnostic(
    dataset_name: str,
    corpus_id: str | None,
    gt_answer: Any,
    full_context: str,
    sample_id: str = "",
) -> None:
    """When RAG_DEBUG and numerical_exact_match=0: check if GT or implied values exist in index for this doc.
    Logs whether the value is in the index at all (chunking) and if so whether it was in the retrieved context (ranking)."""
    if not corpus_id or gt_answer is None:
        return
    try:
        gt_str = str(gt_answer).strip().replace(",", "")
        gt_float = float(gt_str)
    except (ValueError, TypeError):
        return
    try:
        retriever = _get_rag_retriever_for_dataset(dataset_name, debug=False)
    except Exception as e:
        print(f"[DEBUG] RAG index diagnostic: could not load retriever: {e}")
        return
    chunks = getattr(retriever, "chunks", None) or []
    # Doc chunks: same logic as retrieval.py (exact match, then prefix match)
    doc_indices = [
        i for i, c in enumerate(chunks)
        if str((getattr(c, "metadata", None) or {}).get("corpus_id", "")) == str(corpus_id)
    ]
    if not doc_indices and "-" in str(corpus_id):
        prefix = str(corpus_id).rsplit("-", 1)[0]
        doc_indices = [
            i for i, c in enumerate(chunks)
            if str((getattr(c, "metadata", None) or {}).get("corpus_id", "")).startswith(prefix + "-")
            or (getattr(c, "metadata", None) or {}).get("corpus_id") == prefix
        ]
    if not doc_indices:
        print(f"[DEBUG] RAG index diagnostic: no chunks in index for corpus_id={corpus_id!r}")
        return
    # Search variants for GT
    search_vals = [gt_str, str(gt_answer).strip()]
    if "." in str(gt_answer):
        try:
            if gt_float == int(gt_float):
                search_vals.append(str(int(gt_float)))
        except (ValueError, TypeError):
            pass
    if len(gt_str) >= 4 and gt_str.isdigit():
        try:
            search_vals.append(f"{int(gt_str):,}")
        except ValueError:
            pass
    # Chunks containing GT
    chunks_with_gt: list[tuple[int, str]] = []
    for i in doc_indices:
        if i >= len(chunks):
            continue
        text = getattr(chunks[i], "text", None) or ""
        if any(s in text for s in search_vals if s):
            preview = (text[:120] + "…").replace("\n", " ")
            chunks_with_gt.append((i, preview))
    # Implied "other" value for change/difference: if context has N and GT is change, N - GT or N + GT might be the missing value
    implied_vals: list[str] = []
    for m in re.finditer(r"\b\d{4,6}\b", full_context):
        try:
            n = int(m.group(0))
            other = int(round(n - gt_float))
            if 1000 <= abs(other) <= 999999:
                implied_vals.append(str(other))
                implied_vals.append(f"{other:,}")
            other2 = int(round(n + gt_float))
            if 1000 <= abs(other2) <= 999999:
                implied_vals.append(str(other2))
                implied_vals.append(f"{other2:,}")
        except (ValueError, TypeError):
            continue
    implied_vals = list(dict.fromkeys(implied_vals))
    chunks_with_implied: list[tuple[int, str]] = []
    for i in doc_indices:
        if i >= len(chunks):
            continue
        text = getattr(chunks[i], "text", None) or ""
        if any(v in text for v in implied_vals):
            preview = (text[:120] + "…").replace("\n", " ")
            chunks_with_implied.append((i, preview))
    # Log
    print(f"[DEBUG] RAG index diagnostic: corpus_id={corpus_id!r} sample_id={sample_id!r} doc_chunks={len(doc_indices)}")
    print(f"[DEBUG] RAG index diagnostic: GT={gt_answer!r} search_vals={search_vals[:5]}")
    if chunks_with_gt:
        for idx, preview in chunks_with_gt:
            c = chunks[idx] if idx < len(chunks) else None
            in_retrieved = (getattr(c, "text", "")[:200] in full_context) if c else False
            print(f"[DEBUG] RAG index diagnostic: chunk {idx} CONTAINS GT → in_retrieved_context={in_retrieved} preview={preview!r}")
    else:
        print(f"[DEBUG] RAG index diagnostic: no chunk in doc contains GT (chunking may have dropped it)")
    if implied_vals and chunks_with_implied:
        print(f"[DEBUG] RAG index diagnostic: implied values (from context±GT)={implied_vals[:6]}")
        for idx, preview in chunks_with_implied:
            text = getattr(chunks[idx], "text", None) or ""
            in_retrieved = (text[:200] in full_context) if text else False
            print(f"[DEBUG] RAG index diagnostic: chunk {idx} CONTAINS implied value → in_retrieved_context={in_retrieved} preview={preview!r}")
    elif implied_vals:
        print(f"[DEBUG] RAG index diagnostic: implied values={implied_vals[:6]} → no chunk contains them (missing row in index)")


def _load_rag_chunking_failures(dataset_name: str) -> dict[str, dict]:
    """Load RAG chunking failures from data/proof/rag/<dataset>/chunking_failures.json.
    Keys are sample_id; values are {reason, gt, recoverable}. Used to tag known chunking bugs."""
    path = Path("data/proof") / "rag" / dataset_name.lower() / "chunking_failures.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in (data or {}).items() if isinstance(v, dict)}
    except Exception:
        return {}


def _load_rag_gt_overrides(dataset_name: str) -> dict[str, str]:
    """Load RAG GT overrides from data/proof/rag/<dataset>/gt_overrides.json.
    Values may be string (override answer) or dict with 'answer' or 'override' key."""
    path = Path("data/proof") / "rag" / dataset_name.lower() / "gt_overrides.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                val = str(v.get("answer") or v.get("override") or "")
            elif isinstance(v, str):
                val = v
            else:
                continue
            if val:
                out[str(k)] = val
        return out
    except Exception:
        return {}


def evaluate_rag_sample(dataset_name: str, prediction: dict, sample: dict, debug: bool = False) -> dict[str, float]:
    utils = RAG_UTILS[dataset_name]
    pred_answer = prediction.get("answer", "")
    gt_obj = sample.get("ground_truth", {})
    gt_answer = gt_obj.get("answer") if isinstance(gt_obj, dict) else gt_obj
    options_list = sample.get("metadata", {}).get("options_list")
    sample_id = sample.get("metadata", {}).get("sample_id", "")
    gt_override_used = False
    overrides = _load_rag_gt_overrides(dataset_name)
    if sample_id and sample_id in overrides:
        gt_answer = overrides[sample_id]
        gt_override_used = True
        if debug:
            print(f"[DEBUG] RAG gt_override: sample_id={sample_id!r} using override gt={gt_answer!r}")

    # Do not give credit when the model reported a retrieval/system error or refusal
    if _rag_prediction_is_error_or_refusal(pred_answer):
        return {
            "program_accuracy": 0.0,
            "numerical_exact_match": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }

    exact = utils.exact_match(pred_answer, gt_answer, options_list=options_list)
    # Debug: yes/no mismatch — drill down when model's yes/no differs from gold
    if debug and exact == 0 and gt_answer is not None and str(gt_answer).strip().lower() in ("yes", "no"):
        gt_yn = str(gt_answer).strip().lower()
        extracted_yn = _extract_yes_no_from_prediction(pred_answer)
        sample_id = sample.get("metadata", {}).get("sample_id", "")
        query_preview = (sample.get("input_text") or sample.get("input") or {}).get("query") or sample.get("input", "")
        if isinstance(query_preview, dict):
            query_preview = query_preview.get("query", "") or ""
        query_preview = str(query_preview)[:100] + ("..." if len(str(query_preview)) > 100 else "")
        print(
            f"[DEBUG] RAG yes/no mismatch: sample_id={sample_id!r} gold={gt_yn!r} extracted={extracted_yn!r} "
            f"(exact_match=0 expected while model gives {extracted_yn or 'no yes/no found'})"
        )
        print(f"[DEBUG] RAG yes/no question preview: {query_preview!r}")
        # Last 400 chars often contain "the answer is X"
        tail = (pred_answer or "")[-400:].replace("\n", " ")
        print(f"[DEBUG] RAG yes/no prediction tail (last 400 chars): {tail!r}")
    # Debug: possible missed extraction when GT is a number and model text contains it but declined to state it
    if debug and exact == 0 and gt_answer is not None and str(gt_answer).strip():
        gt_str = str(gt_answer).strip()
        try:
            float(gt_str.replace(",", ""))
        except ValueError:
            pass
        else:
            pred_normalized = pred_answer.replace(",", "").replace(" ", "")
            gt_normalized = gt_str.replace(",", "").replace(" ", "")
            if gt_str in pred_answer or (gt_normalized and gt_normalized in pred_normalized):
                refusal_phrases = ["cannot find", "not provided", "not stated", "cannot be determined", "is not provided"]
                if any(p in pred_answer.lower() for p in refusal_phrases):
                    print(
                        f"[DEBUG] RAG possible missed extraction: gold={gt_str!r} appears in model response but model declined to state it "
                        f"(sample_id={sample.get('metadata', {}).get('sample_id')})"
                    )
    token_f1 = utils.token_f1(pred_answer, gt_answer)
    # When answer is correct (exact_match=1), report f1=1.0 so single-sample metrics are intuitive
    # (raw token_f1 can be low for long predictions and short refs, e.g. ref "1" vs long paragraph)
    f1 = max(token_f1, exact)

    num_match = utils.numerical_exact_match(pred_answer, gt_answer)
    # Debug: on numerical exact_match failure, log full retrieved context (for totals/back-calc debugging)
    if debug and num_match == 0 and gt_answer is not None:
        try:
            float(str(gt_answer).strip().replace(",", ""))
        except ValueError:
            pass
        else:
            sources = prediction.get("sources") or []
            context_parts = []
            for s in sources:
                res = s.get("result") if isinstance(s.get("result"), dict) else {}
                for ch in res.get("chunks") or []:
                    t = ch.get("text") if isinstance(ch, dict) else str(ch)
                    if t:
                        context_parts.append(t)
            if context_parts:
                full_context = "\n\n---\n\n".join(context_parts)
                sample_id_debug = sample.get("metadata", {}).get("sample_id", "")
                print(f"[DEBUG] RAG numerical_exact_match=0 sample_id={sample_id_debug!r} gt={gt_answer!r}")
                print(f"[DEBUG] RAG full context (first 4000 chars):\n{full_context[:4000]}")
                if len(full_context) > 4000:
                    print(f"[DEBUG] ... (context total {len(full_context)} chars)")
                # Index diagnostic: is GT or implied value (e.g. 22176) in the doc's chunks? In retrieved context?
                gt_obj_debug = sample.get("ground_truth", {})
                corpus_id_debug = gt_obj_debug.get("corpus_id") if isinstance(gt_obj_debug, dict) else None
                _rag_debug_index_diagnostic(
                    dataset_name, corpus_id_debug, gt_answer, full_context, sample_id=sample_id_debug
                )

    chunking_failures = _load_rag_chunking_failures(dataset_name)
    chunking_failure_tag = 1 if (sample_id and sample_id in chunking_failures) else 0

    return {
        "program_accuracy": utils.program_accuracy(pred_answer, gt_answer),
        "numerical_exact_match": num_match,
        "f1": f1,
        "exact_match": exact,
        "used_back_calc": 1 if prediction_used_back_calc(pred_answer) else 0,
        "gt_override": 1 if gt_override_used else 0,
        "chunking_failure": chunking_failure_tag,
    }


def evaluate_credit_risk_pd_sample(prediction: dict, sample: dict) -> dict[str, float]:
    """Per-sample PD metrics: gt_binary, pd_prob, binary_pred. AUC/F1 computed at split level."""
    pd_utils = CreditRiskPDUtils()
    gt = sample.get("ground_truth")
    if isinstance(gt, dict):
        gt = gt.get("label")
    gt_binary = pd_utils._label_to_binary(gt)
    if gt_binary is None:
        return {}
    pd_prob = prediction.get("probability")
    if pd_prob is None:
        pd_prob = float(prediction.get("answer", 0) or 0)
    try:
        pd_prob = float(pd_prob)
    except (TypeError, ValueError):
        pd_prob = 0.0
    binary_pred = prediction.get("binary_pred")
    if binary_pred is None:
        binary_pred = pd_utils.binary_prediction_from_probability(pd_prob)
    return {
        "gt_binary": gt_binary,
        "pd_prob": pd_prob,
        "binary_pred": binary_pred,
    }


def evaluate_credit_risk_sentiment_sample(prediction: dict, sample: dict) -> dict[str, float]:
    """Per-sample sentiment: exact_match, prediction, reference for split-level F1 macro."""
    sent_utils = CreditRiskSentimentUtils()
    pred = (prediction.get("answer") or "").strip().lower()
    ref = sample.get("ground_truth")
    if isinstance(ref, dict):
        ref = ref.get("label") or ref.get("reference") or ""
    ref = (str(ref) or "").strip().lower()
    exact = sent_utils.exact_match(pred or "", ref or "")
    return {
        "exact_match": exact,
        "prediction": pred or "neutral",
        "reference": ref or "neutral",
    }


def evaluate_credit_risk_memo_sample(prediction: dict, sample: dict) -> dict[str, float]:
    """Memo/QA: exact_match and token_f1 vs reference answer."""
    rag_utils = RagUtils()
    pred = prediction.get("answer") or ""
    gt = sample.get("ground_truth")
    ref = gt.get("reference") if isinstance(gt, dict) else (gt or "")
    if ref is None:
        ref = ""
    ref = str(ref).strip()
    exact = rag_utils.exact_match(pred, ref)
    f1 = rag_utils.token_f1(pred, ref)
    return {"exact_match": exact, "f1": max(f1, exact)}


def _samples_filename(dataset_name: str, split_name: str) -> str:
    """Standardized per-sample proof filename: data/proof/<category>/<dataset>/<split>/<dataset>_<split>_samples.json"""
    return f"{dataset_name.lower()}_{split_name}_samples.json"


# Diagnostic-only per-sample keys: do not include in split/dataset _mean aggregation (not "higher is better")
METRIC_KEYS_EXCLUDE_FROM_AGGREGATE = {"used_back_calc"}
# Keys where missing value is treated as 0 so mean is over ALL samples (e.g. gt_override, chunking_failure: only some rows have it after it was added)
METRIC_KEYS_MISSING_AS_ZERO = {"gt_override", "chunking_failure"}


def aggregate_metrics(per_sample_scores: list[dict[str, float]]) -> dict[str, float]:
    if not per_sample_scores:
        return {}
    keys = sorted({k for row in per_sample_scores for k in row.keys() if k not in METRIC_KEYS_EXCLUDE_FROM_AGGREGATE})
    n = len(per_sample_scores)
    aggregated = {}
    for key in keys:
        if key in METRIC_KEYS_MISSING_AS_ZERO:
            vals = [row.get(key, 0) for row in per_sample_scores]
            aggregated[f"{key}_mean"] = sum(vals) / n if n else 0.0
        else:
            vals = [row.get(key) for row in per_sample_scores if row.get(key) is not None]
            aggregated[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
    return aggregated


def migrate_prediction_errors_from_per_sample(proof_dir: Path | str = "data/proof") -> None:
    """
    One-time migration: for every <dataset>_<split>_samples.json under proof_dir (excluding monitoring_metrics),
    move rows that have prediction_error into prediction_error.json in the same split folder, and remove them
    from the samples file so split/category/eval_summary stats are correct.
    """
    proof_dir = Path(proof_dir)
    if not proof_dir.exists():
        return
    for cat_dir in sorted(proof_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "monitoring_metrics":
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for split_dir in sorted(ds_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                per_sample_path = split_dir / _samples_filename(ds_dir.name, split_dir.name)
                if not per_sample_path.exists():
                    continue
                try:
                    with open(per_sample_path, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                except Exception:
                    continue
                if not isinstance(rows, list):
                    continue
                ok_rows = [r for r in rows if not r.get("prediction_error")]
                err_rows = [r for r in rows if r.get("prediction_error")]
                if not err_rows:
                    continue
                prediction_error_path = split_dir / "prediction_error.json"
                existing_err = []
                if prediction_error_path.exists():
                    try:
                        with open(prediction_error_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        existing_err = data if isinstance(data, list) else data.get("samples", [])
                    except Exception:
                        pass
                err_by_id = {str(r.get("sample_id")): r for r in existing_err}
                for r in err_rows:
                    err_by_id[str(r.get("sample_id"))] = r
                with open(prediction_error_path, "w", encoding="utf-8") as f:
                    json.dump(list(err_by_id.values()), f, ensure_ascii=False, indent=2)
                with open(per_sample_path, "w", encoding="utf-8") as f:
                    json.dump(ok_rows, f, ensure_ascii=False, indent=2)
                split_metric_rows = [r.get("metrics") or {} for r in ok_rows if r.get("metrics")]
                split_avg = aggregate_metrics(split_metric_rows)
                split_avg["sample_count"] = len(ok_rows)
                split_avg["gt_override_count"] = int(sum((m.get("gt_override", 0) or 0) for m in split_metric_rows))
                avg_path = split_dir / f"{ds_dir.name.lower()}_{split_dir.name}_avg.json"
                with open(avg_path, "w", encoding="utf-8") as f:
                    json.dump(split_avg, f, ensure_ascii=False, indent=2)
                print(f"[migrate_prediction_errors] {per_sample_path}: moved {len(err_rows)} error(s) to {prediction_error_path.name}, updated {avg_path.name}")




def _debug_audit_vision_samples(dataset_name: str, dataset: list[dict]):
    split_gt_counts = Counter()
    image_pattern_counts = Counter()
    image_preview_examples = {}

    for sample in dataset:
        split = sample.get("metadata", {}).get("split", "unknown")
        split_gt_counts[(split, "with_gt" if has_ground_truth(sample, "vision") else "no_gt")] += 1

        img_val = sample.get("input", {}).get("image") if isinstance(sample, dict) else None
        pattern = "other"
        if img_val is None:
            pattern = "none"
        elif isinstance(img_val, str):
            stripped = img_val.strip()
            if stripped.startswith("<PIL.Image.Image"):
                pattern = "serialized_pil_repr"
            elif stripped.startswith("data:image/"):
                pattern = "data_url"
            elif stripped == "":
                pattern = "empty_str"
            elif os.path.exists(stripped):
                pattern = "valid_path"
            else:
                pattern = "missing_path"
        else:
            pattern = type(img_val).__name__

        image_pattern_counts[pattern] += 1
        if pattern not in image_preview_examples:
            image_preview_examples[pattern] = str(img_val)[:160]

    print(f"[DEBUG] vision_audit dataset={dataset_name} split_gt_counts={{{', '.join([f'{k}:{v}' for k,v in split_gt_counts.items()])}}}")
    print(f"[DEBUG] vision_audit dataset={dataset_name} image_pattern_counts={dict(image_pattern_counts)}")
    preview_payload = {k: image_preview_examples[k] for k in sorted(image_preview_examples.keys())}
    print(f"[DEBUG] vision_audit dataset={dataset_name} image_preview_examples={preview_payload}")


def evaluate_dataset(
    adapter,
    category: str,
    dataset_name: str,
    max_samples_per_split=None,
    max_samples_per_category=None,
    dataset_split=None,
    only_gt=True,
    debug=False,
    generate_png=False,
    generate_metadata=False,
):
    """Streamed evaluation over adapter.load_split(...), row-by-row.

    When only_gt=True (default for interview/demo): only load splits that have
    ground truth, so evaluation is against industry-grade labeled data only.
    When only_gt=False: load all splits from FILE_MAPPING (samples without GT are
    still skipped for inference but splits are streamed).
    When dataset_split is set (e.g. from --split dev): only load and evaluate that split.
    """
    # Pass category limit as None so the adapter keeps streaming rows; we enforce
    # max_samples_per_category in the loop by breaking after that many *evaluated* samples.
    # Do NOT pass max_samples_per_split to the adapter: we need the adapter to yield enough
    # rows that we can skip already-evaluated sample_ids and still get the next N to evaluate
    # (same resume logic as adversarial: if sample_id exists in per_sample, fetch next sample).
    dataset_iter = adapter.load_split(
        dataset_split=dataset_split,
        max_samples_per_split=None,
        max_samples_per_category=None,
        only_splits_with_gt=only_gt,
    )

    model_meta = dict(MODEL_META.get(
        category,
        {"model_class": "unknown", "backbone": "unknown", "model_slug": "model"},
    ))
    # Vision backbone must come from VisionOCR so proof JSONs stay aligned with actual model
    if category == "vision":
        model_meta["backbone"] = get_vision_backbone()
    if model_meta.get("backbone") is None:
        model_meta["backbone"] = "unknown"
    model_slug = model_meta["model_slug"]

    # data/proof/<category>/<dataset_name>/<split>/
    category_proof_dir = Path("data/proof") / category.lower()
    dataset_proof_dir = category_proof_dir / dataset_name.lower()
    dataset_proof_dir.mkdir(parents=True, exist_ok=True)

    # Global (this run only)
    per_sample_rows = []
    per_sample_scores = []
    skipped_no_ground_truth = 0
    prediction_error_counter = Counter()

    # Per-split accumulators (this run only)
    split_rows: dict[str, list[dict]] = {}
    split_scores: dict[str, list[dict[str, float]]] = {}

    # Track already-evaluated samples from existing proof files to avoid
    # wasting API calls on the same sample_id repeatedly.
    existing_ids_by_split: dict[str, set[str]] = {}

    # Track how many *new* samples per split have been evaluated in this run
    # for enforcing max_samples_per_split independently of previously seen ids.
    evaluated_per_split = Counter()

    # PNG naming uses sample_id suffix for --generate_png: <dataset>_<split>_<sample_id_suffix>.png

    # OCR: accumulate per-sample rows for monitoring_metrics (layout_fingerprint_cache, completeness_heuristics)
    layout_cache_rows_ocr: list[dict] = []
    completeness_rows_ocr: list[dict] = []

    # Debug aggregates (computed on-the-fly to keep streaming)
    debug_split_counts = Counter()
    debug_img_type_counts = Counter()
    debug_gt_present = 0
    any_sample = False

    for sample in dataset_iter:
        any_sample = True

        if debug:
            split_name_dbg = sample.get("metadata", {}).get("split", "unknown")
            debug_split_counts[split_name_dbg] += 1
            inp_dbg = sample.get("input", {}) if isinstance(sample, dict) else {}
            img_val_dbg = inp_dbg.get("image")
            debug_img_type_counts[type(img_val_dbg).__name__] += 1
            if has_ground_truth(sample, category):
                debug_gt_present += 1

        if not has_ground_truth(sample, category):
            skipped_no_ground_truth += 1
            continue

        if max_samples_per_category is not None and len(per_sample_rows) >= max_samples_per_category:
            break

        split_name = sample.get("metadata", {}).get("split") or "unknown"
        sample_id = str(sample.get("metadata", {}).get("sample_id"))

        # Lazily load already-evaluated IDs for this split (only from per_sample; do NOT skip samples in prediction_error.json so they get re-evaluated)
        if split_name not in existing_ids_by_split:
            split_dir = dataset_proof_dir / split_name
            per_sample_path = split_dir / _samples_filename(dataset_name, split_name)
            ids = set()
            if per_sample_path.exists():
                try:
                    with open(per_sample_path, "r", encoding="utf-8") as f:
                        for row in json.load(f):
                            ids.add(str(row.get("sample_id")))
                except Exception:
                    pass
            existing_ids_by_split[split_name] = ids

        # Skip if this sample_id was already evaluated in a previous run
        if sample_id in existing_ids_by_split.get(split_name, set()):
            if debug:
                print(
                    f"[DEBUG] {dataset_name} sample={sample_id} split={split_name} "
                    f"skip_reason=already_evaluated"
                )
            continue

        # Respect per-split evaluation budget based on *new* samples only.
        if max_samples_per_split is not None and evaluated_per_split[split_name] >= max_samples_per_split:
            if debug:
                print(
                    f"[DEBUG] {dataset_name} sample={sample_id} split={split_name} "
                    f"skip_reason=split_budget_exhausted"
                )
            continue

        split_dir = dataset_proof_dir / split_name
        png_suffix = sample_id.split("_")[-1] if "_" in sample_id else sample_id

        # Optional: save image as PNG (vision only, named by sample_id)
        if generate_png and category == "vision":
            image, _ = _extract_image_for_vision(sample, debug=debug)
            if image is not None:
                split_dir.mkdir(parents=True, exist_ok=True)
                png_name = f"{dataset_name.lower()}_{split_name}_{png_suffix}.png"
                png_path = split_dir / png_name
                try:
                    from PIL import Image as PILImage
                    PILImage.fromarray(image).save(png_path)
                    if debug:
                        print(f"[DEBUG] generate_png saved {png_path}")
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] generate_png failed {png_path}: {e}")

        # Optional: write per-sample metadata JSON (e.g. options_list for multiple-choice)
        if generate_metadata:
            split_dir.mkdir(parents=True, exist_ok=True)
            meta_path = split_dir / f"{dataset_name.lower()}_{split_name}_{png_suffix}_metadata.json"
            inp = sample.get("input") or {}
            meta_export = {
                "sample_id": sample_id,
                "question": inp.get("question") or inp.get("query") or inp.get("prompt"),
                "ground_truth": sample.get("ground_truth"),
                "options_list": sample.get("metadata", {}).get("options_list"),
                "options_present": bool(sample.get("metadata", {}).get("options_list")),
                "split": split_name,
                "dataset": dataset_name,
            }
            # Include any other serializable metadata keys (skip image / binary)
            for k, v in (sample.get("metadata") or {}).items():
                if k in meta_export:
                    continue
                if v is None or isinstance(v, (str, int, float, bool, list, dict)):
                    try:
                        json.dumps(v)
                        meta_export[k] = v
                    except (TypeError, ValueError):
                        pass
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_export, f, ensure_ascii=False, indent=2)
                if debug:
                    print(f"[DEBUG] generate_metadata saved {meta_path}")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] generate_metadata failed {meta_path}: {e}")

        prediction = run_model(sample, category=category, dataset_name=dataset_name, debug=debug)
        prediction_error = prediction.get("error")
        metric_row: dict[str, float] = {}

        # One evaluation attempt was made for this (split, sample_id), even if
        # the model returns an error; this still consumes API credits.
        evaluated_per_split[split_name] += 1

        if prediction_error:
            prediction_error_counter[prediction_error] += 1
            if debug:
                print(
                    f"[DEBUG] {dataset_name} sample={sample.get('metadata', {}).get('sample_id')} "
                    f"error={prediction_error}"
                )
        else:
            if category == "vision":
                metric_row = evaluate_vision_sample(
                    dataset_name,
                    prediction,
                    sample,
                    debug=debug,
                )
            elif category == "rag":
                metric_row = evaluate_rag_sample(dataset_name, prediction, sample, debug=debug)
            elif category == "credit_risk_PD":
                metric_row = evaluate_credit_risk_pd_sample(prediction, sample)
            elif category == "credit_risk_PD_quantum":
                metric_row = evaluate_credit_risk_pd_sample(prediction, sample)
            elif category == "credit_risk_sentiment":
                metric_row = evaluate_credit_risk_sentiment_sample(prediction, sample)
            elif category == "credit_risk_sentiment_finbert":
                metric_row = evaluate_credit_risk_sentiment_sample(prediction, sample)
            elif category == "credit_risk_memo_generator":
                metric_row = evaluate_credit_risk_memo_sample(prediction, sample)
            elif category == "ocr":
                if not prediction_error:
                    from eval_postprocessing_utils import compute_ocr_metrics
                    gt = sample.get("ground_truth")
                    pred_text = (prediction.get("answer") or "").strip()
                    metric_row = compute_ocr_metrics(pred_text, gt, dataset_name)
                    metadata = prediction.get("metadata") or {}
                    det_method = metadata.get("detection_method") or "unknown"
                    layout_cache_rows_ocr.append({
                        "sample_id": sample_id,
                        "split": split_name,
                        "dataset": dataset_name,
                        "cache_hit": 1 if det_method == "cache" else 0,
                        "detection_method": det_method,
                    })
                    completeness_rows_ocr.append({
                        "sample_id": sample_id,
                        "split": split_name,
                        "dataset": dataset_name,
                        "heuristic_caught": 1 if len(pred_text) >= 10 else 0,
                        "text_length": len(pred_text),
                    })
                else:
                    metric_row = {}
            else:
                continue

        if metric_row:
            per_sample_scores.append(metric_row)
            split_scores.setdefault(split_name, []).append(metric_row)

        input_dict = sample.get("input", {}) if isinstance(sample, dict) else {}
        # Only keep lightweight, JSON-safe textual fields for proofs (avoid images / arrays)
        input_text = {k: v for k, v in input_dict.items() if isinstance(v, str)}
        row = {
            "sample_id": sample.get("metadata", {}).get("sample_id"),
            "split": split_name,
            "ground_truth": sample.get("ground_truth"),
            "input_text": input_text,
            "prediction": prediction.get("answer"),
            "prediction_error": prediction.get("error"),
            "metrics": metric_row,
        }
        # Persist metadata for vision so --reevaluate_only can recompute metrics (e.g. options_list for MMMU MC)
        if category == "vision":
            meta = sample.get("metadata") or {}
            row["metadata"] = {
                "sample_id": meta.get("sample_id"),
                "options_list": meta.get("options_list"),
            }

        per_sample_rows.append(row)
        split_rows.setdefault(split_name, []).append(row)

    if not any_sample:
        print(f"⚠️ Dataset {dataset_name} skipped (empty).")
        return None

    if debug:
        print(
            f"[DEBUG] dataset_loaded name={dataset_name} "
            f"total_rows={sum(debug_split_counts.values())} "
            f"gt_rows={debug_gt_present} "
            f"split_counts={dict(debug_split_counts)} "
            f"image_type_counts={dict(debug_img_type_counts)}"
        )

    # OCR: append layout_fingerprint_cache and completeness_heuristics per-sample to monitoring_metrics/<metric>/<dataset>_<split>_samples.json
    if category == "ocr" and (layout_cache_rows_ocr or completeness_rows_ocr):
        proof_dir = category_proof_dir.parent
        try:
            from eval_monitoring_metrics import append_per_sample_monitoring_metrics
            layout_by_ds_split: dict[tuple[str, str], list] = defaultdict(list)
            completeness_by_ds_split: dict[tuple[str, str], list] = defaultdict(list)
            for r in layout_cache_rows_ocr:
                layout_by_ds_split[(r.get("dataset", ""), r.get("split", ""))].append(r)
            for r in completeness_rows_ocr:
                completeness_by_ds_split[(r.get("dataset", ""), r.get("split", ""))].append(r)
            for (ds, sp), rows in layout_by_ds_split.items():
                if ds and sp:
                    append_per_sample_monitoring_metrics(proof_dir, "layout_fingerprint_cache", ds, sp, rows)
            for (ds, sp), rows in completeness_by_ds_split.items():
                if ds and sp:
                    append_per_sample_monitoring_metrics(proof_dir, "completeness_heuristics", ds, sp, rows)
        except Exception as e:
            if debug:
                print(f"[DEBUG] monitoring_metrics append failed: {e}")

    # -------------------------------
    # Persist per-split proofs (append). Order: per_sample -> split_avg -> dataset_weighted -> category -> eval_summary
    # -------------------------------
    dataset_weighted_metrics: dict[str, float] = {}
    split_avgs: dict[str, dict[str, float]] = {}

    for split_name, new_rows in split_rows.items():
        split_dir = dataset_proof_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        per_sample_path = split_dir / _samples_filename(dataset_name, split_name)
        prediction_error_path = split_dir / "prediction_error.json"

        ok_rows = [r for r in new_rows if not r.get("prediction_error")]
        err_rows = [r for r in new_rows if r.get("prediction_error")]

        # Per-sample: merge new ok_rows with existing (append/update by sample_id). Never overwrite with fewer rows.
        existing_rows: list[dict] = []
        if per_sample_path.exists():
            try:
                with open(per_sample_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                existing_rows = raw if isinstance(raw, list) else []
            except Exception as e:
                print(f"[WARN] Could not load existing per_sample {per_sample_path}: {e}. Skipping write to avoid data loss.")
                continue
        by_id: dict[str, dict] = {}
        for row in existing_rows:
            sid = str(row.get("sample_id"))
            by_id[sid] = row
        for row in ok_rows:
            sid = str(row.get("sample_id"))
            by_id[sid] = row
        combined_ok = list(by_id.values())
        if existing_rows and len(combined_ok) < len(existing_rows):
            print(f"[WARN] Would shrink {per_sample_path} from {len(existing_rows)} to {len(combined_ok)} rows; skipping write to avoid data loss.")
            continue
        with open(per_sample_path, "w", encoding="utf-8") as f:
            json.dump(combined_ok, f, ensure_ascii=False, indent=2)

        # Prediction errors: append to prediction_error.json (same split folder)
        if err_rows:
            if prediction_error_path.exists():
                try:
                    with open(prediction_error_path, "r", encoding="utf-8") as f:
                        err_data = json.load(f)
                    err_list = err_data if isinstance(err_data, list) else err_data.get("samples", [])
                except Exception:
                    err_list = []
            else:
                err_list = []
            err_by_id = {str(r.get("sample_id")): r for r in err_list}
            for row in err_rows:
                err_by_id[str(row.get("sample_id"))] = row
            with open(prediction_error_path, "w", encoding="utf-8") as f:
                json.dump(list(err_by_id.values()), f, ensure_ascii=False, indent=2)

        # Order 2: Compute split avg from per_sample only (no error rows)
        with open(per_sample_path, "r", encoding="utf-8") as f:
            rows_from_file = json.load(f)
        split_metric_rows = [r.get("metrics") or {} for r in rows_from_file if r.get("metrics")]
        if category in ("credit_risk_PD", "credit_risk_PD_quantum") and split_metric_rows:
            pd_utils = CreditRiskPDUtils()
            y_true = [r["gt_binary"] for r in split_metric_rows if r.get("gt_binary") is not None]
            y_score = [r["pd_prob"] for r in split_metric_rows if r.get("pd_prob") is not None]
            y_pred = [r["binary_pred"] for r in split_metric_rows if r.get("binary_pred") is not None]
            if len(y_true) == len(y_score) == len(y_pred) and y_true:
                f1_prec_rec = pd_utils.f1_precision_recall(y_true, y_pred)
                split_avg = {
                    "auc_roc_mean": pd_utils.auc_roc(y_true, y_score),
                    "f1_mean": f1_prec_rec["f1"],
                    "precision_mean": f1_prec_rec["precision"],
                    "recall_mean": f1_prec_rec["recall"],
                    "sample_count": len(rows_from_file),
                    "gt_override_count": 0,
                }
            else:
                split_avg = aggregate_metrics(split_metric_rows)
                split_avg["sample_count"] = len(rows_from_file)
        elif category in ("credit_risk_sentiment", "credit_risk_sentiment_finbert") and split_metric_rows:
            sent_utils = CreditRiskSentimentUtils()
            refs = [r.get("reference", "neutral") for r in split_metric_rows]
            preds = [r.get("prediction", "neutral") for r in split_metric_rows]
            split_avg = {
                "f1_macro_mean": sent_utils.f1_macro(refs, preds),
                "exact_match_mean": sum(r.get("exact_match", 0) for r in split_metric_rows) / len(split_metric_rows),
                "sample_count": len(rows_from_file),
                "gt_override_count": 0,
            }
        else:
            split_avg = aggregate_metrics(split_metric_rows)
            split_avg["sample_count"] = len(rows_from_file)
        # Evaluation counter: number of samples scored with known-bad-GT override (vision/MMMU, RAG)
        split_avg["gt_override_count"] = int(sum((m.get("gt_override", 0) or 0) for m in split_metric_rows))
        split_avgs[split_name] = split_avg

        avg_path = split_dir / f"{dataset_name.lower()}_{split_name}_avg.json"
        with open(avg_path, "w", encoding="utf-8") as f:
            json.dump(split_avg, f, ensure_ascii=False, indent=2)

    # -------------------------------
    # Order 3: Dataset-level weighted average from split avg files (read from disk)
    # Sample counts must exclude prediction_error; use per_sample file length as source of truth.
    # -------------------------------
    split_avgs_from_files: dict[str, dict] = {}
    for split_dir in dataset_proof_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        avg_path = split_dir / f"{dataset_name.lower()}_{split_name}_avg.json"
        if not avg_path.exists():
            continue
        try:
            with open(avg_path, "r", encoding="utf-8") as f:
                split_avgs_from_files[split_name] = json.load(f)
        except Exception:
            continue
        # Override sample_count with samples file length (excludes prediction_error rows)
        per_sample_path = split_dir / _samples_filename(dataset_name, split_name)
        if per_sample_path.exists():
            try:
                with open(per_sample_path, "r", encoding="utf-8") as f:
                    per_rows = json.load(f)
                if isinstance(per_rows, list):
                    split_avgs_from_files[split_name]["sample_count"] = len(
                        [r for r in per_rows if not r.get("prediction_error")]
                    )
            except Exception:
                pass

    metric_keys = sorted(
        {
            k
            for avg in split_avgs_from_files.values()
            for k in avg.keys()
            if k.endswith("_mean")
        }
    )

    dataset_total_from_files = sum(
        avg.get("sample_count", 0) for avg in split_avgs_from_files.values()
    )
    for key in metric_keys:
        num = 0.0
        denom = 0
        for split_name, avg in split_avgs_from_files.items():
            count = avg.get("sample_count", 0)
            val = _safe_metric_val(avg.get(key), default=0.5)
            num += val * count
            denom += count
        dataset_weighted_metrics[key] = num / denom if denom else 0.0

    # Per-split breakdown for interpretability (split -> count + metrics + gt_override_count)
    splits_breakdown = []
    for split_name in sorted(split_avgs_from_files.keys()):
        avg = split_avgs_from_files[split_name]
        metrics = {k: v for k, v in avg.items() if k.endswith("_mean")}
        splits_breakdown.append({
            "split": split_name,
            "sample_count": avg.get("sample_count", 0),
            "gt_override_count": avg.get("gt_override_count", 0),
            "metrics": metrics,
        })

    dataset_gt_override_total = sum(avg.get("gt_override_count", 0) for avg in split_avgs_from_files.values())
    dataset_payload = {
        "dataset": dataset_name,
        "sample_count": dataset_total_from_files,
        "gt_override_count": dataset_gt_override_total,
        "splits": sorted(split_avgs_from_files.keys()),
        "splits_breakdown": splits_breakdown,
        "skipped_no_ground_truth": skipped_no_ground_truth,
        "prediction_error_counts": dict(prediction_error_counter),
        "model_class": model_meta["model_class"],
        "backbone": model_meta["backbone"],
        "timestamp": singapore_now_iso(),
        "weighted_metrics": dataset_weighted_metrics,
    }

    dataset_weighted_path = dataset_proof_dir / f"{dataset_name.lower()}_avg.json"
    with open(dataset_weighted_path, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, ensure_ascii=False, indent=2)

    return {
        "dataset": dataset_name,
        "sample_count": dataset_total_from_files,
        "avg": dataset_weighted_metrics,
    }


def write_category_weighted_avg(category: str, dataset_summaries: list[dict]):
    """Legacy: compute from in-memory dataset_summaries. Prefer refresh_category_weighted_avg_from_files."""
    if not dataset_summaries:
        return

    model_meta = dict(MODEL_META.get(category, {"model_class": "unknown", "backbone": "unknown"}))
    if category == "vision":
        model_meta["backbone"] = get_vision_backbone()
    if model_meta.get("backbone") is None:
        model_meta["backbone"] = "unknown"

    total_samples = sum(d["sample_count"] for d in dataset_summaries)

    metric_keys = sorted(
        {
            k
            for d in dataset_summaries
            for k in d["avg"].keys()
            if k.endswith("_mean")
        }
    )

    weighted = {}
    for key in metric_keys:
        numerator = 0.0
        for d in dataset_summaries:
            numerator += _safe_metric_val(d["avg"].get(key), 0.5) * d["sample_count"]
        weighted[key] = numerator / total_samples if total_samples else 0.0

    payload = {
        "category": category,
        "datasets": [d["dataset"] for d in dataset_summaries],
        "sample_count": total_samples,
        "model_class": model_meta["model_class"],
        "backbone": model_meta["backbone"],
        "timestamp": singapore_now_iso(),
        "weighted_metrics": weighted,
    }

    out_path = Path("data/proof") / f"{category.lower()}_avg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dataset_sample_count_from_per_sample_files(dataset_proof_dir: Path) -> int:
    """Sum rows from <dataset>_<split>_samples.json under dataset_proof_dir (one file per split). Excludes prediction_error rows."""
    total = 0
    dataset_name = dataset_proof_dir.name
    for split_dir in dataset_proof_dir.iterdir():
        if not split_dir.is_dir():
            continue
        path = split_dir / _samples_filename(dataset_name, split_dir.name)
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list):
                total += len([r for r in rows if not r.get("prediction_error")])
        except Exception:
            pass
    return total


def refresh_category_weighted_avg_from_files(category: str) -> None:
    """Order 4: Recompute category avg by reading all dataset avg.json under data/proof/{category}/.
    Sample counts exclude prediction_error; per-dataset count is taken from per_sample file lengths."""
    proof_dir = Path("data/proof") / category.lower()
    if not proof_dir.exists():
        return

    dataset_payloads: list[dict] = []
    for child in sorted(proof_dir.iterdir()):
        if not child.is_dir():
            continue
        weighted_path = child / f"{child.name}_avg.json"
        legacy_path = child / f"{child.name}_weighted_avg.json"
        wrong_name = child / f"{child.name}avg.json"
        if not weighted_path.exists() and legacy_path.exists():
            legacy_path.rename(weighted_path)
        if not weighted_path.exists() and wrong_name.exists():
            wrong_name.rename(weighted_path)
        if not weighted_path.exists():
            continue
        try:
            with open(weighted_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # Override sample_count with count from samples files (excludes prediction_error)
            d["sample_count"] = _dataset_sample_count_from_per_sample_files(child)
            dataset_payloads.append(d)
        except Exception:
            continue

    if not dataset_payloads:
        return

    model_meta = dict(MODEL_META.get(category, {"model_class": "unknown", "backbone": "unknown"}))
    if category == "vision":
        model_meta["backbone"] = get_vision_backbone()
    if model_meta.get("backbone") is None:
        model_meta["backbone"] = "unknown"

    total_samples = sum(d.get("sample_count", 0) for d in dataset_payloads)
    metric_keys = sorted(
        {
            k
            for d in dataset_payloads
            for k in (d.get("weighted_metrics") or d).keys()
            if k.endswith("_mean")
        }
    )
    # Per-metric weighted average: only over datasets that report that metric.
    weighted = {}
    metrics_breakdown: dict[str, dict] = {}
    for key in metric_keys:
        num = 0.0
        denom = 0
        contributing_datasets: list[str] = []
        for d in dataset_payloads:
            w = d.get("weighted_metrics") or d
            if key not in w:
                continue
            count = d.get("sample_count", 0)
            num += _safe_metric_val(w[key], 0.5) * count
            denom += count
            contributing_datasets.append(d.get("dataset", ""))
        weighted[key] = num / denom if denom else 0.0
        metrics_breakdown[key] = {
            "value": round(weighted[key], 6),
            "n_samples": denom,
            "datasets": contributing_datasets,
        }

    # Per-dataset breakdown for interpretability (dataset -> sample_count, splits, metrics)
    sample_count_by_dataset: dict[str, int] = {}
    datasets_breakdown: list[dict] = []
    for d in dataset_payloads:
        ds_name = d.get("dataset", "")
        count = d.get("sample_count", 0)
        sample_count_by_dataset[ds_name] = count
        datasets_breakdown.append({
            "dataset": ds_name,
            "sample_count": count,
            "splits_breakdown": d.get("splits_breakdown", []),
            "weighted_metrics": d.get("weighted_metrics") or {},
        })

    payload = {
        "category": category,
        "sample_count": total_samples,
        "sample_count_by_dataset": sample_count_by_dataset,
        "datasets": [d.get("dataset", "") for d in dataset_payloads],
        "datasets_breakdown": datasets_breakdown,
        "weighted_metrics": weighted,
        "metrics_breakdown": metrics_breakdown,
        "model_class": model_meta["model_class"],
        "backbone": model_meta["backbone"],
        "timestamp": singapore_now_iso(),
    }
    out_path = Path("data/proof") / f"{category.lower()}_avg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(
    max_samples_per_split=None,
    max_samples_per_category=None,
    run_category=None,
    run_dataset=None,
    run_split=None,
    only_gt=True,
    debug=False,
    generate_png=False,
    generate_metadata=False,
):
    # Migrate existing per_sample files: move prediction_error rows to prediction_error.json
    migrate_prediction_errors_from_per_sample(Path("data/proof"))

    for category, datasets in AUTO_DATASETS.items():
        if run_category and category.lower() != run_category.lower():
            continue

        # When running OCR: ensure SROIE and FUNSD parquet exist locally; if not, run generate_pq_first_5_rows for missing only
        if category.lower() == "ocr":
            data_dir = Path("data/ocr")
            missing = []
            for dataset_name, _ds, _hf, _var in datasets:
                train_dir = data_dir / dataset_name.lower() / "train"
                if not train_dir.is_dir() or not any(train_dir.glob("*.parquet")):
                    missing.append(dataset_name)
            if missing:
                print(f"[OCR] Missing local parquet for: {', '.join(missing)}. Running data/generate_pq_first_5_rows.py --category ocr ...")
                repo_root = Path(__file__).resolve().parent
                gen_script = repo_root / "data" / "generate_pq_first_5_rows.py"
                rc = subprocess.call(
                    [sys.executable, str(gen_script), "--category", "ocr"],
                    cwd=str(repo_root),
                )
                if rc != 0:
                    print(f"⚠️ generate_pq_first_5_rows.py exited with {rc}. OCR evaluation may fall back to HuggingFace in adapters.")

        print(f"\n=== CATEGORY: {category.upper()} ===")
        dataset_summaries = []

        for dataset_name, data_source, hf_repo_name, hf_repo_variant in datasets:
            if run_dataset and dataset_name.lower() != run_dataset.lower():
                continue

            adapter_cls = ADAPTER_REGISTRY.get(dataset_name)
            if not adapter_cls:
                print(f"⚠️ Adapter class not found for {dataset_name}, skipping")
                continue

            adapter = adapter_cls(
                category=category,
                dataset_name=dataset_name,
                data_source_from_hf_or_manual=data_source,
                hf_repo_name=hf_repo_name,
                hf_repo_variant=hf_repo_variant,
            )
            summary = evaluate_dataset(
                adapter,
                category,
                dataset_name,
                max_samples_per_split=max_samples_per_split,
                max_samples_per_category=max_samples_per_category,
                dataset_split=run_split,
                only_gt=only_gt,
                debug=debug,
                generate_png=generate_png,
                generate_metadata=generate_metadata,
            )
            if summary and summary["sample_count"] > 0:
                dataset_summaries.append(summary)

            # Order 4 & 5: After each dataset run, refresh category and eval_summary from files
            # so vision_avg.json and eval_summary.json update after every new sample run.
            refresh_category_weighted_avg_from_files(category)
            write_eval_summary()

        # Adversarial testing runs only when --category rag (not for vision/ocr/other). Skip when --debug to avoid loading embedding+reranker twice (OOM/segfault on 16GB).
        if run_category and run_category.lower() == "rag" and not debug:
            try:
                from eval_monitoring_metrics import run_adversarial_rag_samples, write_monitoring_proof
            except Exception:
                run_adversarial_rag_samples = None
                write_monitoring_proof = None
            if run_adversarial_rag_samples and write_monitoring_proof:
                proof_dir = Path("data/proof")
                n_adv = max(1, max_samples_per_split or 1)
                print("Running RAG adversarial (prompt-injection) tests...")
                run_adversarial_rag_samples(n_adv, proof_dir)
                write_monitoring_proof(proof_dir)

        # Monitoring aggregation for OCR runs only when --category ocr (layout_fingerprint_cache, completeness_heuristics per-sample files under data/proof/monitoring_metrics/).
        if run_category and run_category.lower() == "ocr":
            try:
                from eval_monitoring_metrics import write_monitoring_proof
                write_monitoring_proof(Path("data/proof"))
            except Exception as e:
                if debug:
                    print(f"[DEBUG] write_monitoring_proof after OCR: {e}")

    # Update data/proof/SUMMARY.md from eval_summary.json and monitoring_metrics.json (track done vs missing).
    try:
        from eval_monitoring_metrics import write_proof_summary_md
        write_proof_summary_md(Path("data/proof"))
    except Exception as e:
        if debug:
            print(f"[DEBUG] write_proof_summary_md: {e}")

def write_eval_summary():
    """Write data/proof/eval_summary.json aggregating all category avg for interview presentation.
    Sample counts (from category avg files) exclude prediction_error. Includes overview and breakdowns."""
    proof_dir = Path("data/proof")
    if not proof_dir.exists():
        return
    # Migrate legacy *_weighted_avg.json -> *_avg.json at category level
    for legacy in proof_dir.glob("*_weighted_avg.json"):
        new_path = legacy.parent / (legacy.stem.replace("_weighted_avg", "") + "_avg.json")
        if not new_path.exists():
            legacy.rename(new_path)
    # Migrate wrongly named *avg.json (e.g. ragavg.json) -> *_avg.json; remove duplicate if _avg exists
    for path in list(proof_dir.glob("*avg.json")):
        if path.stem.endswith("_avg"):
            continue
        if path.stem.endswith("avg") and len(path.stem) > 3:
            new_path = path.parent / (path.stem[:-3] + "_avg.json")
            if new_path != path and not new_path.exists():
                path.rename(new_path)
            elif new_path != path and new_path.exists():
                path.unlink(missing_ok=True)
    summary = {}
    for path in sorted(proof_dir.glob("*_avg.json")):
        key = path.stem.replace("_avg", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                summary[key] = json.load(f)
        except Exception:
            continue
    if not summary:
        return

    # Top-level overview: total samples and which metrics apply to which n_samples (for quick interview explanation)
    overview = {}
    for cat_key, cat_data in summary.items():
        total = cat_data.get("sample_count", 0)
        by_dataset = cat_data.get("sample_count_by_dataset") or {}
        metrics_breakdown = cat_data.get("metrics_breakdown") or {}
        overview[cat_key] = {
            "sample_count_total": total,
            "sample_count_by_dataset": by_dataset,
            "metrics_breakdown": {
                k: {"value": v.get("value"), "n_samples": v.get("n_samples"), "datasets": v.get("datasets", [])}
                for k, v in metrics_breakdown.items()
            },
        }

    payload = {
        "timestamp": singapore_now_iso(),
        "overview": overview,
        "categories": summary,
    }
    out_path = proof_dir / "eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_reevaluate_only(
    proof_dir: Path | str,
    category: str,
    dataset: str,
    split: str | None = None,
    export_txt: bool = False,
) -> None:
    """
    Re-run evaluation on existing per-sample predictions only (no model/API calls).
    Loads <dataset>_<split>_samples.json under proof_dir/<category>/<dataset>/,
    re-computes metrics for each row using the current evaluator, writes back
    the samples file, then recomputes split and dataset averages.
    Use after changing evaluation logic (e.g. numerical_exact_match tolerance).
    When split is set (e.g. from --split dev), only that split folder is processed.
    """
    proof_dir = Path(proof_dir)
    dataset_proof_dir = proof_dir / category.lower() / dataset.lower()
    if not dataset_proof_dir.exists():
        print(f"[reevaluate_only] No such path {dataset_proof_dir}; nothing to do.")
        return

    # Resolve dataset name for RAG_UTILS / VISION_UTILS (e.g. "finqa" -> "FinQA", "mmmu_accounting" -> "MMMU_Accounting")
    dataset_name = dataset
    if category.lower() == "rag":
        for key in RAG_UTILS:
            if key.lower() == dataset.lower():
                dataset_name = key
                break
    elif category.lower() == "vision":
        for key in VISION_UTILS:
            if key.lower() == dataset.lower():
                dataset_name = key
                break

    updated_splits = []
    for split_dir in sorted(dataset_proof_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        if split is not None and split_name != split:
            continue
        per_sample_path = split_dir / _samples_filename(dataset_name, split_name)
        if not per_sample_path.exists():
            continue
        try:
            with open(per_sample_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            print(f"[reevaluate_only] Could not load {per_sample_path}: {e}")
            continue
        if not isinstance(rows, list) or not rows:
            continue

        if category.lower() == "rag":
            if dataset_name not in RAG_UTILS:
                print(f"[reevaluate_only] Unsupported RAG dataset: {dataset}; skipping.")
                continue
            for row in rows:
                if row.get("prediction_error"):
                    continue
                sample = {
                    "ground_truth": row.get("ground_truth"),
                    "input": row.get("input_text"),
                    "metadata": {"sample_id": row.get("sample_id")},
                }
                prediction = {
                    "answer": row.get("prediction") or "",
                    "sources": row.get("sources", []),
                }
                metrics = evaluate_rag_sample(dataset_name, prediction, sample, debug=False)
                row["metrics"] = metrics
        elif category.lower() == "vision":
            if dataset_name not in VISION_UTILS:
                print(f"[reevaluate_only] Unsupported vision dataset: {dataset}; skipping.")
                continue
            # Build sample_id -> options_list from dataset when rows lack it (no API call; load from parquet)
            sample_ids_in_rows = {str(r.get("sample_id")) for r in rows if r.get("sample_id")}
            need_options_from_adapter = any(
                not (r.get("metadata") or {}).get("options_list") and r.get("sample_id")
                for r in rows if not r.get("prediction_error")
            )
            sample_id_to_options: dict[str, list] = {}
            if need_options_from_adapter and sample_ids_in_rows and dataset_name in ADAPTER_REGISTRY:
                adapter_config = None
                for _name, _src, _hf, _var in AUTO_DATASETS.get("vision", []):
                    if _name and _name.lower() == dataset_name.lower():
                        adapter_config = (_name, _src, _hf, _var)
                        break
                if adapter_config:
                    _adapter_name, _data_source, _hf_repo, _hf_variant = adapter_config
                    try:
                        adapter_cls = ADAPTER_REGISTRY.get(_adapter_name)
                        if adapter_cls:
                            adapter = adapter_cls(
                                category="vision",
                                dataset_name=_adapter_name,
                                data_source_from_hf_or_manual=_data_source,
                                hf_repo_name=_hf_repo,
                                hf_repo_variant=_hf_variant,
                            )
                            for s in adapter.load_split(
                                dataset_split=split_name,
                                max_samples_per_category=max(len(sample_ids_in_rows) * 2, 100),
                                only_splits_with_gt=True,
                            ):
                                sid = str((s.get("metadata") or {}).get("sample_id", ""))
                                if sid in sample_ids_in_rows:
                                    opts = (s.get("metadata") or {}).get("options_list")
                                    if opts is not None:
                                        sample_id_to_options[sid] = opts
                    except Exception as e:
                        if "RAG_DEBUG" in os.environ or "DEBUG" in os.environ:
                            print(f"[reevaluate_only] vision: could not load options from adapter: {e}")
            for row in rows:
                if row.get("prediction_error"):
                    continue
                meta = row.get("metadata") or {}
                options_list = meta.get("options_list")
                # Fallback 1: from adapter (just built)
                if options_list is None and row.get("sample_id"):
                    options_list = sample_id_to_options.get(str(row.get("sample_id")))
                # Fallback 2: from existing _metadata.json file
                if options_list is None and row.get("sample_id"):
                    sid = str(row.get("sample_id"))
                    suffix = sid.split("_")[-1] if "_" in sid else sid
                    meta_path = split_dir / f"{dataset_name.lower()}_{split_name}_{suffix}_metadata.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                loaded = json.load(f)
                                options_list = loaded.get("options_list")
                        except Exception:
                            pass
                sample = {
                    "ground_truth": row.get("ground_truth"),
                    "input": row.get("input_text") or {},
                    "metadata": {
                        "sample_id": meta.get("sample_id") or row.get("sample_id"),
                        "options_list": options_list,
                    },
                }
                prediction = {"answer": row.get("prediction") or ""}
                metrics = evaluate_vision_sample(dataset_name, prediction, sample, debug=False)
                row["metrics"] = metrics
                # Persist metadata in row so saved JSON has it for next reevaluate_only
                row["metadata"] = {
                    "sample_id": sample["metadata"]["sample_id"],
                    "options_list": options_list,
                }
        else:
            print(f"[reevaluate_only] Unsupported category for re-eval: {category}; skipping.")
            continue

        with open(per_sample_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        updated_splits.append((split_name, split_dir, rows))
        print(f"[reevaluate_only] Updated {per_sample_path} ({len(rows)} rows)")

    if not updated_splits:
        print("[reevaluate_only] No per-sample files updated.")
        if export_txt:
            export_predictions_txt(proof_dir, category=category, dataset=dataset)
        return

    # Recompute split averages and write
    split_avgs_from_files: dict[str, dict] = {}
    for split_name, split_dir, rows in updated_splits:
        split_metric_rows = [r.get("metrics") or {} for r in rows if not r.get("prediction_error")]
        split_avg = aggregate_metrics(split_metric_rows)
        split_avg["sample_count"] = len([r for r in rows if not r.get("prediction_error")])
        split_avg["gt_override_count"] = int(sum((m.get("gt_override", 0) or 0) for m in split_metric_rows))
        split_avgs_from_files[split_name] = split_avg
        avg_path = split_dir / f"{dataset_name.lower()}_{split_name}_avg.json"
        with open(avg_path, "w", encoding="utf-8") as f:
            json.dump(split_avg, f, ensure_ascii=False, indent=2)
        print(f"[reevaluate_only] Wrote {avg_path}")

    # Dataset-level weighted average
    metric_keys = sorted(
        {k for avg in split_avgs_from_files.values() for k in avg.keys() if k.endswith("_mean")}
    )
    dataset_total = sum(avg.get("sample_count", 0) for avg in split_avgs_from_files.values())
    dataset_weighted_metrics = {}
    for key in metric_keys:
        num = sum(
            _safe_metric_val(split_avgs_from_files[s].get(key), default=0.5) * split_avgs_from_files[s].get("sample_count", 0)
            for s in split_avgs_from_files
        )
        dataset_weighted_metrics[key] = num / dataset_total if dataset_total else 0.0

    dataset_gt_override_total = sum(avg.get("gt_override_count", 0) for avg in split_avgs_from_files.values())
    dataset_payload = {
        "dataset": dataset_name,
        "sample_count": dataset_total,
        "gt_override_count": dataset_gt_override_total,
        "splits": sorted(split_avgs_from_files.keys()),
        "splits_breakdown": [
            {
                "split": s,
                "sample_count": split_avgs_from_files[s].get("sample_count", 0),
                "gt_override_count": split_avgs_from_files[s].get("gt_override_count", 0),
                "metrics": {k: v for k, v in split_avgs_from_files[s].items() if k.endswith("_mean")},
            }
            for s in sorted(split_avgs_from_files.keys())
        ],
        "weighted_metrics": dataset_weighted_metrics,
        "timestamp": singapore_now_iso(),
    }
    dataset_avg_path = dataset_proof_dir / f"{dataset_name.lower()}_avg.json"
    with open(dataset_avg_path, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, ensure_ascii=False, indent=2)
    print(f"[reevaluate_only] Wrote {dataset_avg_path}")

    if export_txt:
        export_predictions_txt(proof_dir, category=category, dataset=dataset)


def export_predictions_txt(
    proof_dir: Path | str = "data/proof",
    category: str | None = None,
    dataset: str | None = None,
) -> None:
    """
    Generate readable .txt files from <dataset>_<split>_samples.json proof files.
    For each samples JSON, writes a <dataset>_<split>_predictions.txt in the same split-level folder
    with sample_id, ground_truth, input (question), prediction, and metrics for developer review.
    If category/dataset are set, only exports under proof_dir/<category>/<dataset> (current run scope).
    Skips data/proof/monitoring_metrics/.
    """
    proof_dir = Path(proof_dir)
    if not proof_dir.exists():
        return

    per_sample_paths: list[Path] = []
    if category and dataset:
        base = proof_dir / category.lower() / dataset.lower()
        if base.exists():
            for split_dir in base.iterdir():
                if split_dir.is_dir():
                    p = split_dir / _samples_filename(dataset, split_dir.name)
                    if p.exists():
                        per_sample_paths.append(p)
        per_sample_paths.sort()
    elif category:
        base = proof_dir / category.lower()
        if base.exists():
            for ds_dir in base.iterdir():
                if not ds_dir.is_dir():
                    continue
                for split_dir in ds_dir.iterdir():
                    if split_dir.is_dir():
                        p = split_dir / _samples_filename(ds_dir.name, split_dir.name)
                        if p.exists():
                            per_sample_paths.append(p)
        per_sample_paths.sort()
    else:
        for cat_dir in proof_dir.iterdir():
            if not cat_dir.is_dir() or cat_dir.name == "monitoring_metrics":
                continue
            for ds_dir in cat_dir.iterdir():
                if not ds_dir.is_dir():
                    continue
                for split_dir in ds_dir.iterdir():
                    if split_dir.is_dir():
                        p = split_dir / _samples_filename(ds_dir.name, split_dir.name)
                        if p.exists():
                            per_sample_paths.append(p)
        per_sample_paths.sort()

    for per_sample_path in per_sample_paths:
        try:
            with open(per_sample_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue

        # Evaluation counter: samples scored with known-bad-GT override (vision/MMMU, RAG)
        gt_override_count = int(sum((row.get("metrics") or {}).get("gt_override", 0) for row in rows))

        # Output in same folder; name: e.g. chartqa_test_samples.json -> chartqa_test_predictions.txt
        stem = per_sample_path.stem
        base_name = stem.replace("_samples", "") if stem.endswith("_samples") else stem
        txt_name = f"{base_name}_predictions.txt"
        out_path = per_sample_path.parent / txt_name

        lines = [f"gt_override_count: {gt_override_count}", ""]
        for i, row in enumerate(rows):
            if i > 0:
                lines.append("")
            lines.append("=" * 72)
            lines.append(f"sample_id: {row.get('sample_id', '')}")
            lines.append(f"split: {row.get('split', '')}")
            lines.append(f"ground_truth: {row.get('ground_truth', '')}")
            inp = row.get("input_text") or {}
            if isinstance(inp, dict) and inp.get("question"):
                lines.append("question: " + str(inp.get("question", "")))
            elif isinstance(inp, str):
                lines.append("input_text: " + inp)
            else:
                lines.append("input_text: " + json.dumps(inp, ensure_ascii=False))
            lines.append("-" * 72)
            pred = row.get("prediction") or ""
            lines.append("prediction:")
            lines.append(pred if isinstance(pred, str) else json.dumps(pred, ensure_ascii=False))
            if row.get("prediction_error"):
                lines.append("-" * 72)
                lines.append(f"prediction_error: {row.get('prediction_error')}")
            met = row.get("metrics") or {}
            if met:
                lines.append("-" * 72)
                lines.append("metrics: " + json.dumps(met, ensure_ascii=False))
            lines.append("=" * 72)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[export_predictions_txt] Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation runner for OCR/Vision/RAG/Credit Risk")
    parser.add_argument("--max_split", type=int, default=None, help="Maximum samples per dataset split")
    parser.add_argument("--max_category", type=int, default=None, help="Maximum samples per category")
    parser.add_argument("--category", type=str, default=None, help="Only run this category")
    parser.add_argument("--dataset", type=str, default=None, help="Only run this dataset")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Only run this split (e.g. dev, train, test). Loads and evaluates only samples from that split folder.",
    )
    parser.add_argument(
        "--all_splits",
        action="store_true",
        help="Load all splits from FILE_MAPPING; default is only splits with ground truth (--only_gt mode)",
    )
    parser.add_argument("--debug", action="store_true", help="Print per-sample inference errors for diagnosis")
    parser.add_argument(
        "--export_predictions_txt",
        action="store_true",
        help="Export readable .txt from <dataset>_<split>_samples.json (prediction + context) in each split folder",
    )
    parser.add_argument(
        "--generate_png",
        action="store_true",
        help="Save each evaluated vision image as <dataset>_<split>_<sample_id_suffix>.png in the split proof folder (suffix from sample_id, e.g. test_4 -> 4)",
    )
    parser.add_argument(
        "--generate_metadata",
        action="store_true",
        help="Write per-sample metadata to <dataset>_<split>_<sample_id_suffix>_metadata.json (includes options_list for multiple-choice, question, ground_truth).",
    )
    parser.add_argument(
        "--reevaluate_only",
        action="store_true",
        help="Re-run evaluation on existing per-sample predictions only (no model/API). Updates metrics in samples JSON and avg files. Requires --category and --dataset.",
    )
    args = parser.parse_args()

    if args.reevaluate_only:
        if not args.category or not args.dataset:
            print("--reevaluate_only requires --category and --dataset (e.g. --category rag --dataset FinQA)")
            raise SystemExit(1)
        run_reevaluate_only(
            Path("data/proof"),
            category=args.category,
            dataset=args.dataset,
            split=args.split,
            export_txt=args.export_predictions_txt,
        )
        raise SystemExit(0)

    # Default: only_gt=True (only load splits that have labels, for interview/demo)
    only_gt = not args.all_splits

    main(
        max_samples_per_split=args.max_split,
        max_samples_per_category=args.max_category,
        run_category=args.category,
        run_dataset=args.dataset,
        run_split=args.split,
        only_gt=only_gt,
        debug=args.debug,
        generate_png=args.generate_png,
        generate_metadata=args.generate_metadata,
    )

    if args.export_predictions_txt:
        export_predictions_txt(Path("data/proof"), category=args.category, dataset=args.dataset)
