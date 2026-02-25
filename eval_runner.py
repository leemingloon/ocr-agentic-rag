#!/usr/bin/env python3
"""
Unified evaluation runner for OCR / Vision / RAG / Credit Risk.

Architecture notes:
- AUTO_DATASETS and ADAPTER_REGISTRY are authoritative benchmark registries.
- Evaluates one dataset at a time and writes:
  - per-sample JSON: {dataset}_per_sample_{model}.json
  - dataset average JSON: {dataset}_avg.json
  - category weighted average JSON: {category}_weighted_avg.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_dataset_adapters import (
    SROIEAdapter,
    FUNSDAdapter,
    DocVQAAdapter,
    ChartQAAdapter,
    InfographicsVQAAdapter,
    OmniDocBenchAdapter,
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
    OmniDocBenchUtils,
    TATQAUtils,
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
        ("OmniDocBench", "hf", "Quivr/OmniDocBench", "full_dataset"),
        ("MMMU_Accounting", "hf", "MMMU/MMMU", "Accounting"),
        ("MMMU_Economics", "hf", "MMMU/MMMU", "Economics"),
        ("MMMU_Finance", "hf", "MMMU/MMMU", "Finance"),
        ("MMMU_Math", "hf", "MMMU/MMMU", "Math"),
    ],
    "rag": [
        ("FinQA", "hf", "FinanceMTEB/FinQA", None),
        ("TATQA", "hf", None, None),
    ],
    "credit_risk_PD": [
        ("LendingClub", "hf", "TheFinAI/lendingclub-benchmark", None),
    ],
    "credit_risk_sentiment": [
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
    "OmniDocBench": OmniDocBenchAdapter,
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
    "OmniDocBench": OmniDocBenchUtils(),
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
    "vision": {
        "model_class": "VisionOCR",
        "backbone": "claude_3_5_sonnet",
        "model_slug": "visionocr",
    },
    "rag": {
        "model_class": "AgenticRAG",
        "backbone": "langgraph_hybrid_retriever_bge_reranker",
        "model_slug": "agenticrag",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return False




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

        return None, f"unsupported_image_type:{type(img).__name__}"
    except Exception as exc:
        return None, f"image_decode_exception:{exc}"


def _run_vision_model(sample: dict, debug: bool = False) -> dict:
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

    try:
        from ocr_pipeline.recognition.vision_ocr import VisionOCR
        vision = VisionOCR(api_key=api_key)
        # Use chart-aware path when question exists; otherwise generic recognize.
        if question:
            out = vision.extract_charts(image=image, question=question)
            return {"answer": str(out.get("chart_analysis", ""))}

        out = vision.recognize(image=image, task="extract")
        return {"answer": str(getattr(out, "text", ""))}
    except Exception as exc:
        return {"answer": "", "error": f"vision_inference_failed:{exc}"}


def run_model(sample: dict, category: str, dataset_name: str, debug: bool = False) -> dict:
    """Model inference hook.

    NOTE: this is intentionally a baseline stub so the evaluation plumbing can run end-to-end.
    Replace this with real pipeline calls (VisionOCR / AgenticRAG) per deployment mode.
    """
    sample_input = sample.get("input", {}) if isinstance(sample, dict) else {}

    if category == "vision":
        return _run_vision_model(sample, debug=debug)

    if category == "rag":
        query = sample_input.get("query") or sample_input.get("question") or ""
        # Placeholder until AgenticRAG is wired with retriever/reranker/model creds.
        return {
            "answer": "",
            "sources": [],
            "reasoning": f"baseline_stub_query={query[:80]}",
        }

    return {"answer": ""}


def evaluate_vision_sample(dataset_name: str, prediction: dict, sample: dict) -> dict[str, float]:
    utils = VISION_UTILS[dataset_name]
    pred_answer = prediction.get("answer", "")
    gt = sample.get("ground_truth")
    if isinstance(gt, list):
        gt_answer = gt[0] if gt else ""
    else:
        gt_answer = gt if gt is not None else ""

    if dataset_name in {"DocVQA", "InfographicsVQA", "OmniDocBench"}:
        return {
            "anls": utils.anls(pred_answer, gt_answer),
            "exact_match": utils.exact_match(pred_answer, gt_answer),
        }

    if dataset_name == "ChartQA":
        return {
            "strict_accuracy": utils.exact_match(pred_answer, gt_answer),
            "relaxed_accuracy": utils.relaxed_numeric_accuracy(pred_answer, gt_answer),
        }

    return {"accuracy": utils.accuracy(pred_answer, gt_answer)}


def evaluate_rag_sample(dataset_name: str, prediction: dict, sample: dict) -> dict[str, float]:
    utils = RAG_UTILS[dataset_name]
    pred_answer = prediction.get("answer", "")
    gt_obj = sample.get("ground_truth", {})
    gt_answer = gt_obj.get("answer") if isinstance(gt_obj, dict) else gt_obj

    return {
        "program_accuracy": utils.program_accuracy(pred_answer, gt_answer),
        "numerical_exact_match": utils.numerical_exact_match(pred_answer, gt_answer),
        "f1": utils.token_f1(pred_answer, gt_answer),
        "exact_match": utils.exact_match(pred_answer, gt_answer),
    }


def aggregate_metrics(per_sample_scores: list[dict[str, float]]) -> dict[str, float]:
    if not per_sample_scores:
        return {}
    keys = sorted({k for row in per_sample_scores for k in row.keys()})
    aggregated = {}
    for key in keys:
        vals = [row.get(key) for row in per_sample_scores if row.get(key) is not None]
        aggregated[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
    return aggregated




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


def evaluate_dataset(adapter, category: str, dataset_name: str, max_samples_per_split=None, max_samples_per_category=None, debug=False):
    # Load split-limited rows first; apply category cap after GT filtering
    # so test-only rows with missing labels do not consume the entire budget.
    dataset = adapter.load_split(
        dataset_split=None,
        max_samples_per_split=max_samples_per_split,
        max_samples_per_category=None,
    )
    if not dataset:
        print(f"⚠️ Dataset {dataset_name} skipped (empty).")
        return None

    if debug:
        img_type_counts = Counter()
        split_counts = Counter()
        gt_present = 0
        for sample in dataset:
            split_counts[sample.get("metadata", {}).get("split", "unknown")] += 1
            inp = sample.get("input", {}) if isinstance(sample, dict) else {}
            img_val = inp.get("image")
            img_type_counts[type(img_val).__name__] += 1
            if has_ground_truth(sample, category):
                gt_present += 1
        print(f"[DEBUG] dataset_loaded name={dataset_name} total_rows={len(dataset)} gt_rows={gt_present} split_counts={dict(split_counts)} image_type_counts={dict(img_type_counts)}")
        if category == "vision":
            _debug_audit_vision_samples(dataset_name, dataset)

    model_meta = MODEL_META.get(
        category,
        {"model_class": "unknown", "backbone": "unknown", "model_slug": "model"},
    )
    model_slug = model_meta["model_slug"]

    category_proof_dir = Path("data/proof") / category.lower()
    category_proof_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    per_sample_scores = []
    skipped_no_ground_truth = 0
    prediction_error_counter = Counter()

    for sample in dataset:
        if not has_ground_truth(sample, category):
            skipped_no_ground_truth += 1
            continue

        if max_samples_per_category is not None and len(per_sample_rows) >= max_samples_per_category:
            break

        prediction = run_model(sample, category=category, dataset_name=dataset_name, debug=debug)
        prediction_error = prediction.get("error")
        if prediction_error:
            prediction_error_counter[prediction_error] += 1
            if debug:
                print(f"[DEBUG] {dataset_name} sample={sample.get('metadata', {}).get('sample_id')} error={prediction_error}")

        if category == "vision":
            metric_row = evaluate_vision_sample(dataset_name, prediction, sample)
        elif category == "rag":
            metric_row = evaluate_rag_sample(dataset_name, prediction, sample)
        else:
            continue

        per_sample_scores.append(metric_row)
        per_sample_rows.append(
            {
                "sample_id": sample.get("metadata", {}).get("sample_id"),
                "split": sample.get("metadata", {}).get("split"),
                "ground_truth": sample.get("ground_truth"),
                "prediction": prediction.get("answer"),
                "prediction_error": prediction.get("error"),
                "metrics": metric_row,
            }
        )

    dataset_avg = aggregate_metrics(per_sample_scores)
    dataset_avg.update(
        {
            "sample_count": len(per_sample_rows),
            "skipped_no_ground_truth": skipped_no_ground_truth,
            "prediction_error_counts": dict(prediction_error_counter),
        }
    )

    per_sample_path = category_proof_dir / f"{dataset_name.lower()}_per_sample_{model_slug}.json"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_rows, f, ensure_ascii=False, indent=2)

    avg_path = category_proof_dir / f"{dataset_name.lower()}_avg.json"
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(dataset_avg, f, ensure_ascii=False, indent=2)

    return {
        "dataset": dataset_name,
        "sample_count": len(per_sample_rows),
        "avg": dataset_avg,
    }


def write_category_weighted_avg(category: str, dataset_summaries: list[dict]):
    if not dataset_summaries:
        return

    model_meta = MODEL_META.get(category, {"model_class": "unknown", "backbone": "unknown"})
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
            numerator += d["avg"].get(key, 0.0) * d["sample_count"]
        weighted[key] = numerator / total_samples if total_samples else 0.0

    payload = {
        "category": category,
        "datasets": [d["dataset"] for d in dataset_summaries],
        "sample_count": total_samples,
        "model_class": model_meta["model_class"],
        "backbone": model_meta["backbone"],
        "timestamp": utc_now_iso(),
        "weighted_metrics": weighted,
    }

    out_path = Path("data/proof") / f"{category.lower()}_weighted_avg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(max_samples_per_split=None, max_samples_per_category=None, run_category=None, run_dataset=None, debug=False):
    for category, datasets in AUTO_DATASETS.items():
        if run_category and category.lower() != run_category.lower():
            continue

        print(f"\n=== CATEGORY: {category.upper()} ===")
        dataset_summaries = []

        for dataset_name, data_source, hf_repo_name, hf_repo_variant in datasets:
            if run_dataset and dataset_name.lower() != run_dataset.lower():
                continue
            if category.lower() not in {"vision", "rag"}:
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
                debug=debug,
            )
            if summary and summary["sample_count"] > 0:
                dataset_summaries.append(summary)

        write_category_weighted_avg(category, dataset_summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation runner for OCR/Vision/RAG/Credit Risk")
    parser.add_argument("--max_split", type=int, default=None, help="Maximum samples per dataset split")
    parser.add_argument("--max_category", type=int, default=None, help="Maximum samples per category")
    parser.add_argument("--category", type=str, default=None, help="Only run this category")
    parser.add_argument("--dataset", type=str, default=None, help="Only run this dataset")
    parser.add_argument("--debug", action="store_true", help="Print per-sample inference errors for diagnosis")
    args = parser.parse_args()

    main(
        max_samples_per_split=args.max_split,
        max_samples_per_category=args.max_category,
        run_category=args.category,
        run_dataset=args.dataset,
        debug=args.debug,
    )
