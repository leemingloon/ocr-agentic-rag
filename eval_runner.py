#!/usr/bin/env python3
"""
Unified evaluation runner for OCR / Vision / RAG / Credit Risk

Logic preserved:
- Skip datasets with existing evaluation proof
- Download HF datasets if marked automated
- Save evaluation scores to data/proof
- Delete datasets only if automated
- Supports quick-run via command-line argument
"""
import os
import io
import sys
import cv2
import time
import json
import argparse
import pytesseract
import numpy as np
from PIL import Image
from pathlib import Path

from dataset_adapters import (
    OCRDatasetAdapter,
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

# ------------------------
# Tesseract setup
# ------------------------
# Set tesseract_cmd, if TESSERACT_CMD environment variable is provided
tesseract_path = os.environ.get("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("✅ Using Tesseract at:", tesseract_path)
else:
    print("⚠️ Tesseract executable not found in environment variable 'TESSERACT_CMD'. Make sure to set it if Tesseract is not in your system PATH.")

print("RUNNING FILE:", __file__)

# ------------------------
# Datasets to evaluate
# ------------------------
AUTO_DATASETS = {
    "ocr": [ # Tesseract 5.x, PaddleOCR (PP-OCRv3 / PP-OCRv4) ONNX optimized
        ("SROIE", "hf", "jsdnrs/ICDAR2019-SROIE", None),
        ("FUNSD", "hf", "nielsr/funsd", None),
    ],

    "vision": [ # Claude 3.5 Sonnet
        ("DocVQA", "hf", "lmms-lab/DocVQA", "DocVQA"),
        ("ChartQA", "hf", "HuggingFaceM4/ChartQA", "vis-nlp/ChartQA"),
        ("InfographicsVQA", "hf", "lmms-lab/DocVQA", "InfographicVQA"),
        ("OmniDocBench", "hf", "Quivr/OmniDocBench", "full_dataset"),
        ("MMMU_Accounting", "hf", "MMMU/MMMU", "Accounting"),
        ("MMMU_Economics", "hf", "MMMU/MMMU", "Economics"),
        ("MMMU_Finance", "hf", "MMMU/MMMU", "Finance"),
        ("MMMU_Math", "hf", "MMMU/MMMU", "Math"),
    ],

    "rag": [ # LangGraph, BM25 + BGE-M3
        ("FinQA", "hf", "FinanceMTEB/FinQA", None),
        ("TATQA", "hf", None, None), # "next-tat/TAT-QA"
    ],

    "credit_risk_PD": [ # XGBoost
        ("LendingClub", "hf", "TheFinAI/lendingclub-benchmark", None),
    ],

    "credit_risk_sentiment": [ # FinBERT
        ("FinancialPhraseBank", "hf", "takala/financial_phrasebank", "sentences_allagree"),
        ("FiQA", "hf", "TheFinAI/fiqa-sentiment-classification", None),
    ],

    "credit_risk_memo_generator": [ # Claude 4 Sonnet
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

# ------------------------
# Import pipelines (real implementations)
# ------------------------
from ocr_pipeline import ClassicalDetector, TesseractOCR, HybridOCR
from ocr_pipeline.recognition.vision_ocr import VisionOCR
from rag_system import HybridRetriever, BGEReranker, AgenticRAG
from credit_risk import RatioBuilder,TrendEngine, PDModel, NLPSignalExtractor, CounterfactualAnalyzer, RiskMemoGenerator, PromptRegistry, SafetyFilter, DataDriftDetector, PredictionDriftDetector

classical_detector = ClassicalDetector(
    min_box_width = 10, # 50
    min_box_height = 5, # 20
    # max_box_width = 2000,
    # max_box_height = 500,
    morphology_kernel_size = (20, 1), # (50, 1)
)
# hybrid_ocr_pipeline = HybridOCR( # Tesseract 5.5
#     # tesseract_threshold = 85.0,
#     # paddleocr_threshold = 60.0,
#     # use_quality_assessment = True,
#     # use_detection_router = True,
#     # use_vision_augmentation = False,  # NEW
#     # vision_threshold = 60.0,  # NEW
# )
# vision_pipeline = VisionOCR( # real Vision engine instance
#     # api_key = None,
#     # model = "claude-3-5-sonnet-20241022",
#     # max_tokens = 1000,
# )
# retriever = HybridRetriever(
#     # embedding_model = "BAAI/bge-m3",
#     # top_k_sparse = 20,
#     # top_k_dense = 20,
#     # top_k_final = 10,
#     # rrf_k = 60,  # Reciprocal Rank Fusion parameter
# )
# reranker = BGEReranker(
#     # model_name = "BAAI/bge-reranker-v2-m3"
# )
# rag_pipeline = AgenticRAG( # real RAG system instance
#     retriever=retriever,
#     reranker=reranker
#     # api_key = None,
#     # model = "claude-sonnet-4-20250514"
# )
# pd_model = PDModel(
#     mode = "local",
#     # s3_bucket = None,
#     # s3_model_prefix = "models/pd/",
#     local_model_dir = "models/pd"
# )
# nlp_extractor = NLPSignalExtractor( # FinBERT
#     # model_name = "ProsusAI/finbert",
#     mode = "local",
#     # s3_bucket = None,
#     # s3_model_prefix = "models/finbert/",
#     # cache_dir = "models/cache/finbert"
# )
# generator = RiskMemoGenerator( # Claude 4 Sonnet
#     mode = "local",
#     # s3_bucket = None,
#     # s3_model_prefix = "models/pd/",
#     # local_model_dir = "models/pd"
# )

EMPTY_OCR_PREDICTION = {
    "model_output": {
        "words": [],
        "bboxes": [],  # [[x1, y1, x2, y2], ...]
        "confidence": 0.0
    }
}

# ------------------------
# Classical CV + Tesseract OCR
# ------------------------
def detect_and_recognize_image(classical_detector, image, template_type="invoice"):
    """
    Runs ClassicalDetector to detect text boxes, then Tesseract OCR per box.
    
    Args:
        classical_detector (ClassicalDetector): The detector instance to use
        image (np.ndarray): Input image (BGR or grayscale)
        template_type (str): Optional template hint (invoice/form/statement)
        
    Returns:
        dict: {
            "results": List[{"text": str, "bbox": (x,y,w,h)}],
            "confidence": float,
            "metadata": dict
        }
    """
    detection_result = classical_detector.detect(image, template_type=template_type)
    if not detection_result or not detection_result.boxes:
        print(f"[WARN] ClassicalDetector found no boxes for {getattr(image, 'image', 'unknown')}")
        return {"results": [{"text": "", "bbox": [0,0,0,0], "label": None}], "metadata": {}, "confidence": 0.0}
    boxes = detection_result.boxes
    texts = []

    # Ensure grayscale for Tesseract
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    for box in boxes:
        x, y, w, h = box
        crop = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(crop)
        texts.append({
            "text": text.strip(),
            "bbox": (int(x), int(y), int(w), int(h))
        })

    return {
        "results": texts,
        "confidence": detection_result.confidence,
        "metadata": detection_result.metadata
    }

# ------------------------
# Helpers
# ------------------------
def extract_image(sample):
    img = sample.get("image")
    if hasattr(img, "shape"):
        return img
    from PIL import Image
    import numpy as np
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if isinstance(img, str) and os.path.exists(img):
        return cv2.imread(img)
    print(f"[WARN] Cannot extract image for sample {sample.get('metadata', {}).get('sample_id')}")
    return None

def run_classical_ocr(image):
    if image is None:
        return EMPTY_OCR_PREDICTION
    det_res = classical_detector.detect(image, template_type="invoice")
    words, bboxes = [], []
    if det_res and det_res.boxes:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        for x, y, w, h in det_res.boxes:
            crop = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(crop).strip()
            if text:
                words.append(text)
                bboxes.append([int(x), int(y), int(x+w), int(y+h)])
    for idx, (text, b) in enumerate(zip(words, bboxes)):
        print(f"Box {idx}: {b} -> '{text}'")
    return {"model_output": {"words": words, "bboxes": bboxes, "confidence": det_res.confidence if det_res else 0.0}}

def run_model(sample, category, adapter=None):
    base_dir = getattr(adapter, "root_dir", None)

    # Extract model_input if structured sample
    model_input = sample.get("model_input") if isinstance(sample, dict) else sample

    if category == "ocr":
        if isinstance(adapter, SROIEAdapter):
            predicted_entities = adapter.extract_sroie_entities(
                sample["input"]["ocr"]["words"]
            )
            
            result = {
                "document_entities": predicted_entities
            }

            # ---- CER-based soft matching ----
            # Only do this if GT exists in the sample
            gt_entities = sample.get("ground_truth", {}).get("document_entities")
            if gt_entities:
                # Normalize both sides
                def normalize_text(s):
                    return "" if not s else s.lower().replace(" ", "").replace(".", "").replace(",", "")
                norm_pred = {k: normalize_text(v) for k, v in predicted_entities.items()}
                norm_gt   = {k: normalize_text(v) for k, v in gt_entities.items()}

                p, r, f1 = adapter.soft_entity_match(norm_pred, norm_gt)
                result["metrics"] = {
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "f1_soft": f1
                }

            return result
            

        image = extract_image(sample)
        if image is None:
            return EMPTY_OCR_PREDICTION

        # Auto-classical OCR for SROIE / FUNSD
        if adapter and getattr(adapter, "dataset_name", "").upper() in ["SROIE", "FUNSD"]:
            ocr_result = run_classical_ocr(image)

            # Optional: print each detected box
            words = ocr_result.get("model_output", {}).get("words", [])
            bboxes = ocr_result.get("model_output", {}).get("bboxes", [])
            for idx, (text, b) in enumerate(zip(words, bboxes)):
                print(f"Box {idx}: {b} -> '{text}'")

            return ocr_result
            # detection_result = classical_detector.detect(image, template_type="invoice")
            # print(detection_result.boxes)
            # words, bboxes = [], []

            # if detection_result and detection_result.boxes:
            #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     for box in detection_result.boxes:
            #         x, y, w, h = box
            #         crop = gray[y:y+h, x:x+w]
            #         text = pytesseract.image_to_string(crop).strip()
            #         if text:
            #             words.append(text)
            #             bboxes.append([int(x), int(y), int(x+w), int(y+h)])

            # return {
            #     "model_output": {
            #         "words": words,
            #         "bboxes": bboxes,
            #         "confidence": detection_result.confidence if detection_result else 0.0
            #     }
            # }
            
        #     ocr_result = detect_and_recognize_image(
        #         classical_detector, 
        #         image, 
        #         template_type="invoice"
        #     ) or {}

        #     return {
        #         "model_output": {
        #             "results": ocr_result.get("results", []),
        #             "confidence": ocr_result.get("confidence", 0.0),
        #             "metadata": ocr_result.get("metadata", {})
        #         }
        #     }
            
        # ocr_result = hybrid_ocr_pipeline.process_document(image)
        # return {
        #     "model_output": ocr_result
        # }

    elif category == "vision":
        image = extract_image(sample)
        if image is None:
            return None

        vision_result = vision_pipeline.recognize(image)
        return {
            "model_output": vision_result
        }

    elif category == "rag":
        image = extract_image(sample)
        text_ctx = hybrid_ocr_pipeline.process_document(image) if image is not None else None
        visual_ctx = vision_pipeline.recognize(image) if image is not None else None
            
        rag_result = rag_pipeline.query(text_ctx, visual_ctx)
        return {
            "model_output": rag_result
        }
    
    elif category == "credit_risk_pd":
        probability_default = pd_model.predict_pd(model_input)
        return {
            "model_output": probability_default
        }
    
    elif category == "credit_risk_sentiment":
        sentiment_signals = nlp_extractor.extract_signals(model_input)
        return {
            "model_output": sentiment_signals
        }
    
    elif category == "credit_risk_memo_generator":
        risk_memo = generator.generate_memo(model_input)
        return {
            "model_output": risk_memo
        }

    return {"model_output": str(model_input)}

def evaluate_dataset(
    adapter,
    category,
    max_samples_per_split=None,
    max_samples_per_category=None
):
    if getattr(adapter, "automated", False):
        adapter.download()

    dataset = adapter.load_split(
        dataset_split=None, 
        max_samples_per_split=max_samples_per_split, 
        max_samples_per_category=max_samples_per_category
    )
    if not dataset:
        print(f"⚠️ Dataset {getattr(adapter, 'name', 'unknown')} skipped (empty/missing).")
        return None
    
    proof_dir = Path("data/proof")
    proof_dir.mkdir(parents=True, exist_ok=True)
    adapter_name = type(adapter).__name__
    if adapter_name is None:
        # fallback to root_dir basename if no name
        adapter_name = Path(adapter.root_dir).name

    total_samples = len(dataset)
    print(f"SAMPLE COUNT IN DATASET: {total_samples}")
    scores = []
    start_time = time.time()

    for i, sample in enumerate(dataset):
        result = run_model(sample, category, adapter=adapter)
        print("Result type:", type(result))
        print(type(adapter))
        print(type(adapter).__name__)
        
        # ================= OCR SPECIAL CASE =================
        if category == "ocr" and isinstance(adapter, OCRDatasetAdapter):
            ocr_output = result.get("model_output", {})

            # Directly read universal format fields
            
            print("GT regions:", sample.get("ground_truth"))
            # gt_tokens = sample.get("ground_truth", {}).get("token_labels", [])
            # gt_regions = [
            #     {"text": str(t), "bbox": [0,0,0,0]} for t in gt_tokens
            # ]
            gt_regions = sample.get("ground_truth", {}).get("regions", [])

            pred_words = ocr_output.get("words", [])
            pred_bboxes = ocr_output.get("bboxes", [])
            pred_regions = [
                {"text": str(t), "bbox": b} for t, b in zip(pred_words, pred_bboxes)
            ]

            # Token-aligned evaluation
            metrics = adapter.evaluate_all_ocr_metrics(
                gt_regions,
                pred_regions
            )
            scores.append(metrics)

            # Minimal debug
            print(f"[{i+1}/{total_samples}] Pred tokens: {len(pred_words)}", end="\r") # GT tokens: {len(gt_tokens)}, 
            continue  # skip to next sample
            
        # ================= ALL OTHER CATEGORIES =================
        prediction = result.get("model_output", "")
        gt_text = str(sample.get("answer") or "") if isinstance(sample, dict) else ""

        correct = 1 if str(gt_text) == str(prediction) else 0
        scores.append(correct)

        running_avg = sum(scores) / len(scores) * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        eta_seconds = (total_samples - (i + 1)) * avg_time
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

        print(
            f"[{type(adapter).__name__}] "
            f"{i+1}/{total_samples} | "
            f"Running Accuracy: {running_avg:.1f}% | "
            f"ETA: {eta_str}",
            end="\r"
        )
        
    # ================= OCR METRIC AGGREGATION =================
    if category == "ocr":
        if not scores:
            final_results = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "cer": 1.0,
                "wer": 1.0
            }
            print(f"{adapter.dataset_name}: {final_results}")
            return final_results
        else:
            avg_precision = np.mean([m["precision"] for m in scores])
            avg_recall = np.mean([m["recall"] for m in scores])
            avg_f1 = np.mean([m["f1"] for m in scores])
            avg_cer = np.mean([m["cer"] for m in scores])
            avg_wer = np.mean([m["wer"] for m in scores])

            final_results = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "cer": avg_cer,
                "wer": avg_wer
            }

            print(f"{adapter.dataset_name}: {final_results}")
            return final_results
    
    # ================= OTHER CATEGORIES FINAL =================
    weighted_avg = sum(scores)/len(scores) if scores else 0.0
    print(
        f"{type(adapter).__name__}: "
        f"Final Accuracy: {weighted_avg:.4f} "
        f"({sum(scores)}/{len(dataset)} samples matched)"
    )

    if hasattr(adapter, "cleanup"):
        adapter.cleanup()

    proof_file = proof_dir / f"{category}_{adapter_name}.json"
    os.makedirs("data/proof", exist_ok=True)
    with open(proof_file, "w") as f:
        json.dump(scores, f)

    return weighted_avg


def main(max_samples_per_split=None, max_samples_per_category=None, run_category=None, run_dataset=None):
    for category, datasets in AUTO_DATASETS.items():
        # Skip category if a specific category/dataset was chosen
        if run_category and category.lower() != run_category.lower():
            continue

        print(f"\n=== CATEGORY: {category.upper()} ===")
        category_scores = []

        for dataset_name, data_source_from_hf_or_manual, hf_repo_name, hf_repo_variant in datasets:
            # Skip dataset if a specific dataset was chosen
            if run_dataset and dataset_name.lower() != run_dataset.lower():
                continue

            adapter_cls = ADAPTER_REGISTRY.get(dataset_name)
            if not adapter_cls:
                print(f"⚠️ Adapter class not found for {dataset_name}, skipping")
                continue

            adapter_kwargs = {}
            # Pass dataset split only if adapter supports it
            if "dataset_split" in adapter_cls.__init__.__code__.co_varnames:
                adapter_kwargs["dataset_split"] = "train"
            adapter = adapter_cls(
                category=category,
                dataset_name=dataset_name,
                data_source_from_hf_or_manual=data_source_from_hf_or_manual,
                hf_repo_name=hf_repo_name,
                hf_repo_variant=hf_repo_variant
            )

            score = evaluate_dataset(
                adapter,
                category,
                max_samples_per_split=max_samples_per_split,
                max_samples_per_category=max_samples_per_category
            )
            if score is not None:
                category_scores.append(score)

        if category_scores:
            if category == "ocr":
                weighted_avg = sum([c["f1"] for c in category_scores]) / len(category_scores)
                print(f"\nWeighted Average F1 across all categories: {weighted_avg:.4f}")
            else:
                weighted_avg = sum(category_scores) / len(category_scores)
                print(f"\nWeighted Average for {category}: {weighted_avg:.4f}")
        else:
            print(f"No evaluated datasets for {category}")

# ------------------------
# Run script
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation runner for OCR/Vision/RAG/Credit Risk")
    parser.add_argument("--max_split", type=int, default=None, help="Maximum samples per dataset split")
    parser.add_argument("--max_category", type=int, default=None, help="Maximum samples per category")
    parser.add_argument("--category", type=str, default=None, help="Only run this category")
    parser.add_argument("--dataset", type=str, default=None, help="Only run this dataset")

    args = parser.parse_args()

    main(
        max_samples_per_split=args.max_split,
        max_samples_per_category=args.max_category,
        run_category=args.category,
        run_dataset=args.dataset
    )