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
import sys
import cv2
import time
import json
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
    MMMUAdapter,

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
# On local, set tesseract_cmd if environment variable is provided
tesseract_path = os.environ.get("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("✅ Using Tesseract at:", tesseract_path)
else:
    print("⚠️ Tesseract executable not found in environment variable 'TESSERACT_CMD'. Make sure to set it if Tesseract is not in your system PATH.")
# tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# if os.path.exists(tesseract_path):
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
#     print("✅ Using Tesseract at:", tesseract_path)
# else:
#     print("⚠️ Tesseract executable not found at:", tesseract_path)

print("RUNNING FILE:", __file__)

# ------------------------
# Datasets to evaluate
# ------------------------
AUTO_DATASETS = {
    "ocr": [ # Tesseract 5.x, PaddleOCR (PP-OCRv3 / PP-OCRv4) ONNX optimized
        ("SROIE", "manual", None, None),
        ("FUNSD", "manual", None, None),
    ],

    "vision": [ # Claude 3.5 Sonnet
        ("DocVQA", "hf", "lmms-lab/DocVQA", "DocVQA"),
        ("ChartQA", "hf", "HuggingFaceM4/ChartQA", "vis-nlp/ChartQA"),
        ("InfographicsVQA", "hf", "lmms-lab/DocVQA", "InfographicVQA"),
        ("OmniDocBench", "hf", "Quivr/OmniDocBench", "full_dataset"),
        ("MMMU", "hf", "MMMU/MMMU", ["Finance", "Economics", "Accounting", "Math"]),  # manual selection of subjects
    ],

    "rag": [ # LangGraph, BM25 + BGE-M3
        ("FinQA", "hf", "FinanceMTEB/FinQA", None),
        ("TAT-QA", "manual", None, None),
    ],

    "credit_risk_PD": [ # XGBoost
        ("LendingClub", "manual", None, None),
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
    "MMMU" : MMMUAdapter,

    "FinQA": FinQAAdapter,
    "TAT-QA": TATQAAdapter,

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
    # min_box_width = 50,
    # min_box_height = 20,
    # max_box_width = 2000,
    # max_box_height = 500,
    # morphology_kernel_size = (50, 1),
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
def extract_image(data, base_dir=None):
    import cv2
    import numpy as np
    from PIL import Image
    import io

    img_path = data.get("image") if isinstance(data, dict) else data
    if isinstance(img_path, Path) and img_path.exists():
        return cv2.imread(str(img_path))
    elif isinstance(img_path, Image.Image):
        return cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
    else:
        print(f"[WARN] File not found or unreadable: {img_path}")
        return None

    if hasattr(data, "shape"):
        return data
    if isinstance(data, Image.Image):
        return np.array(data)
    if isinstance(data, dict):
        if "image_path" in data:
            data = data["image_path"]
        elif "bytes" in data and data["bytes"] is not None:
            return np.array(Image.open(io.BytesIO(data["bytes"])))
        else:
            for v in data.values():
                img = extract_image(v, base_dir)
                if img is not None:
                    return img
    if isinstance(data, str):
        candidates = [data]
        if base_dir:
            candidates.insert(0, os.path.join(base_dir, data))
        for path in candidates:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return img
            name, ext = os.path.splitext(path)
            if not ext:
                for ext_try in [".jpg", ".png", ".jpeg", ".png"]:
                    trial = name + ext_try
                    if os.path.exists(trial):
                        img = cv2.imread(trial)
                        if img is not None:
                            return img
        print(f"[WARN] File not found or unreadable: {data}")
    return None


def run_model(sample, category, adapter=None):
    base_dir = getattr(adapter, "root_dir", None)

    # Extract model_input if structured sample
    model_input = sample.get("model_input") if isinstance(sample, dict) else sample

    if category == "ocr":
        image = extract_image(sample, base_dir)
        if image is None:
            return None

        # Auto-classical OCR for SROIE / FUNSD
        if adapter and getattr(adapter, "dataset_name", "").upper() in ["SROIE", "FUNSD"]:
            ocr_result = detect_and_recognize_image(classical_detector, image, template_type="invoice")
            return {
                "model_output": ocr_result
            }
            
        ocr_result = hybrid_ocr_pipeline.process_document(image)
        return {
            "model_output": ocr_result
        }

    elif category == "vision":
        image = extract_image(sample, base_dir)
        if image is None:
            return None

        vision_result = vision_pipeline.recognize(image)
        return {
            "model_output": vision_result
        }

    elif category == "rag":
        image = extract_image(sample, base_dir)
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
        dataset_split="train", 
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
        if result is None:
            continue
        print(type(adapter))
        print(type(adapter).__name__)
        # ================= OCR SPECIAL CASE =================
        if category == "ocr" and isinstance(adapter, OCRDatasetAdapter):
            # Convert HybridOCR output → unified regions
            pred_regions = []

            ocr_output = result.get("model_output")
            if isinstance(ocr_output, dict) and "results" in ocr_output:
                for r in ocr_output["results"]:
                    pred_regions.append({
                        "text": r.get("text", ""),
                        "bbox": r.get("bbox", None)
                    })

            if not pred_regions:
                print("[WARN] No prediction returned for sample:", sample.get('image'))
                continue  # skip this sample
            gt_regions = sample.get("regions", [])
            print("GT example:", gt_regions[0])
            print("PRED example:", pred_regions[0])
            print(len(gt_regions))
            print(len(pred_regions))
            metrics = adapter.evaluate_all(gt_regions, pred_regions)

            scores.append(metrics)
            continue
            
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
    if category == "ocr" and scores:
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

    print()
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


def main(max_samples_per_split=None, max_samples_per_category=None):
    for category, datasets in AUTO_DATASETS.items():
        print(f"\n=== CATEGORY: {category.upper()} ===")
        category_scores = []

        for dataset_name, data_source_from_hf_or_manual, hf_repo_name, hf_repo_variant in datasets:
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
    max_samples_per_split = None
    max_samples_per_category = None

    if len(sys.argv) >= 2:
        try:
            max_samples_per_split = int(sys.argv[1])
            print(f"⚡ max_samples_per_split = {max_samples_per_split}")
        except:
            pass

    if len(sys.argv) >= 3:
        try:
            max_samples_per_category = int(sys.argv[2])
            print(f"⚡ max_samples_per_category = {max_samples_per_category}")
        except:
            pass

    main(max_samples_per_split, max_samples_per_category)
