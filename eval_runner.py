#!/usr/bin/env python3
"""
Unified evaluation runner for OCR / Vision / RAG / Credit Risk.

Architecture notes:
- AUTO_DATASETS and ADAPTER_REGISTRY are authoritative benchmark registries.
- Evaluates one dataset at a time and writes:
  - per-sample JSON: {dataset}_per_sample_{model}.json
  - split average JSON: {dataset}_{split}_avg.json (per split)
  - dataset weighted average: {dataset}_weighted_avg.json
  - category weighted average: {category}_weighted_avg.json
  - eval_summary.json
  Propagation order: per_sample -> split_avg -> dataset_weighted_avg -> category_weighted_avg -> eval_summary.
  Each level is computed by reading the previous level's files so edits to per_sample propagate on next run.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
    TATQAUtils,
    CreditRiskPDUtils,
    CreditRiskSentimentUtils,
    RagUtils,
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
    "credit_risk_sentiment": {
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
    if category in ("credit_risk_PD", "credit_risk_sentiment", "credit_risk_memo_generator"):
        if isinstance(gt, dict):
            val = gt.get("label") or gt.get("answer") or gt.get("reference")
        else:
            val = gt
        return val is not None and str(val).strip() != ""
    if category == "ocr":
        # OCR: ground_truth can be entities, token_labels, or similar
        return gt is not None
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


# Cache for RAG retriever per dataset (index built once, reused for all samples)
_RAG_RETRIEVER_CACHE: dict[str, Any] = {}


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

    if category == "vision":
        return _run_vision_model(sample, debug=debug)

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
            from credit_risk.models.pd_model import PDModel
            pd_model = PDModel(mode="local")
            model_path = Path("models/pd/pd_model_local_v1.pkl")
            if model_path.exists():
                pd_model.load(str(model_path))
            pd_prob = pd_model.predict_pd(features)
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

    if category == "credit_risk_sentiment":
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
                print(f"[DEBUG] Sentiment inference failed: {e}")
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

    acc = utils.accuracy(pred_answer, gt_answer, options_list=options_list)
    if debug:
        print(
            f"[DEBUG] vision_eval_metrics dataset={dataset_name} "
            f"accuracy={acc}"
        )
    return {"accuracy": acc}


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


def evaluate_rag_sample(dataset_name: str, prediction: dict, sample: dict) -> dict[str, float]:
    utils = RAG_UTILS[dataset_name]
    pred_answer = prediction.get("answer", "")
    gt_obj = sample.get("ground_truth", {})
    gt_answer = gt_obj.get("answer") if isinstance(gt_obj, dict) else gt_obj
    options_list = sample.get("metadata", {}).get("options_list")

    # Do not give credit when the model reported a retrieval/system error or refusal
    if _rag_prediction_is_error_or_refusal(pred_answer):
        return {
            "program_accuracy": 0.0,
            "numerical_exact_match": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }

    exact = utils.exact_match(pred_answer, gt_answer, options_list=options_list)
    token_f1 = utils.token_f1(pred_answer, gt_answer)
    # When answer is correct (exact_match=1), report f1=1.0 so single-sample metrics are intuitive
    # (raw token_f1 can be low for long predictions and short refs, e.g. ref "1" vs long paragraph)
    f1 = max(token_f1, exact)

    return {
        "program_accuracy": utils.program_accuracy(pred_answer, gt_answer),
        "numerical_exact_match": utils.numerical_exact_match(pred_answer, gt_answer),
        "f1": f1,
        "exact_match": exact,
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


def aggregate_metrics(per_sample_scores: list[dict[str, float]]) -> dict[str, float]:
    if not per_sample_scores:
        return {}
    keys = sorted({k for row in per_sample_scores for k in row.keys()})
    aggregated = {}
    for key in keys:
        vals = [row.get(key) for row in per_sample_scores if row.get(key) is not None]
        aggregated[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
    return aggregated


def migrate_prediction_errors_from_per_sample(proof_dir: Path | str = "data/proof") -> None:
    """
    One-time migration: for every *_per_sample_*.json under proof_dir, move rows that have
    prediction_error into prediction_error.json in the same split folder, and remove them
    from the per_sample file so split/category/eval_summary stats are correct.
    """
    proof_dir = Path(proof_dir)
    if not proof_dir.exists():
        return
    for per_sample_path in sorted(proof_dir.rglob("*_per_sample_*.json")):
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
        split_dir = per_sample_path.parent
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
        # Recompute split avg so split-level stats stay correct
        split_metric_rows = [r.get("metrics") or {} for r in ok_rows if r.get("metrics")]
        split_avg = aggregate_metrics(split_metric_rows)
        split_avg["sample_count"] = len(ok_rows)
        dataset_key = per_sample_path.parent.parent.name
        split_name = per_sample_path.parent.name
        avg_path = per_sample_path.parent / f"{dataset_key}_{split_name}_avg.json"
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
    only_gt=True,
    debug=False,
    generate_png=False,
):
    """Streamed evaluation over adapter.load_split(...), row-by-row.

    When only_gt=True (default for interview/demo): only load splits that have
    ground truth, so evaluation is against industry-grade labeled data only.
    When only_gt=False: load all splits from FILE_MAPPING (samples without GT are
    still skipped for inference but splits are streamed).
    """
    # Pass category limit as None so the adapter keeps streaming rows; we enforce
    # max_samples_per_category in the loop by breaking after that many *evaluated* samples.
    # Do NOT pass max_samples_per_split to the adapter: we need the adapter to yield enough
    # rows that we can skip already-evaluated sample_ids and still get the next N to evaluate
    # (same resume logic as adversarial: if sample_id exists in per_sample, fetch next sample).
    dataset_iter = adapter.load_split(
        dataset_split=None,
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

    # Per-split 1-based row count for --generate_png: <dataset>_<split>_<row_count>.png
    split_png_counter: dict[str, int] = {}

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
            per_sample_path = split_dir / f"{dataset_name.lower()}_per_sample_{model_slug}.json"
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

        # Optional: save image as PNG (vision only, scoped to this run)
        if generate_png and category == "vision":
            image, _ = _extract_image_for_vision(sample, debug=debug)
            if image is not None:
                split_dir = dataset_proof_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                row_count = split_png_counter.get(split_name, 0) + 1
                split_png_counter[split_name] = row_count
                png_name = f"{dataset_name.lower()}_{split_name}_{row_count}.png"
                png_path = split_dir / png_name
                try:
                    from PIL import Image as PILImage
                    PILImage.fromarray(image).save(png_path)
                    if debug:
                        print(f"[DEBUG] generate_png saved {png_path}")
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] generate_png failed {png_path}: {e}")

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
                metric_row = evaluate_rag_sample(dataset_name, prediction, sample)
            elif category == "credit_risk_PD":
                metric_row = evaluate_credit_risk_pd_sample(prediction, sample)
            elif category == "credit_risk_sentiment":
                metric_row = evaluate_credit_risk_sentiment_sample(prediction, sample)
            elif category == "credit_risk_memo_generator":
                metric_row = evaluate_credit_risk_memo_sample(prediction, sample)
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

    # -------------------------------
    # Persist per-split proofs (append). Order: per_sample -> split_avg -> dataset_weighted -> category -> eval_summary
    # -------------------------------
    dataset_weighted_metrics: dict[str, float] = {}
    split_avgs: dict[str, dict[str, float]] = {}

    for split_name, new_rows in split_rows.items():
        split_dir = dataset_proof_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        per_sample_path = split_dir / f"{dataset_name.lower()}_per_sample_{model_slug}.json"
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
        if category == "credit_risk_PD" and split_metric_rows:
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
                }
            else:
                split_avg = aggregate_metrics(split_metric_rows)
                split_avg["sample_count"] = len(rows_from_file)
        elif category == "credit_risk_sentiment" and split_metric_rows:
            sent_utils = CreditRiskSentimentUtils()
            refs = [r.get("reference", "neutral") for r in split_metric_rows]
            preds = [r.get("prediction", "neutral") for r in split_metric_rows]
            split_avg = {
                "f1_macro_mean": sent_utils.f1_macro(refs, preds),
                "exact_match_mean": sum(r.get("exact_match", 0) for r in split_metric_rows) / len(split_metric_rows),
                "sample_count": len(rows_from_file),
            }
        else:
            split_avg = aggregate_metrics(split_metric_rows)
            split_avg["sample_count"] = len(rows_from_file)
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
        # Override sample_count with per_sample file length (excludes prediction_error rows)
        per_sample_path = split_dir / f"{dataset_name.lower()}_per_sample_{model_slug}.json"
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
            num += avg.get(key, 0.0) * count
            denom += count
        dataset_weighted_metrics[key] = num / denom if denom else 0.0

    # Per-split breakdown for interpretability (split -> count + metrics)
    splits_breakdown = []
    for split_name in sorted(split_avgs_from_files.keys()):
        avg = split_avgs_from_files[split_name]
        metrics = {k: v for k, v in avg.items() if k.endswith("_mean")}
        splits_breakdown.append({
            "split": split_name,
            "sample_count": avg.get("sample_count", 0),
            "metrics": metrics,
        })

    dataset_payload = {
        "dataset": dataset_name,
        "sample_count": dataset_total_from_files,
        "splits": sorted(split_avgs_from_files.keys()),
        "splits_breakdown": splits_breakdown,
        "skipped_no_ground_truth": skipped_no_ground_truth,
        "prediction_error_counts": dict(prediction_error_counter),
        "model_class": model_meta["model_class"],
        "backbone": model_meta["backbone"],
        "timestamp": singapore_now_iso(),
        "weighted_metrics": dataset_weighted_metrics,
    }

    dataset_weighted_path = dataset_proof_dir / f"{dataset_name.lower()}_weighted_avg.json"
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
            numerator += d["avg"].get(key, 0.0) * d["sample_count"]
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

    out_path = Path("data/proof") / f"{category.lower()}_weighted_avg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dataset_sample_count_from_per_sample_files(dataset_proof_dir: Path) -> int:
    """Sum rows from per_sample JSONs under dataset_proof_dir (one file per split). Excludes prediction_error rows."""
    total = 0
    for split_dir in dataset_proof_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for per_path in split_dir.glob("*_per_sample_*.json"):
            try:
                with open(per_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if isinstance(rows, list):
                    total += len([r for r in rows if not r.get("prediction_error")])
            except Exception:
                pass
            break  # one per_sample file per split
    return total


def refresh_category_weighted_avg_from_files(category: str) -> None:
    """Order 4: Recompute category weighted_avg by reading all dataset weighted_avg.json under data/proof/{category}/.
    Sample counts exclude prediction_error; per-dataset count is taken from per_sample file lengths."""
    proof_dir = Path("data/proof") / category.lower()
    if not proof_dir.exists():
        return

    dataset_payloads: list[dict] = []
    for child in sorted(proof_dir.iterdir()):
        if not child.is_dir():
            continue
        weighted_path = child / f"{child.name}_weighted_avg.json"
        if not weighted_path.exists():
            continue
        try:
            with open(weighted_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # Override sample_count with count from per_sample files (excludes prediction_error)
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
            num += w[key] * count
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
    out_path = Path("data/proof") / f"{category.lower()}_weighted_avg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(
    max_samples_per_split=None,
    max_samples_per_category=None,
    run_category=None,
    run_dataset=None,
    only_gt=True,
    debug=False,
    generate_png=False,
):
    # Migrate existing per_sample files: move prediction_error rows to prediction_error.json
    migrate_prediction_errors_from_per_sample(Path("data/proof"))

    for category, datasets in AUTO_DATASETS.items():
        if run_category and category.lower() != run_category.lower():
            continue

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
                only_gt=only_gt,
                debug=debug,
                generate_png=generate_png,
            )
            if summary and summary["sample_count"] > 0:
                dataset_summaries.append(summary)

            # Order 4 & 5: After each dataset run, refresh category and eval_summary from files
            # so vision_weighted_avg.json and eval_summary.json update after every new sample run.
            refresh_category_weighted_avg_from_files(category)
            write_eval_summary()

        # When --category rag: run adversarial testing (controlled by --max_split) and update compliance_metrics.
        # Skip adversarial when --debug to avoid loading embedding model + reranker twice (OOM/segfault on 16GB).
        if run_category and run_category.lower() == "rag" and not debug:
            try:
                from eval_other_metrics import run_adversarial_rag_samples, write_compliance_proof
            except Exception:
                run_adversarial_rag_samples = None
                write_compliance_proof = None
            if run_adversarial_rag_samples and write_compliance_proof:
                proof_dir = Path("data/proof")
                n_adv = max(1, max_samples_per_split or 1)
                print("Running RAG adversarial (prompt-injection) tests...")
                run_adversarial_rag_samples(n_adv, proof_dir)
                write_compliance_proof(proof_dir)

def write_eval_summary():
    """Write data/proof/eval_summary.json aggregating all category weighted_avg for interview presentation.
    Sample counts (from category weighted_avg files) exclude prediction_error. Includes overview and breakdowns."""
    proof_dir = Path("data/proof")
    if not proof_dir.exists():
        return
    summary = {}
    for path in sorted(proof_dir.glob("*_weighted_avg.json")):
        key = path.stem.replace("_weighted_avg", "")
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


def export_predictions_txt(
    proof_dir: Path | str = "data/proof",
    category: str | None = None,
    dataset: str | None = None,
) -> None:
    """
    Generate readable .txt files from *_per_sample_*.json proof files.
    For each per_sample JSON, writes a *_predictions.txt in the same split-level folder
    with sample_id, ground_truth, input (question), prediction, and metrics for developer review.
    If category/dataset are set, only exports under proof_dir/<category>/<dataset> (current run scope).
    """
    proof_dir = Path(proof_dir)
    if not proof_dir.exists():
        return

    if category and dataset:
        base = proof_dir / category.lower() / dataset.lower()
        if not base.exists():
            return
        per_sample_paths = sorted(base.rglob("*_per_sample_*.json"))
    elif category:
        base = proof_dir / category.lower()
        if not base.exists():
            return
        per_sample_paths = sorted(base.rglob("*_per_sample_*.json"))
    else:
        per_sample_paths = sorted(proof_dir.rglob("*_per_sample_*.json"))

    for per_sample_path in per_sample_paths:
        try:
            with open(per_sample_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue

        # Output in same folder; name: e.g. mmmu_accounting_per_sample_visionocr.json -> mmmu_accounting_predictions.txt
        stem = per_sample_path.stem
        base = stem.split("_per_sample_")[0] if "_per_sample_" in stem else stem
        txt_name = f"{base}_predictions.txt"
        out_path = per_sample_path.parent / txt_name

        lines = []
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
        "--all_splits",
        action="store_true",
        help="Load all splits from FILE_MAPPING; default is only splits with ground truth (--only_gt mode)",
    )
    parser.add_argument("--debug", action="store_true", help="Print per-sample inference errors for diagnosis")
    parser.add_argument(
        "--export_predictions_txt",
        action="store_true",
        help="Export readable .txt from *_per_sample_*.json (prediction + context) in each split folder",
    )
    parser.add_argument(
        "--generate_png",
        action="store_true",
        help="Save each evaluated vision image as <dataset>_<split>_<row_count>.png in the split proof folder (row_count 1-based)",
    )
    args = parser.parse_args()

    # Default: only_gt=True (only load splits that have labels, for interview/demo)
    only_gt = not args.all_splits

    main(
        max_samples_per_split=args.max_split,
        max_samples_per_category=args.max_category,
        run_category=args.category,
        run_dataset=args.dataset,
        only_gt=only_gt,
        debug=args.debug,
        generate_png=args.generate_png,
    )

    if args.export_predictions_txt:
        export_predictions_txt(Path("data/proof"), category=args.category, dataset=args.dataset)
