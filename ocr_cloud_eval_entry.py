"""OCR eval entrypoints for cloud notebooks when GitHub eval_runner.py lags behind local."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_OCR_HYBRID_CACHE: dict[str, Any] = {}
_OCR_WARMED_UP = False


def _ocr_use_ensemble(dataset_name: str) -> bool:
    try:
        from ocr_pipeline.ocr_eval_config import ocr_skip_tesseract_ensemble
    except ImportError:
        ocr_skip_tesseract_ensemble = lambda: os.environ.get(  # type: ignore[misc]
            "OCR_SKIP_TESSERACT_ENSEMBLE", ""
        ).strip().lower() in ("1", "true", "yes")
    if ocr_skip_tesseract_ensemble():
        return False
    return str(dataset_name).upper() in ("SROIE", "FUNSD")


def warmup_ocr_pipeline(dataset_name: str = "FUNSD") -> None:
    global _OCR_WARMED_UP
    if _OCR_WARMED_UP:
        return
    try:
        from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim

        install_paddle_langchain_shim()
        try:
            from ocr_pipeline.detection.paddleocr_detector import (
                PADDLEOCR_AVAILABLE,
                get_or_build_native_paddle_ocr,
            )
        except ImportError:
            from ocr_pipeline.detection.paddle_cloud_api import (
                PADDLEOCR_AVAILABLE,
                get_or_build_native_paddle_ocr,
            )
        if PADDLEOCR_AVAILABLE:
            get_or_build_native_paddle_ocr(show_log=False)
        cache_key = "ocr_ensemble" if _ocr_use_ensemble(dataset_name) else "ocr"
        if cache_key not in _OCR_HYBRID_CACHE:
            from ocr_pipeline.recognition.hybrid_ocr import HybridOCR

            _OCR_HYBRID_CACHE[cache_key] = HybridOCR(
                use_detection_router=False,
                use_vision_augmentation=False,
                use_ensemble_for_accuracy=_ocr_use_ensemble(dataset_name),
            )
        _OCR_WARMED_UP = True
        print("[OCR] Pipeline warmed up.", flush=True)
    except Exception as exc:
        print(f"[OCR] Warmup skipped: {exc}", flush=True)


def run_ocr_all_splits(
    *,
    datasets: list[str] | None = None,
    force_reeval: bool = False,
    proof_dir: str | Path = "data/proof",
    debug: bool = False,
) -> None:
    from eval_runner import ADAPTER_REGISTRY, AUTO_DATASETS, evaluate_dataset

    warmup_ocr_pipeline()
    splits_plan = [
        ("FUNSD", "train"),
        ("FUNSD", "test"),
        ("SROIE", "train"),
        ("SROIE", "test"),
    ]
    want = {d.upper() for d in (datasets or ["FUNSD", "SROIE"])}
    for ds_name, split in splits_plan:
        if ds_name.upper() not in want:
            continue
        adapter_cls = ADAPTER_REGISTRY.get(ds_name)
        if adapter_cls is None:
            continue
        meta = AUTO_DATASETS.get("ocr", [])
        src = "hf"
        hf_repo = None
        for entry in meta:
            if entry[0].upper() == ds_name.upper():
                src = entry[1]
                hf_repo = entry[2]
                break
        adapter = adapter_cls(
            category="ocr",
            dataset_name=ds_name,
            data_source_from_hf_or_manual=src,
            hf_repo_name=hf_repo,
        )
        print(f"\n=== OCR eval {ds_name}/{split} ===", flush=True)
        evaluate_dataset(
            adapter,
            "ocr",
            ds_name,
            dataset_split=split,
            force_reeval=force_reeval,
            proof_dir=proof_dir,
            debug=debug,
        )
