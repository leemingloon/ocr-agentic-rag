#!/usr/bin/env python3
"""Full trace + preflight for FUNSD/SROIE OCR eval on Kaggle/Colab.

Run from repo root:
  python scripts/audit_ocr_eval_cloud.py
  OCR_USE_GPU=1 python scripts/audit_ocr_eval_cloud.py --smoke-paddle

Exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# pip packages (not full requirements.txt — no torch/langchain/anthropic)
REQUIRED_PACKAGES = [
    "numpy",
    "cv2",
    "PIL",
    "pyarrow",
    "datasets",
    "scipy",
    "pandas",
    "pytesseract",
    "tqdm",
    "dotenv",
    "requests",
]
OPTIONAL_GPU = ["paddle", "paddleocr"]

IMPORT_STEPS = [
    ("ocr_pipeline root", "ocr_pipeline"),
    ("ocr_eval_config", "ocr_pipeline.ocr_eval_config"),
    ("paddle_langchain_shim", "ocr_pipeline.compat.paddle_langchain_shim"),
    ("paddleocr_detector (direct)", "ocr_pipeline.detection.paddleocr_detector"),
    ("hybrid_ocr (direct)", "ocr_pipeline.recognition.hybrid_ocr"),
    ("eval_postprocess_utils", "eval_postprocess_utils"),
    ("eval_dataset_adapters", "eval_dataset_adapters"),
    ("eval_runner", "eval_runner"),
]

REPO_INVARIANTS = [
    ("slim ocr_pipeline __init__", REPO / "ocr_pipeline" / "__init__.py", 'HybridOCR",'),
    ("ocr_eval_config paddle_rec_batch_num", REPO / "ocr_pipeline" / "ocr_eval_config.py", "def paddle_rec_batch_num"),
    ("hybrid lazy vision", REPO / "ocr_pipeline" / "recognition" / "hybrid_ocr.py", "use_vision_augmentation"),
    ("strict paddle default", REPO / "ocr_pipeline" / "recognition" / "hybrid_ocr.py", "OCR_EVAL_STRICT_PADDLE"),
]


def _env_defaults() -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("OCR_USE_GPU", os.environ.get("OCR_USE_GPU", "0"))
    os.environ.setdefault("OCR_SKIP_TESSERACT_ENSEMBLE", "1")
    os.environ.setdefault("OCR_FAST", "1")
    os.environ.setdefault("OCR_REC_BATCH_NUM", "16")
    os.environ.setdefault("OCR_PADDLE_MIN_SIDE", "720")


def _run_py(snippet: str, *, env: dict | None = None) -> tuple[int, str]:
    full = f"import os, sys\nsys.path.insert(0, {str(REPO)!r})\n{snippet}"
    proc = subprocess.run(
        [sys.executable, "-c", full],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR cloud eval preflight")
    parser.add_argument(
        "--smoke-paddle",
        action="store_true",
        help="Also construct PaddleOCR (GPU wheel if OCR_USE_GPU=1)",
    )
    args = parser.parse_args()
    _env_defaults()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    failed: list[str] = []

    print("=== Repo invariants (source files) ===\n")
    init_py = REPO / "ocr_pipeline" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    if "from .recognition import HybridOCR" in text or "from .detection import" in text:
        failed.append("ocr_pipeline/__init__.py still eager-imports recognition/detection")
        print("  ocr_pipeline/__init__.py: FAIL (eager imports)")
    else:
        print("  ocr_pipeline/__init__.py: OK (slim)")

    for label, path, needle in REPO_INVARIANTS[1:]:
        ok = path.is_file() and needle in path.read_text(encoding="utf-8")
        print(f"  {label}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(f"missing {label} in {path.name}")

    print("\n=== Pip packages ===\n")
    for pkg in REQUIRED_PACKAGES:
        mod = "cv2" if pkg == "cv2" else ("dotenv" if pkg == "dotenv" else pkg)
        try:
            importlib.import_module(mod)
            print(f"  {pkg}: OK")
        except ImportError as exc:
            failed.append(f"package {pkg}: {exc}")
            print(f"  {pkg}: FAIL ({exc})")

    if os.environ.get("OCR_USE_GPU", "0") == "1":
        print("\n=== GPU pip packages ===\n")
        for pkg in OPTIONAL_GPU:
            try:
                importlib.import_module(pkg)
                print(f"  {pkg}: OK")
            except ImportError as exc:
                failed.append(f"package {pkg}: {exc}")
                print(f"  {pkg}: FAIL ({exc})")

    print("\n=== System ===\n")
    import shutil

    skip_tesseract = os.environ.get("OCR_SKIP_TESSERACT_ENSEMBLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not shutil.which("tesseract"):
        if skip_tesseract:
            print("  tesseract: WARN (binary missing; OK when OCR_SKIP_TESSERACT_ENSEMBLE=1)")
        else:
            failed.append("tesseract binary missing")
            print("  tesseract: FAIL")
    else:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            print("  tesseract: OK")
        except Exception as exc:
            failed.append(f"tesseract: {exc}")
            print(f"  tesseract: FAIL ({exc})")

    print("\n=== Import chain (in-process) ===\n")
    for label, mod in IMPORT_STEPS:
        try:
            importlib.import_module(mod)
            print(f"  {label}: OK")
        except Exception as exc:
            failed.append(f"{label}: {exc}")
            print(f"  {label}: FAIL ({type(exc).__name__}: {exc})")

    print("\n=== Isolated subprocess checks (fresh interpreter) ===\n")

    checks = [
        (
            "notebook smoke imports (no anthropic)",
            """
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE
from ocr_pipeline.recognition.hybrid_ocr import HybridOCR
HybridOCR(use_detection_router=False, use_vision_augmentation=False, use_ensemble_for_accuracy=False)
import sys
assert 'anthropic' not in sys.modules, 'anthropic must not load on OCR path'
print('OK')
""",
        ),
        (
            "recognition package import (no anthropic)",
            """
import ocr_pipeline.recognition as rec
import sys
assert 'anthropic' not in sys.modules
print('OK')
""",
        ),
        (
            "eval_runner OCR entry (no anthropic)",
            """
try:
    from eval_runner import run_ocr_all_splits, warmup_ocr_pipeline, _ocr_use_ensemble
except ImportError:
    from eval_runner import run_ocr_all_splits, warmup_ocr_pipeline
    from ocr_cloud_eval_entry import _ocr_use_ensemble
assert not _ocr_use_ensemble('FUNSD'), 'ensemble must be off when OCR_SKIP_TESSERACT_ENSEMBLE=1'
import sys
assert 'anthropic' not in sys.modules
print('OK')
""",
        ),
        (
            "ocr_eval_config env wiring",
            """
import os
os.environ['OCR_REC_BATCH_NUM'] = '24'
from ocr_pipeline.ocr_eval_config import paddle_rec_batch_num, ocr_skip_tesseract_ensemble
assert paddle_rec_batch_num() == 24
assert ocr_skip_tesseract_ensemble()
print('OK')
""",
        ),
        (
            "compute_ocr_metrics FUNSD/SROIE",
            """
from eval_postprocess_utils import compute_ocr_metrics
m = compute_ocr_metrics('hello world', ['hello', 'world'], 'FUNSD')
assert 'word_recall' in m
m2 = compute_ocr_metrics('total 10', {'total': '10'}, 'SROIE')
assert 'entity_match' in m2
print('OK')
""",
        ),
    ]

    sub_env = os.environ.copy()
    sub_env["PYTHONPATH"] = str(REPO)

    for label, code in checks:
        rc, out = _run_py(code, env=sub_env)
        if rc == 0:
            print(f"  {label}: OK")
        else:
            failed.append(f"{label}: {out[-500:]}")
            print(f"  {label}: FAIL")
            for line in out.splitlines()[-8:]:
                print(f"    {line}")

    print("\n=== In-process OCR path ===\n")
    try:
        if "anthropic" in sys.modules:
            del sys.modules["anthropic"]
        from ocr_pipeline.recognition.hybrid_ocr import HybridOCR

        h = HybridOCR(
            use_detection_router=False,
            use_vision_augmentation=False,
            use_ensemble_for_accuracy=False,
        )
        if "anthropic" in sys.modules:
            failed.append("anthropic loaded after HybridOCR()")
            print("  HybridOCR init: FAIL (anthropic in sys.modules)")
        else:
            print("  HybridOCR init: OK (no anthropic)")
        if h.detection_router is not None:
            failed.append("detection_router should be None when use_detection_router=False")
            print("  detection_router lazy: FAIL")
        else:
            print("  detection_router lazy: OK")
    except Exception as exc:
        failed.append(f"HybridOCR: {exc}")
        print(f"  HybridOCR init: FAIL ({exc})")

    try:
        from eval_runner import warmup_ocr_pipeline

        warmup_ocr_pipeline("FUNSD")
        print("  warmup_ocr_pipeline: OK")
    except Exception as exc:
        failed.append(f"warmup: {exc}")
        print(f"  warmup_ocr_pipeline: FAIL ({exc})")

    if args.smoke_paddle:
        print("\n=== PaddleOCR construct ===\n")
        rc, out = _run_py(
            """
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE, get_or_build_native_paddle_ocr
from ocr_pipeline.ocr_eval_config import paddle_rec_batch_num
assert PADDLEOCR_AVAILABLE
get_or_build_native_paddle_ocr(show_log=False)
print('rec_batch', paddle_rec_batch_num())
print('OK')
""",
            env=sub_env,
        )
        if rc == 0:
            print(f"  Paddle smoke: OK ({out.splitlines()[-2:]})")
        else:
            failed.append(f"Paddle smoke: {out[-400:]}")
            print("  Paddle smoke: FAIL")
            print(out[-600:])

    print("\n=== Notebook cell order reminder ===\n")
    print("  Kaggle: (1) config+env -> (2) pip+apt -> (3) GPU -> (4) clone+audit -> (5) parquet")
    print("          -> (6) Paddle smoke+audit --smoke-paddle -> (7) run_ocr_all_splits")
    print("  Colab:  (1) config -> (2) GPU -> (3) clone -> (4) pip+audit -> (5) smoke -> (6) eval")

    if failed:
        print(f"\nFAILED: {len(failed)} check(s)")
        for i, f in enumerate(failed, 1):
            print(f"  {i}. {f}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
