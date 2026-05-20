#!/usr/bin/env python3
"""Apply OCR cloud fixes on disk after git clone (GitHub may lag behind local).

Run before any `import ocr_pipeline` in Colab/Kaggle:
  python scripts/apply_ocr_cloud_patches.py [REPO_ROOT]

Idempotent — safe to run every session.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SLIM_OCR_PIPELINE_INIT = '''"""
OCR Pipeline — import submodules directly (no eager HybridOCR / anthropic).
"""
__version__ = "1.0.0"
__all__ = ["__version__"]
'''

LAZY_RECOGNITION_INIT = '''"""
Text Recognition — lazy HybridOCR import (OCR eval does not need vision/anthropic).
"""
from __future__ import annotations

from .tesseract_ocr import TesseractOCR, OCRResult
from .ocr_schema import StandardOCROutput, OCRWord, OCRLine

__all__ = [
    "TesseractOCR",
    "HybridOCR",
    "OCRResult",
    "StandardOCROutput",
    "OCRWord",
    "OCRLine",
]


def __getattr__(name: str):
    if name == "HybridOCR":
        from .hybrid_ocr import HybridOCR

        return HybridOCR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''

LAZY_DETECTION_INIT = '''"""
Text detection — lazy DetectionRouter / PaddleOCRDetector imports.
"""
from __future__ import annotations

from .classical_detector import ClassicalDetector

__all__ = ["ClassicalDetector", "PaddleOCRDetector", "DetectionRouter"]


def __getattr__(name: str):
    if name == "DetectionRouter":
        from .detection_router import DetectionRouter

        return DetectionRouter
    if name == "PaddleOCRDetector":
        try:
            from .paddleocr_detector import PaddleOCRDetector

            return PaddleOCRDetector
        except Exception as e:
            print(f"⚠ PaddleOCR detector unavailable: {e}")

            class _Unavailable:
                mode = "unavailable"

                def __init__(self, *a, **k):
                    pass

                def detect(self, image):
                    return []

            return _Unavailable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''

# Inlined so patch+verify works when GitHub clone lacks ocr_pipeline/compat/ (apply script may be the only new file).
EMBEDDED_PADDLE_LANGCHAIN_SHIM = '''"""
PaddleX (pulled by recent paddleocr wheels) imports legacy LangChain 0.1 modules:
  langchain.docstore.document, langchain.text_splitter

Those paths were removed in LangChain 1.x. Installing real legacy langchain conflicts
with this repo's langgraph stack. Minimal stubs unblock `from paddleocr import PaddleOCR`
for document OCR only (retriever code in paddlex is not used).

Call `install_paddle_langchain_shim()` before importing paddleocr anywhere.
"""


from __future__ import annotations

import sys
import types


def install_paddle_langchain_shim() -> None:
    if sys.modules.get("langchain.docstore.document") and sys.modules.get("langchain.text_splitter"):
        return

    doc_mod = types.ModuleType("langchain.docstore.document")

    class Document:
        __slots__ = ("page_content", "metadata")

        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata if metadata is not None else {}

    doc_mod.Document = Document
    sys.modules.setdefault("langchain.docstore.document", doc_mod)

    ds_pkg = types.ModuleType("langchain.docstore")
    sys.modules.setdefault("langchain.docstore", ds_pkg)

    ts_mod = types.ModuleType("langchain.text_splitter")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def split_text(self, text: str) -> list[str]:
            return [text] if text else []

    ts_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules.setdefault("langchain.text_splitter", ts_mod)
'''

EMBEDDED_OCR_EVAL_CONFIG = '''"""OCR eval runtime flags (env). Used by eval_runner and Kaggle notebook."""

from __future__ import annotations

import os


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def ocr_use_gpu() -> bool:
    return _truthy("OCR_USE_GPU")


def ocr_skip_tesseract_ensemble() -> bool:
    return _truthy("OCR_SKIP_TESSERACT_ENSEMBLE")


def ocr_fast_mode() -> bool:
    return _truthy("OCR_FAST") or (ocr_skip_tesseract_ensemble() and ocr_use_gpu())


def ocr_cpu_workers() -> int:
    if ocr_use_gpu():
        return 0
    try:
        return max(0, int(os.environ.get("OCR_CPU_WORKERS", "0")))
    except ValueError:
        return 0


def ocr_prefetch_enabled() -> bool:
    return _truthy("OCR_PREFETCH", "1")


def paddle_use_angle_cls() -> bool:
    if ocr_fast_mode():
        return _truthy("OCR_PADDLE_USE_CLS", "0")
    return _truthy("OCR_PADDLE_USE_CLS", "1")


def paddle_min_side() -> int:
    if ocr_fast_mode():
        try:
            return max(480, int(os.environ.get("OCR_PADDLE_MIN_SIDE", "720")))
        except ValueError:
            return 720
    try:
        return max(480, int(os.environ.get("OCR_PADDLE_MIN_SIDE", "900")))
    except ValueError:
        return 900


def paddle_rec_batch_num() -> int:
    raw = os.environ.get("OCR_REC_BATCH_NUM", "").strip()
    if raw:
        try:
            return max(1, min(96, int(raw)))
        except ValueError:
            pass
    if ocr_use_gpu() and ocr_fast_mode():
        return 16
    if ocr_use_gpu():
        return 8
    return 6
'''


def _patch_file(path: Path, content: str, reason: str, *, repo: Path) -> bool:
    if path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    try:
        rel = path.relative_to(repo)
    except ValueError:
        rel = path
    print(f"  patched {rel} ({reason})")
    return True


def _patch_vision_ocr(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^from anthropic import Anthropic\s*$", text, re.MULTILINE):
        return False
    text = re.sub(r"^from anthropic import Anthropic\s*\n", "", text, count=1, flags=re.MULTILINE)
    if "from anthropic import Anthropic" in text.split("def __init__")[0]:
        # Still at module level in another form
        lines = [ln for ln in text.splitlines() if ln.strip() != "from anthropic import Anthropic"]
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"  patched {path.name} (removed module-level anthropic import)")
    return True


def _needs_slim_init(path: Path) -> bool:
    if not path.is_file():
        return True
    t = path.read_text(encoding="utf-8")
    return "from .recognition import" in t or "from .detection import" in t or "HybridOCR" in t and "__getattr__" not in t


def _write_if_missing(path: Path, content: str, *, repo: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    try:
        rel = path.relative_to(repo)
    except ValueError:
        rel = path
    print(f"  bundled {rel}")
    return True


def ensure_cloud_bundle(repo: Path) -> int:
    """Write modules that may be missing on GitHub clone (compat shim, ocr_eval_config)."""
    n = 0
    shim_path = repo / "ocr_pipeline" / "compat" / "paddle_langchain_shim.py"
    script_root = Path(__file__).resolve().parent.parent
    local_shim = script_root / "ocr_pipeline" / "compat" / "paddle_langchain_shim.py"
    if local_shim.is_file():
        shim_src = local_shim.read_text(encoding="utf-8")
    else:
        shim_src = EMBEDDED_PADDLE_LANGCHAIN_SHIM
    if _write_if_missing(repo / "ocr_pipeline" / "compat" / "__init__.py", '"""Compatibility shims."""\n', repo=repo):
        n += 1
    if _write_if_missing(shim_path, shim_src, repo=repo):
        n += 1
    cfg_path = repo / "ocr_pipeline" / "ocr_eval_config.py"
    if not cfg_path.is_file():
        local_cfg = script_root / "ocr_pipeline" / "ocr_eval_config.py"
        cfg_src = local_cfg.read_text(encoding="utf-8") if local_cfg.is_file() else EMBEDDED_OCR_EVAL_CONFIG
        if _write_if_missing(cfg_path, cfg_src, repo=repo):
            n += 1
    entry_path = repo / "ocr_cloud_eval_entry.py"
    local_entry = script_root / "ocr_cloud_eval_entry.py"
    if local_entry.is_file() and _write_if_missing(entry_path, local_entry.read_text(encoding="utf-8"), repo=repo):
        n += 1
    er = repo / "eval_runner.py"
    if er.is_file():
        text = er.read_text(encoding="utf-8")
        stub = (
            "\n# Cloud notebook patch: GitHub eval_runner may lack run_ocr_all_splits\n"
            "try:\n"
            "    run_ocr_all_splits  # noqa: B018\n"
            "    _ocr_use_ensemble  # noqa: B018\n"
            "except NameError:\n"
            "    from ocr_cloud_eval_entry import (  # noqa: F401\n"
            "        _ocr_use_ensemble,\n"
            "        run_ocr_all_splits,\n"
            "        warmup_ocr_pipeline,\n"
            "    )\n"
        )
        old_line = "from ocr_cloud_eval_entry import run_ocr_all_splits, warmup_ocr_pipeline"
        if old_line in text and "_ocr_use_ensemble" not in text:
            text = text.replace(
                old_line + "  # noqa: F401",
                (
                    "from ocr_cloud_eval_entry import (  # noqa: F401\n"
                    "        _ocr_use_ensemble,\n"
                    "        run_ocr_all_splits,\n"
                    "        warmup_ocr_pipeline,\n"
                    "    )"
                ),
            )
            if "_ocr_use_ensemble  # noqa: B018" not in text:
                text = text.replace(
                    "    run_ocr_all_splits  # noqa: B018\nexcept NameError:",
                    "    run_ocr_all_splits  # noqa: B018\n    _ocr_use_ensemble  # noqa: B018\nexcept NameError:",
                    1,
                )
            er.write_text(text, encoding="utf-8")
            print("  upgraded eval_runner.py stub (_ocr_use_ensemble)")
            n += 1
        elif "def run_ocr_all_splits" not in text and stub.strip() not in text:
            er.write_text(text.rstrip() + stub + "\n", encoding="utf-8")
            print("  patched eval_runner.py (ocr_cloud_eval_entry)")
            n += 1
    api_local = script_root / "ocr_pipeline" / "detection" / "paddle_cloud_api.py"
    api_path = repo / "ocr_pipeline" / "detection" / "paddle_cloud_api.py"
    if api_local.is_file() and _write_if_missing(api_path, api_local.read_text(encoding="utf-8"), repo=repo):
        n += 1
    det = repo / "ocr_pipeline" / "detection" / "paddleocr_detector.py"
    if det.is_file():
        text = det.read_text(encoding="utf-8")
        pstub = (
            "\n# Cloud notebook patch: GitHub paddleocr_detector may lack full-page OCR API\n"
            "try:\n"
            "    get_or_build_native_paddle_ocr  # noqa: B018\n"
            "except NameError:\n"
            "    from ocr_pipeline.detection.paddle_cloud_api import (  # noqa: F401\n"
            "        build_native_paddle_ocr,\n"
            "        get_or_build_native_paddle_ocr,\n"
            "        run_paddle_full_ocr,\n"
            "    )\n"
        )
        if "def get_or_build_native_paddle_ocr" not in text and pstub.strip() not in text:
            det.write_text(text.rstrip() + pstub + "\n", encoding="utf-8")
            print("  patched paddleocr_detector.py (paddle_cloud_api)")
            n += 1
    return n


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    if not (repo / "eval_runner.py").is_file():
        print(f"ERROR: not a repo root: {repo}")
        return 1

    print(f"apply_ocr_cloud_patches: {repo}")
    n = 0

    init_py = repo / "ocr_pipeline" / "__init__.py"
    if _needs_slim_init(init_py):
        init_py.parent.mkdir(parents=True, exist_ok=True)
        if _patch_file(init_py, SLIM_OCR_PIPELINE_INIT, "slim __init__", repo=repo):
            n += 1

    rec_init = repo / "ocr_pipeline" / "recognition" / "__init__.py"
    t = rec_init.read_text(encoding="utf-8") if rec_init.is_file() else ""
    if "from .hybrid_ocr import HybridOCR" in t and "__getattr__" not in t:
        if _patch_file(rec_init, LAZY_RECOGNITION_INIT, "lazy recognition __init__", repo=repo):
            n += 1

    det_init = repo / "ocr_pipeline" / "detection" / "__init__.py"
    t = det_init.read_text(encoding="utf-8") if det_init.is_file() else ""
    if "from .detection_router import DetectionRouter" in t and "__getattr__" not in t:
        if _patch_file(det_init, LAZY_DETECTION_INIT, "lazy detection __init__", repo=repo):
            n += 1

    vo = repo / "ocr_pipeline" / "recognition" / "vision_ocr.py"
    if _patch_vision_ocr(vo):
        n += 1

    n += ensure_cloud_bundle(repo)

    code = """
from pathlib import Path
er = Path("eval_runner.py").read_text(encoding="utf-8")
assert "def run_ocr_all_splits" in er or "ocr_cloud_eval_entry" in er, "eval_runner missing OCR API patch"
assert Path("ocr_cloud_eval_entry.py").is_file(), "ocr_cloud_eval_entry.py missing"
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE
from ocr_cloud_eval_entry import run_ocr_all_splits, warmup_ocr_pipeline
try:
    from ocr_pipeline.detection.paddleocr_detector import get_or_build_native_paddle_ocr
except ImportError:
    from ocr_pipeline.detection.paddle_cloud_api import get_or_build_native_paddle_ocr
import sys
assert 'anthropic' not in sys.modules
assert callable(run_ocr_all_splits) and callable(warmup_ocr_pipeline)
assert callable(get_or_build_native_paddle_ocr)
print('verify_ok', PADDLEOCR_AVAILABLE)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    if proc.returncode != 0:
        print("VERIFY FAILED after patch:")
        print(proc.stdout)
        print(proc.stderr)
        return 1

    print(proc.stdout.strip())
    print(f"Done ({n} file(s) patched)." if n else "Done (already up to date).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
