"""Bundled files for Colab when GitHub clone is behind local. Imported by the Colab notebook."""
from __future__ import annotations

import shutil
from pathlib import Path

PADDLE_LANGCHAIN_SHIM = '''"""
PaddleX imports legacy langchain.docstore.document / langchain.text_splitter.
Minimal stubs unblock paddleocr without installing legacy langchain.
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
    sys.modules.setdefault("langchain.docstore", types.ModuleType("langchain.docstore"))
    ts_mod = types.ModuleType("langchain.text_splitter")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass
        def split_text(self, text: str) -> list[str]:
            return [text] if text else []
    ts_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules.setdefault("langchain.text_splitter", ts_mod)
'''

OCR_EVAL_CONFIG = '''"""OCR eval runtime flags (env)."""
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

OCR_CLOUD_EVAL_ENTRY = r'''"""OCR eval entrypoints for cloud notebooks when GitHub eval_runner.py lags behind local."""
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
        ocr_skip_tesseract_ensemble = lambda: os.environ.get(
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
'''


def _ocr_cloud_eval_entry_source() -> str:
    try:
        here = Path(__file__).resolve()
        local_entry = here.parents[2] / "ocr_cloud_eval_entry.py"
        if local_entry.is_file():
            return local_entry.read_text(encoding="utf-8")
    except NameError:
        pass
    return OCR_CLOUD_EVAL_ENTRY


def _eval_runner_stub() -> str:
    return """
# Cloud notebook patch: GitHub eval_runner may lack OCR eval entrypoints
try:
    run_ocr_all_splits  # noqa: B018
    _ocr_use_ensemble  # noqa: B018
except NameError:
    from ocr_cloud_eval_entry import (  # noqa: F401
        _ocr_use_ensemble,
        run_ocr_all_splits,
        warmup_ocr_pipeline,
    )
"""


def _paddleocr_detector_stub() -> str:
    return """
# Cloud notebook patch: GitHub paddleocr_detector may lack full-page OCR API
try:
    get_or_build_native_paddle_ocr  # noqa: B018
except NameError:
    from ocr_pipeline.detection.paddle_cloud_api import (  # noqa: F401
        build_native_paddle_ocr,
        get_or_build_native_paddle_ocr,
        run_paddle_full_ocr,
    )
"""


PADDLE_CLOUD_API_FALLBACK = """\"\"\"
Paddle full-page OCR API for cloud notebooks when GitHub paddleocr_detector.py lags behind local.

Used by eval_runner warmup and hybrid_ocr.run_paddle_full_ocr imports.
\"\"\"
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

try:
    from ..compat.paddle_langchain_shim import install_paddle_langchain_shim

    install_paddle_langchain_shim()
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None  # type: ignore[misc, assignment]
    _PADDLE_IMPORT_ERR = e
else:
    _PADDLE_IMPORT_ERR = None


def _paddle_use_gpu() -> bool:
    return os.environ.get("OCR_USE_GPU", "").strip().lower() in ("1", "true", "yes")


def build_native_paddle_ocr(*, show_log: bool = False):
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        raise RuntimeError(f"PaddleOCR is not available: {_PADDLE_IMPORT_ERR}")
    import logging

    try:
        from ocr_pipeline.ocr_eval_config import paddle_rec_batch_num, paddle_use_angle_cls
    except ImportError:
        from ..ocr_eval_config import paddle_rec_batch_num, paddle_use_angle_cls

    logging.getLogger("ppocr").setLevel(logging.ERROR)
    use_gpu = _paddle_use_gpu()
    use_cls = paddle_use_angle_cls()
    rec_batch = paddle_rec_batch_num()
    base_kw: dict = {
        "lang": "en",
        "use_gpu": use_gpu,
        "use_angle_cls": use_cls,
        "rec_batch_num": rec_batch,
    }
    attempts: list[dict] = []
    if show_log:
        attempts.append({**base_kw, "show_log": True})
    attempts.extend([dict(base_kw), {"lang": "en", "use_gpu": use_gpu}, {"lang": "en"}, {}])
    last_err: Exception | None = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except (TypeError, ValueError, Exception) as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


_CACHED_NATIVE_FULL_PADDLE: Any = None


def _paddle_lines_reading_order(ocr_lines: list | None) -> list:
    if not ocr_lines:
        return []
    keyed: list[tuple[int, float, float, object]] = []
    for line in ocr_lines:
        if line is None or len(line) < 2:
            continue
        try:
            pts = np.asarray(line[0], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            y_ctr = float(pts[:, 1].mean())
            x_left = float(pts[:, 0].min())
        except Exception:
            continue
        row_band = int(round(y_ctr / 15.0))
        keyed.append((row_band, x_left, y_ctr, line))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


def _parse_paddle_result(result: Any) -> tuple[str, float, int]:
    lines_raw: list = []
    if result is None:
        return "", 0.0, 0
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], list):
            lines_raw = result[0] or []
        elif result and isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
            if isinstance(result[0][0], (list, tuple, np.ndarray)):
                lines_raw = result
            else:
                lines_raw = result[0] if isinstance(result[0], list) else result
    text_parts: list[str] = []
    conf_sum, conf_n = 0.0, 0
    for line in _paddle_lines_reading_order(lines_raw):
        if not line or len(line) < 2:
            continue
        rec = line[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
            t = str(rec[0]).strip()
            if t:
                text_parts.append(t)
            if len(rec) >= 2:
                try:
                    conf_sum += float(rec[1])
                    conf_n += 1
                except (TypeError, ValueError):
                    pass
    text = "\\n".join(text_parts)
    confidence = (conf_sum / conf_n * 100.0) if conf_n else 85.0
    return text, confidence, len(text_parts)


def get_or_build_native_paddle_ocr(*, show_log: bool = False) -> Any:
    global _CACHED_NATIVE_FULL_PADDLE
    if _CACHED_NATIVE_FULL_PADDLE is not None:
        return _CACHED_NATIVE_FULL_PADDLE
    _CACHED_NATIVE_FULL_PADDLE = build_native_paddle_ocr(show_log=show_log)
    return _CACHED_NATIVE_FULL_PADDLE


def run_paddle_full_ocr(
    image: np.ndarray,
    paddle_ocr: Any | None = None,
) -> tuple[str, float, int]:
    if paddle_ocr is None:
        paddle_ocr = get_or_build_native_paddle_ocr(show_log=False)
    try:
        from ocr_pipeline.ocr_eval_config import paddle_min_side, paddle_use_angle_cls
    except ImportError:
        from ..ocr_eval_config import paddle_min_side, paddle_use_angle_cls

    work = image
    if len(work.shape) == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    h, w = work.shape[:2]
    target_min = paddle_min_side()
    min_side = min(h, w)
    if min_side < target_min:
        scale = target_min / min_side
        work = cv2.resize(
            work,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    use_cls = paddle_use_angle_cls()
    result = None
    ocr_kw_attempts: list[dict] = []
    if use_cls:
        ocr_kw_attempts.append({"cls": True})
    ocr_kw_attempts.extend([{"det": True, "rec": True, "cls": use_cls}, {}])
    for kw in ocr_kw_attempts:
        try:
            result = paddle_ocr.ocr(work, **kw)
            if result:
                break
        except TypeError:
            try:
                result = paddle_ocr.ocr(work)
                if result:
                    break
            except Exception:
                continue
        except Exception:
            continue
    if result is None:
        legacy_calls = []
        if use_cls:
            legacy_calls.append(lambda: paddle_ocr.ocr(work, cls=True))
        legacy_calls.extend(
            [
                lambda: paddle_ocr.ocr(work, det=True, rec=True, cls=True),
                lambda: paddle_ocr.ocr(work),
            ]
        )
        for call in legacy_calls:
            try:
                result = call()
                if result:
                    break
            except TypeError:
                continue
            except Exception:
                continue
    if result is None and hasattr(paddle_ocr, "predict"):
        try:
            result = list(paddle_ocr.predict(work))
        except Exception:
            result = None
    return _parse_paddle_result(result)
"""

def _paddle_cloud_api_source(root: Path | None = None) -> str:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root) / "ocr_pipeline" / "detection" / "paddle_cloud_api.py")
    try:
        here = Path(__file__).resolve()
        candidates.append(here.parents[2] / "ocr_pipeline" / "detection" / "paddle_cloud_api.py")
    except NameError:
        pass
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return PADDLE_CLOUD_API_FALLBACK


def ensure_paddleocr_detector_cloud_api(root: Path) -> None:
    root = Path(root).resolve()
    api_path = root / "ocr_pipeline" / "detection" / "paddle_cloud_api.py"
    content = _paddle_cloud_api_source(root)
    if not api_path.is_file() or api_path.read_text(encoding="utf-8") != content:
        api_path.parent.mkdir(parents=True, exist_ok=True)
        api_path.write_text(content, encoding="utf-8")
        print("bundled ocr_pipeline/detection/paddle_cloud_api.py (notebook)")
    det = root / "ocr_pipeline" / "detection" / "paddleocr_detector.py"
    if not det.is_file():
        raise RuntimeError(f"paddleocr_detector.py not found under {root}")
    text = det.read_text(encoding="utf-8")
    if "def get_or_build_native_paddle_ocr" in text:
        return
    stub = _paddleocr_detector_stub().strip()
    if stub not in text:
        det.write_text(text.rstrip() + "\n" + stub + "\n", encoding="utf-8")
        print("patched paddleocr_detector.py (import paddle_cloud_api)")


def ensure_eval_runner_ocr_api(root: Path) -> None:
    root = Path(root).resolve()
    entry_src = root / "ocr_cloud_eval_entry.py"
    content = _ocr_cloud_eval_entry_source()
    if not entry_src.is_file() or entry_src.read_text(encoding="utf-8") != content:
        entry_src.write_text(content, encoding="utf-8")
        print("bundled ocr_cloud_eval_entry.py (notebook)")
    er = root / "eval_runner.py"
    if not er.is_file():
        raise RuntimeError(f"eval_runner.py not found under {root}")
    text = er.read_text(encoding="utf-8")
    if "def run_ocr_all_splits" in text:
        return
    stub = _eval_runner_stub().strip()
    old_stub = (
        "from ocr_cloud_eval_entry import run_ocr_all_splits, warmup_ocr_pipeline  # noqa: F401"
    )
    if stub not in text:
        if old_stub in text:
            text = text.replace(old_stub, (
                "from ocr_cloud_eval_entry import (  # noqa: F401\n"
                "        _ocr_use_ensemble,\n"
                "        run_ocr_all_splits,\n"
                "        warmup_ocr_pipeline,\n"
                "    )"
            ))
            if "_ocr_use_ensemble  # noqa: B018" not in text:
                text = text.replace(
                    "    run_ocr_all_splits  # noqa: B018\nexcept NameError:",
                    "    run_ocr_all_splits  # noqa: B018\n    _ocr_use_ensemble  # noqa: B018\nexcept NameError:",
                    1,
                )
            er.write_text(text, encoding="utf-8")
            print("patched eval_runner.py (upgraded ocr_cloud_eval_entry imports)")
        else:
            er.write_text(text.rstrip() + "\n" + stub + "\n", encoding="utf-8")
            print("patched eval_runner.py (import ocr_cloud_eval_entry)")


def _bundle_repo_file(root: Path, rel: str) -> None:
    """Copy a repo file into the clone when GitHub is behind local."""
    root = Path(root).resolve()
    try:
        src = Path(__file__).resolve().parents[2] / rel
    except NameError:
        return
    if not src.is_file():
        return
    dest = root / rel
    if not dest.is_file() or dest.read_text(encoding="utf-8") != src.read_text(encoding="utf-8"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"bundled {rel} (notebook)")


def write_cloud_bundle(root: Path) -> None:
    root = Path(root).resolve()
    compat = root / "ocr_pipeline" / "compat"
    shim = compat / "paddle_langchain_shim.py"
    if not shim.is_file():
        compat.mkdir(parents=True, exist_ok=True)
        (compat / "__init__.py").write_text('"""Compatibility shims."""\n', encoding="utf-8")
        shim.write_text(PADDLE_LANGCHAIN_SHIM, encoding="utf-8")
        print("bundled ocr_pipeline/compat/paddle_langchain_shim.py (notebook)")
    cfg = root / "ocr_pipeline" / "ocr_eval_config.py"
    if not cfg.is_file():
        cfg.write_text(OCR_EVAL_CONFIG, encoding="utf-8")
        print("bundled ocr_pipeline/ocr_eval_config.py (notebook)")
    ensure_eval_runner_ocr_api(root)
    ensure_paddleocr_detector_cloud_api(root)
    ensure_cloud_audit_script(root)
    _bundle_repo_file(root, "ocr_pipeline/paddle_gpu_check.py")
    _bundle_repo_file(root, "ocr_pipeline/detection/paddle_cloud_api.py")
    _bundle_repo_file(root, "ocr_pipeline/detection/paddleocr_detector.py")
    _ensure_script = root / "scripts" / "ensure_ocr_parquet_from_hf.py"
    try:
        src = Path(__file__).resolve().parents[2] / "scripts" / "ensure_ocr_parquet_from_hf.py"
        if src.is_file() and (not _ensure_script.is_file() or _ensure_script.read_text() != src.read_text()):
            _ensure_script.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, _ensure_script)
    except NameError:
        pass


def ensure_cloud_audit_script(root: Path) -> None:
    """GitHub clone may lack scripts/audit_ocr_eval_cloud.py (Post-Paddle cell)."""
    root = Path(root).resolve()
    dest = root / "scripts" / "audit_ocr_eval_cloud.py"
    try:
        here = Path(__file__).resolve()
        src = here.parents[2] / "scripts" / "audit_ocr_eval_cloud.py"
        if src.is_file():
            content = src.read_text(encoding="utf-8")
        else:
            return
    except NameError:
        return
    if not dest.is_file() or dest.read_text(encoding="utf-8") != content:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print("bundled scripts/audit_ocr_eval_cloud.py (notebook)")
