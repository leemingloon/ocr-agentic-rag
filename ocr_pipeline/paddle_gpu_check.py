"""Verify Paddle is actually using CUDA when OCR_USE_GPU=1."""
from __future__ import annotations

import os
import subprocess
from typing import Any


def ocr_wants_gpu() -> bool:
    return os.environ.get("OCR_USE_GPU", "").strip().lower() in ("1", "true", "yes")


def gpu_mem_used_mb() -> float:
    import shutil

    if not shutil.which("nvidia-smi"):
        return 0.0
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return 0.0
    return float(p.stdout.strip().splitlines()[0])


def paddle_cuda_diagnostic() -> dict:
    out: dict[str, Any] = {
        "ocr_use_gpu_env": ocr_wants_gpu(),
        "paddle_installed": False,
        "paddle_version": None,
        "cuda_compiled": False,
        "cuda_device_count": 0,
        "tensor_place": None,
        "pip_paddle_dist": None,
    }
    try:
        import importlib.metadata as md

        for name in ("paddlepaddle-gpu", "paddlepaddle"):
            try:
                out["pip_paddle_dist"] = f"{name}=={md.version(name)}"
                break
            except md.PackageNotFoundError:
                continue
    except Exception:
        pass
    try:
        import paddle

        out["paddle_installed"] = True
        out["paddle_version"] = paddle.__version__
        out["cuda_compiled"] = bool(paddle.device.is_compiled_with_cuda())
        out["cuda_device_count"] = int(paddle.device.cuda.device_count())
        if out["cuda_compiled"] and out["cuda_device_count"] > 0:
            paddle.device.set_device("gpu")
            t = paddle.to_tensor([1.0])
            out["tensor_place"] = str(t.place)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def assert_paddle_cuda_ready() -> None:
    """Fail fast if OCR_USE_GPU=1 but only CPU paddle is installed."""
    if not ocr_wants_gpu():
        print("[Paddle] OCR_USE_GPU=0 — CPU mode expected")
        return
    d = paddle_cuda_diagnostic()
    print(
        f"[Paddle] pip={d.get('pip_paddle_dist')} version={d.get('paddle_version')} "
        f"cuda_compiled={d.get('cuda_compiled')} devices={d.get('cuda_device_count')} "
        f"tensor_place={d.get('tensor_place')}"
    )
    if d.get("pip_paddle_dist", "").startswith("paddlepaddle==") and "gpu" not in d.get(
        "pip_paddle_dist", ""
    ):
        raise RuntimeError(
            "CPU-only paddlepaddle is installed. Colab/Kaggle need GPU wheel:\n"
            "  pip uninstall -y paddlepaddle paddlepaddle-gpu\n"
            "  pip install paddlepaddle-gpu==2.6.2\n"
            "Then Runtime → Restart session and Run all from the top."
        )
    if not d.get("cuda_compiled"):
        raise RuntimeError(
            "Paddle not built with CUDA (GPU wheel missing). "
            "Uninstall paddlepaddle, install paddlepaddle-gpu, restart runtime."
        )
    if int(d.get("cuda_device_count") or 0) < 1:
        raise RuntimeError("Paddle sees zero CUDA devices — check runtime is T4 GPU.")
    place = str(d.get("tensor_place") or "")
    if "gpu" not in place.lower() and "cuda" not in place.lower():
        raise RuntimeError(f"Paddle probe tensor not on GPU (place={place})")
    print("[Paddle] GPU probe OK — OCR should use VRAM during eval (not only system RAM).")


def paddle_ocr_constructor_attempts(
    *,
    use_gpu: bool,
    use_cls: bool,
    rec_batch: int,
    show_log: bool,
) -> list[dict]:
    """Keyword attempts for PaddleOCR(). Never fall back to PaddleOCR({}) when use_gpu=True."""
    base: dict = {
        "lang": "en",
        "use_angle_cls": use_cls,
        "rec_batch_num": rec_batch,
    }
    attempts: list[dict] = []
    if use_gpu:
        for dev in ("gpu:0", "gpu"):
            attempts.append({**base, "device": dev})
        attempts.append({**base, "use_gpu": True})
        attempts.append({"lang": "en", "use_gpu": True})
    else:
        attempts.append({**base, "use_gpu": False})
        attempts.append({**base, "device": "cpu"})
        attempts.append({"lang": "en", "use_gpu": False})
        attempts.append({"lang": "en"})
    if show_log:
        attempts = [{**kw, "show_log": True} for kw in attempts] + attempts
    return attempts


def _safe_kw_repr(kw: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in kw.items())


def warmup_paddleocr_allocates_gpu(
    paddle_ocr: Any,
    *,
    min_used_mb: float = 400.0,
    min_delta_mb: float = 150.0,
) -> None:
    """Run a tiny OCR forward pass; fail if GPU VRAM does not rise (CPU fallback)."""
    if not ocr_wants_gpu():
        return
    import cv2
    import numpy as np

    before = gpu_mem_used_mb()
    img = np.full((320, 320, 3), 255, dtype=np.uint8)
    result = None
    for kw in ({"cls": False}, {"det": True, "rec": True, "cls": False}, {}):
        try:
            result = paddle_ocr.ocr(img, **kw)
            if result:
                break
        except TypeError:
            try:
                result = paddle_ocr.ocr(img)
                if result:
                    break
            except Exception:
                continue
        except Exception:
            continue
    if result is None and hasattr(paddle_ocr, "predict"):
        try:
            result = list(paddle_ocr.predict(img))
        except Exception:
            pass
    after = gpu_mem_used_mb()
    print(f"[Paddle] VRAM used: before={before:.0f} MB after={after:.0f} MB (delta={after - before:.0f} MB)")
    if after < min_used_mb and (after - before) < min_delta_mb:
        raise RuntimeError(
            f"PaddleOCR warmup did not allocate GPU memory (VRAM {after:.0f} MB). "
            "You are on CPU Paddle — Runtime → Restart session, then Run all from the top "
            "after the deps cell installs paddlepaddle-gpu. Do not skip Restart."
        )
    print(f"[Paddle] GPU warmup OK — VRAM {after:.0f} MB")


def build_paddle_ocr_engine(*, show_log: bool = False) -> Any:
    """Construct PaddleOCR; on GPU path refuse silent CPU fallback."""
    from paddleocr import PaddleOCR

    try:
        from ocr_pipeline.ocr_eval_config import paddle_rec_batch_num, paddle_use_angle_cls
    except ImportError:
        from .ocr_eval_config import paddle_rec_batch_num, paddle_use_angle_cls

    import logging

    logging.getLogger("ppocr").setLevel(logging.ERROR)
    use_gpu = ocr_wants_gpu()
    use_cls = paddle_use_angle_cls()
    rec_batch = paddle_rec_batch_num()

    if use_gpu:
        assert_paddle_cuda_ready()

    last_err: Exception | None = None
    for kw in paddle_ocr_constructor_attempts(
        use_gpu=use_gpu, use_cls=use_cls, rec_batch=rec_batch, show_log=show_log
    ):
        try:
            ocr = PaddleOCR(**kw)
            if use_gpu:
                print(f"[Paddle] PaddleOCR constructed with {_safe_kw_repr(kw)}")
                warmup_paddleocr_allocates_gpu(ocr)
            return ocr
        except (TypeError, ValueError, Exception) as e:
            last_err = e
            continue
    msg = (
        "PaddleOCR failed to initialize on GPU with device=gpu / use_gpu=True. "
        "Restart runtime after installing paddlepaddle-gpu, then Run all."
    )
    if use_gpu:
        raise RuntimeError(f"{msg} Last error: {last_err}") from last_err
    assert last_err is not None
    raise last_err
