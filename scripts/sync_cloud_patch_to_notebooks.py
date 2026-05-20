#!/usr/bin/env python3
"""Keep Colab + Kaggle clone cells in sync (embedded compat bundle)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "notebooks" / "colab" / "embedded_cloud_bundle.py"
B64 = base64.b64encode(BUNDLE.read_bytes()).decode()

PATCH_BLOCK = f'''# Patch stale GitHub clone (slim __init__, compat shim, no anthropic on import path)
import re as _re
import base64 as _b64

# Bundled inside this notebook — works when GitHub clone lacks ocr_pipeline/compat/
_EMB_BUNDLE_B64 = {B64!r}

def _write_notebook_cloud_bundle(root: Path) -> None:
    ns: dict = {{}}
    exec(_b64.b64decode(_EMB_BUNDLE_B64), ns)
    ns["write_cloud_bundle"](root)


def _bundle_compat_if_missing(root: Path) -> None:
    shim = root / "ocr_pipeline" / "compat" / "paddle_langchain_shim.py"
    if shim.is_file():
        return
    for _rel in (
        "notebooks/colab/embedded_cloud_bundle.py",
        "scripts/apply_ocr_cloud_patches.py",
    ):
        emb = root / _rel
        if not emb.is_file():
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location("_cloud_bundle", emb)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        if hasattr(mod, "write_cloud_bundle"):
            mod.write_cloud_bundle(root)
        elif hasattr(mod, "ensure_cloud_bundle"):
            mod.ensure_cloud_bundle(root)
        if shim.is_file():
            return
    _write_notebook_cloud_bundle(root)
    if not shim.is_file():
        raise RuntimeError("Failed to write ocr_pipeline.compat (notebook bundle)")


def _apply_cloud_patches(root: Path) -> None:
    ps = root / "scripts" / "apply_ocr_cloud_patches.py"
    if ps.is_file():
        pr = subprocess.run([sys.executable, str(ps), str(root)], cwd=str(root))
        if pr.returncode == 0:
            return
        print("apply_ocr_cloud_patches failed; using inline fallback")
    init_py = root / "ocr_pipeline" / "__init__.py"
    if init_py.is_file():
        t = init_py.read_text(encoding="utf-8")
        if "from .recognition import" in t or ("HybridOCR" in t and "__getattr__" not in t):
            init_py.write_text(
                '"""Slim init for OCR eval."""\\n'
                '__version__ = "1.0.0"\\n__all__ = ["__version__"]\\n',
                encoding="utf-8",
            )
            print("patched ocr_pipeline/__init__.py")
    vo = root / "ocr_pipeline" / "recognition" / "vision_ocr.py"
    if vo.is_file():
        t = vo.read_text(encoding="utf-8")
        if _re.search(r"^from anthropic import Anthropic", t, _re.MULTILINE):
            vo.write_text(
                _re.sub(r"^from anthropic import Anthropic\\s*\\n", "", t, count=1, flags=_re.M),
                encoding="utf-8",
            )
            print("patched vision_ocr.py")
    _bundle_compat_if_missing(root)
    verify = """
from pathlib import Path
assert Path("ocr_cloud_eval_entry.py").is_file()
_er = Path("eval_runner.py").read_text(encoding="utf-8")
assert "def run_ocr_all_splits" in _er or "ocr_cloud_eval_entry" in _er
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE
from ocr_cloud_eval_entry import _ocr_use_ensemble, run_ocr_all_splits, warmup_ocr_pipeline
try:
    from ocr_pipeline.detection.paddleocr_detector import get_or_build_native_paddle_ocr
except ImportError:
    from ocr_pipeline.detection.paddle_cloud_api import get_or_build_native_paddle_ocr
import sys
assert "anthropic" not in sys.modules
assert callable(run_ocr_all_splits)
assert callable(get_or_build_native_paddle_ocr)
print("verify_ok", PADDLEOCR_AVAILABLE)
"""
    p = subprocess.run(
        [sys.executable, "-c", verify],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={{**os.environ, "PYTHONPATH": str(root)}},
    )
    if p.returncode != 0:
        print(p.stdout, p.stderr)
        raise RuntimeError("Patch verify failed")

_apply_cloud_patches(REPO_ROOT)
for _k in list(sys.modules):
    if _k == "ocr_pipeline" or _k.startswith("ocr_pipeline."):
        del sys.modules[_k]
print("Cloud patches OK.")
'''

GATE_PATCH_BLOCK = '''
def _cloud_ocr_gate(root: Path) -> None:
    root = Path(root).resolve()
    if not (root / "eval_runner.py").is_file():
        raise FileNotFoundError("Run clone cell first.")
    # Re-ensure compat (apply script on GitHub may be old)
    if "_bundle_compat_if_missing" in globals():
        _bundle_compat_if_missing(root)
    else:
        _write_notebook_cloud_bundle(root)
    _patch_script = root / "scripts" / "apply_ocr_cloud_patches.py"
    if _patch_script.is_file():
        _pr = subprocess.run([sys.executable, str(_patch_script), str(root)], cwd=str(root))
        if _pr.returncode != 0:
            print("apply_ocr_cloud_patches returned", _pr.returncode, "(continuing if compat exists)")
    for _k in list(sys.modules):
        if _k == "ocr_pipeline" or _k.startswith("ocr_pipeline."):
            del sys.modules[_k]
    _verify = """
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE
from pathlib import Path
assert Path("ocr_cloud_eval_entry.py").is_file()
_er = Path("eval_runner.py").read_text(encoding="utf-8")
assert "def run_ocr_all_splits" in _er or "ocr_cloud_eval_entry" in _er
try:
    from eval_runner import run_ocr_all_splits, _ocr_use_ensemble
except ImportError:
    from ocr_cloud_eval_entry import run_ocr_all_splits, _ocr_use_ensemble
try:
    from ocr_pipeline.detection.paddleocr_detector import get_or_build_native_paddle_ocr
except ImportError:
    from ocr_pipeline.detection.paddle_cloud_api import get_or_build_native_paddle_ocr
import sys
assert "anthropic" not in sys.modules
assert callable(run_ocr_all_splits)
assert callable(_ocr_use_ensemble)
assert callable(get_or_build_native_paddle_ocr)
print("GATE_VERIFY_OK", PADDLEOCR_AVAILABLE)
"""
    _vp = subprocess.run(
        [sys.executable, "-c", _verify],
        cwd=str(root),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    if _vp.returncode != 0:
        print(_vp.stdout, _vp.stderr)
        raise RuntimeError("GATE FAILED — fix before Paddle (anthropic/import path)")
    _audit = root / "scripts" / "audit_ocr_eval_cloud.py"
    if _audit.is_file():
        _ar = subprocess.run([sys.executable, str(_audit)], cwd=str(root))
        if _ar.returncode != 0:
            raise RuntimeError("audit_ocr_eval_cloud failed")
    print("=" * 60)
    print("GATE PASSED — safe to load Paddle / run_ocr_all_splits")
    print(_vp.stdout.strip())
    print("=" * 60)


_cloud_ocr_gate(REPO_ROOT)
'''

PADDLE_SMOKE_BLOCK = '''
# HARD GATE again before any ocr_pipeline import in this kernel
try:
    _cloud_ocr_gate(REPO_ROOT)
except NameError:
    raise RuntimeError("Run the GATE cell above first (must print GATE PASSED)")

from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
from ocr_pipeline.detection.paddleocr_detector import PADDLEOCR_AVAILABLE
try:
    from ocr_pipeline.detection.paddleocr_detector import get_or_build_native_paddle_ocr
except ImportError:
    from ocr_pipeline.detection.paddle_cloud_api import get_or_build_native_paddle_ocr
from ocr_pipeline.ocr_eval_config import paddle_rec_batch_num, paddle_min_side, paddle_use_angle_cls

print(f"PADDLEOCR_AVAILABLE={PADDLEOCR_AVAILABLE}")
print(f"rec_batch_num={paddle_rec_batch_num()} min_side={paddle_min_side()} angle_cls={paddle_use_angle_cls()}")
if PADDLEOCR_AVAILABLE:
    _ = get_or_build_native_paddle_ocr(show_log=False)
    print("PaddleOCR ready.")
show_gpu_memory()
_post_env = {
    **os.environ,
    "OCR_USE_GPU": "1",
    "OCR_SKIP_TESSERACT_ENSEMBLE": os.environ.get("OCR_SKIP_TESSERACT_ENSEMBLE", "1"),
    "OCR_FAST": os.environ.get("OCR_FAST", "1"),
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}
_POST_PADDLE_INLINE = """
from ocr_pipeline.compat.paddle_langchain_shim import install_paddle_langchain_shim
install_paddle_langchain_shim()
try:
    from eval_runner import run_ocr_all_splits, warmup_ocr_pipeline, _ocr_use_ensemble
except ImportError:
    from eval_runner import run_ocr_all_splits, warmup_ocr_pipeline
    from ocr_cloud_eval_entry import _ocr_use_ensemble
from ocr_pipeline.recognition.hybrid_ocr import HybridOCR
HybridOCR(use_detection_router=False, use_vision_augmentation=False, use_ensemble_for_accuracy=False)
import sys
assert "anthropic" not in sys.modules
assert not _ocr_use_ensemble("FUNSD")
assert callable(run_ocr_all_splits)
print("POST_PADDLE_VERIFY_OK")
"""
_audit = REPO_ROOT / "scripts" / "audit_ocr_eval_cloud.py"
if _audit.is_file():
    _r = subprocess.run(
        [sys.executable, str(_audit), "--smoke-paddle"],
        cwd=str(REPO_ROOT),
        env=_post_env,
        capture_output=True,
        text=True,
        check=False,
    )
else:
    print("Note: scripts/audit_ocr_eval_cloud.py not on GitHub clone — inline Post-Paddle verify")
    _r = subprocess.run(
        [sys.executable, "-c", _POST_PADDLE_INLINE],
        cwd=str(REPO_ROOT),
        env={**_post_env, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
if _r.stdout:
    print(_r.stdout)
if _r.stderr:
    print(_r.stderr, file=sys.stderr)
if _r.returncode != 0:
    raise RuntimeError(
        "Post-Paddle audit failed — do not start run_ocr_all_splits. "
        "See output above (re-run clone cell; need ocr_cloud_eval_entry + eval_runner stub)."
    )
print("Post-Paddle audit OK.")
'''


_CLONE_HEAD_MARKERS = (
    'print(f"Platform: {CLOUD_PLATFORM}")',
    'print(f"REPO_ROOT: {REPO_ROOT.resolve()}")',
    'print("REPO_ROOT:", REPO_ROOT.resolve())',
    'print(f"REPO_ROOT: {REPO_ROOT}")',
)


def _set_cell_source(cell: dict, text: str) -> None:
    import re as _re

    text = text.replace("\r\n", "\n").strip() + "\n"
    text = _re.sub(r"\n{3,}", "\n\n", text)
    cell["source"] = text


def _inject(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "_apply_cloud_patches" not in src:
            continue
        keep = next((m for m in _CLONE_HEAD_MARKERS if m in src), None)
        if not keep:
            continue
        head = src.split(keep)[0] + keep + "\n"
        _set_cell_source(cell, head + PATCH_BLOCK.strip())
        nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated {nb_path.name}")
        return
    raise SystemExit(f"patch cell not found in {nb_path}")


_GATE_HEADER = (
    '# ========== MANDATORY GATE — must print "GATE PASSED" before GPU eval =========='
)


def _inject_gate(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "MANDATORY GATE" not in src or "_cloud_ocr_gate" not in src:
            continue
        _set_cell_source(cell, f"{_GATE_HEADER}\n\n{GATE_PATCH_BLOCK.strip()}")
        nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated GATE in {nb_path.name}")
        return
    raise SystemExit(f"GATE cell not found in {nb_path.name}")


def _inject_gate_colab() -> None:
    _inject_gate(REPO / "notebooks" / "colab" / "ocr_funsd_sroie_eval_colab.ipynb")


_PADDLE_HEADER = "# Paddle load + GPU VRAM check (must be >> 0.1 GB before full eval)"


def _inject_paddle_smoke(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "get_or_build_native_paddle_ocr" not in src or "Post-Paddle audit" not in src:
            continue
        body = (
            f"{_PADDLE_HEADER}\n\n{PADDLE_SMOKE_BLOCK.strip()}\n\n"
            "_assert_paddle_cuda_ready(min_vram_mb=500)\n"
        )
        _set_cell_source(cell, body)
        nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated Paddle smoke in {nb_path.name}")
        return
    raise SystemExit(f"Paddle smoke cell not found in {nb_path}")


def main() -> None:
    for nb_path in (
        REPO / "notebooks" / "colab" / "ocr_funsd_sroie_eval_colab.ipynb",
        REPO
        / "notebooks"
        / "kaggle-kernels"
        / "ocr_funsd_sroie_eval_kaggle"
        / "ocr_funsd_sroie_eval_kaggle.ipynb",
    ):
        _inject(nb_path)
    _inject_gate_colab()
    _inject_gate_kaggle()
    _inject_paddle_smoke(REPO / "notebooks" / "colab" / "ocr_funsd_sroie_eval_colab.ipynb")
    _inject_paddle_smoke(
        REPO
        / "notebooks"
        / "kaggle-kernels"
        / "ocr_funsd_sroie_eval_kaggle"
        / "ocr_funsd_sroie_eval_kaggle.ipynb",
    )


def _inject_gate_kaggle() -> None:
    _inject_gate(
        REPO
        / "notebooks"
        / "kaggle-kernels"
        / "ocr_funsd_sroie_eval_kaggle"
        / "ocr_funsd_sroie_eval_kaggle.ipynb"
    )


if __name__ == "__main__":
    main()
