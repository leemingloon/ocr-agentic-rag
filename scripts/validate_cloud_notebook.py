#!/usr/bin/env python3
"""Preflight: cloud OCR notebooks must compile and match GPU-fix invariants."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPO / "notebooks" / "colab" / "ocr_funsd_sroie_eval_colab.ipynb",
    REPO
    / "notebooks"
    / "kaggle-kernels"
    / "ocr_funsd_sroie_eval_kaggle"
    / "ocr_funsd_sroie_eval_kaggle.ipynb",
)
FORBIDDEN = (
    "min_vram_mb=500",
    "PaddleOCR({})",
    'attempts.extend([dict(base_kw), {"lang": "en", "use_gpu": use_gpu}, {"lang": "en"}, {}])',
)


def _cell_source(cell: dict) -> str:
    raw = cell.get("source", "")
    return raw if isinstance(raw, str) else "".join(raw)


def _validate_notebook(path: Path) -> list[str]:
    errs: list[str] = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = _cell_source(cell)
        for bad in FORBIDDEN:
            if bad in src:
                errs.append(f"{path.name} cell {i}: forbidden {bad!r}")
        stripped = "\n".join(
            ln for ln in src.splitlines() if not ln.strip().startswith("!")
        )
        try:
            compile(stripped, f"{path.name}:{i}", "exec")
        except SyntaxError as e:
            errs.append(f"{path.name} cell {i}: syntax error: {e}")
    setup = _cell_source(nb["cells"][1])
    try:
        tree = ast.parse(setup)
    except SyntaxError as e:
        errs.append(f"{path.name} setup cell: {e}")
        return errs
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_paddle_cuda_ready":
            for child in ast.walk(node):
                if isinstance(child, ast.If) and isinstance(child.test, ast.Name):
                    if child.test.id == "min_vram_mb":
                        for stmt in child.body:
                            if isinstance(stmt, ast.Raise):
                                break
                        else:
                            errs.append(
                                f"{path.name}: _assert_paddle_cuda_ready min_vram_mb branch malformed"
                            )
            break
    clone_idx = next(
        (
            i
            for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and _cell_source(c).startswith("# Clone or locate")
        ),
        None,
    )
    if clone_idx is None:
        errs.append(f"{path.name}: missing clone cell")
    else:
        clone = _cell_source(nb["cells"][clone_idx])
        if "from pathlib import Path" not in clone:
            errs.append(f"{path.name} clone cell: missing Path import")
        if '"IN_COLAB" not in globals()' not in clone:
            errs.append(f"{path.name} clone cell: missing setup prerequisite check")
    return errs


def main() -> int:
    all_errs: list[str] = []
    for nb in NOTEBOOKS:
        if not nb.is_file():
            all_errs.append(f"missing {nb}")
            continue
        all_errs.extend(_validate_notebook(nb))
    pg = REPO / "ocr_pipeline" / "paddle_gpu_check.py"
    if "PaddleOCR({})" in pg.read_text(encoding="utf-8") and "Never fall back" not in pg.read_text(
        encoding="utf-8"
    ):
        all_errs.append("paddle_gpu_check.py still allows PaddleOCR({}) fallback")
    if all_errs:
        for e in all_errs:
            print("FAIL:", e, file=sys.stderr)
        return 1
    print("OK: cloud notebooks compile and invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
