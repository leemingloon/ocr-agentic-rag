#!/usr/bin/env python3
"""
Export first sample image + metadata from vision parquet to data/proof/.../dev|test|validation/
for developer inspection. Answers:
- Is multiple-choice options metadata present (MMMU)?
- What does the image look like (save as PNG)?

Usage: python scripts/export_first_sample_images.py
"""
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def image_from_row_value(val):
    """Convert parquet cell (dict with bytes, or bytes, or PIL) to PIL Image."""
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict):
        b = val.get("bytes")
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = val.get("path")
        if p:
            return Image.open(str(p)).convert("RGB")
        return None
    if isinstance(val, bytes):
        return Image.open(io.BytesIO(val)).convert("RGB")
    return None


def _cell_to_py(cell):
    """Convert a pyarrow cell (e.g. StructScalar) to Python dict/bytes for image_from_row_value."""
    if cell is None:
        return None
    if hasattr(cell, "as_py"):
        return cell.as_py()
    return cell


def get_first_row(folder: Path, max_samples: int = 1):
    """Yield first row(s) from parquet shards in folder as dicts."""
    for f in sorted(folder.glob("*.parquet")):
        try:
            tbl = pq.read_table(f)
        except Exception:
            continue
        try:
            rows = tbl.to_pylist()
        except Exception:
            d = tbl.to_pydict()
            rows = [
                {k: _cell_to_py(v[i]) if hasattr(v, "__getitem__") else v for k, v in d.items()}
                for i in range(tbl.num_rows)
            ]
        for i, row in enumerate(rows):
            if i >= max_samples:
                break
            yield row
        return
    return


def main():
    base = Path(__file__).resolve().parent.parent
    proof = base / "data" / "proof" / "vision"

    # 1. MMMU_Accounting dev first sample
    mmmu_dev = base / "data" / "vision" / "MMMU_Accounting" / "dev"
    out_mmmu = proof / "mmmu_accounting" / "dev"
    out_mmmu.mkdir(parents=True, exist_ok=True)
    if mmmu_dev.exists():
        for row in get_first_row(mmmu_dev, 1):
            img = image_from_row_value(row.get("image_1"))
            if img is not None:
                img.save(out_mmmu / "first_sample_image.png")
                print(f"Saved {out_mmmu / 'first_sample_image.png'}")
            meta = {
                "sample_id": row.get("id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "options": row.get("options"),
                "options_present": row.get("options") is not None and str(row.get("options", "")).strip() not in ("", "[]"),
                "img_type": row.get("img_type"),
                "note": "Options are a string list: A=index0, B=index1, C=index2, D=index3. Answer 'D' = 4th option ($77,490 for first sample).",
            }
            with open(out_mmmu / "first_sample_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(out_mmmu / "first_sample_metadata.txt", "w", encoding="utf-8") as f:
                f.write("MMMU_Accounting dev – first sample\n")
                f.write("Multiple-choice options metadata: YES (in parquet 'options' column)\n")
                f.write(f"options (string): {meta.get('options')}\n")
                f.write(f"answer (correct letter): {meta.get('answer')}\n")
                f.write("Image: pictorial (table/chart). Saved as first_sample_image.png\n")
            break

    # 2. ChartQA test first sample
    chartqa_test = base / "data" / "vision" / "ChartQA" / "test"
    out_chartqa = proof / "chartqa" / "test"
    out_chartqa.mkdir(parents=True, exist_ok=True)
    if chartqa_test.exists():
        for row in get_first_row(chartqa_test, 1):
            img = image_from_row_value(row.get("image"))
            if img is not None:
                img.save(out_chartqa / "first_sample_image.png")
                print(f"Saved {out_chartqa / 'first_sample_image.png'}")
            meta = {
                "query": row.get("query"),
                "label": row.get("label"),
                "options_present": False,
                "note": "ChartQA has no multiple-choice options; GT is short answer in 'label' (e.g. ['14']).",
            }
            with open(out_chartqa / "first_sample_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(out_chartqa / "first_sample_metadata.txt", "w", encoding="utf-8") as f:
                f.write("ChartQA test – first sample\n")
                f.write("Multiple-choice options metadata: NO\n")
                f.write(f"query: {meta.get('query')}\n")
                f.write(f"label (ground truth): {meta.get('label')}\n")
                f.write("Image: chart/graph. Saved as first_sample_image.png\n")
            break

    # 3. InfographicsVQA validation first sample
    infovqa_val = base / "data" / "vision" / "InfographicsVQA" / "validation"
    out_infovqa = proof / "infographicsvqa" / "validation"
    out_infovqa.mkdir(parents=True, exist_ok=True)
    if infovqa_val.exists():
        for row in get_first_row(infovqa_val, 1):
            img = image_from_row_value(row.get("image"))
            if img is not None:
                img.save(out_infovqa / "first_sample_image.png")
                print(f"Saved {out_infovqa / 'first_sample_image.png'}")
            meta = {
                "question": row.get("question"),
                "answers": row.get("answers"),
                "options_present": False,
                "note": "InfographicsVQA has no MC options; GT in 'answers'. Case-insensitive match used in eval.",
            }
            with open(out_infovqa / "first_sample_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(out_infovqa / "first_sample_metadata.txt", "w", encoding="utf-8") as f:
                f.write("InfographicsVQA validation – first sample\n")
                f.write("Multiple-choice options metadata: NO\n")
                f.write(f"question: {meta.get('question')}\n")
                f.write(f"answers (ground truth): {meta.get('answers')}\n")
                f.write("Image: infographic. Saved as first_sample_image.png\n")
            break

    print("Done. Check data/proof/vision/<dataset>/<split>/ for first_sample_image.png and first_sample_metadata.txt")


if __name__ == "__main__":
    main()
