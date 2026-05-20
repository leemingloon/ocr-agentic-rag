#!/usr/bin/env python3
"""Download FUNSD + SROIE from HuggingFace into data/ocr/<DS>/<split>/*.parquet (Kaggle/Colab)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SPLITS = [
    ("FUNSD", "train", "nielsr/funsd"),
    ("FUNSD", "test", "nielsr/funsd"),
    ("SROIE", "train", "jsdnrs/ICDAR2019-SROIE"),
    ("SROIE", "test", "jsdnrs/ICDAR2019-SROIE"),
]


def export_one(repo_root: Path, dataset: str, split: str, hf_repo: str) -> int:
    from datasets import load_dataset

    out_dir = repo_root / "data" / "ocr" / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.parquet"))
    if existing and sum(p.stat().st_size for p in existing) > 10_000:
        print(f"  OK {dataset}/{split}: {len(existing)} shard(s) already present")
        return len(existing)

    print(f"  HF load {hf_repo} split={split} ...")
    ds = load_dataset(hf_repo, split=split)
    out_path = out_dir / f"{dataset.lower()}_{split}.parquet"
    ds.to_parquet(str(out_path))
    print(f"  wrote {out_path} ({ds.num_rows} rows)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print(f"Export OCR parquet under {root / 'data' / 'ocr'}")
    n = 0
    for dataset, split, hf_repo in SPLITS:
        try:
            n += export_one(root, dataset, split, hf_repo)
        except Exception as exc:
            print(f"  FAIL {dataset}/{split}: {exc}")
            return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
