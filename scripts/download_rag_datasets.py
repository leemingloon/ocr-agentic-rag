#!/usr/bin/env python3
"""
Download RAG evaluation datasets (FinQA, TAT-QA) into data/rag/.

Adapted from download_datasets.py; no HuggingFace token required for these.
Use --datasets to choose which to download (default: finqa tatqa).

Usage:
  python scripts/download_rag_datasets.py
  python scripts/download_rag_datasets.py --datasets finqa
  python scripts/download_rag_datasets.py --datasets finqa tatqa
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

# Base dir: run from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASE_DIR = REPO_ROOT / "data" / "rag"

# FinQA: single file from official GitHub
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
FINQA_PATH = BASE_DIR / "FinQA" / "train" / "train_qa.json"

# TAT-QA: JSON files from NExTplusplus/TAT-QA
TATQA_BASE = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw"
TATQA_FILES = [
    "tatqa_dataset_train.json",
    "tatqa_dataset_dev.json",
    "tatqa_dataset_test.json",
    "tatqa_dataset_test_gold.json",
]
TATQA_DIR = BASE_DIR / "TAT-QA"


def download_finqa(force: bool = False) -> bool:
    """Download FinQA train_qa.json from GitHub. Returns True if success."""
    if FINQA_PATH.exists() and not force:
        print(f"FinQA: already present at {FINQA_PATH}")
        return True
    FINQA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading FinQA train_qa.json from GitHub...")
        urllib.request.urlretrieve(FINQA_URL, FINQA_PATH)
        print(f"Saved to {FINQA_PATH}")
        return True
    except Exception as e:
        print(f"FinQA download failed: {e}")
        return False


def download_tatqa(force: bool = False) -> bool:
    """Download TAT-QA dataset JSON files from GitHub. Returns True if all success."""
    TATQA_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in TATQA_FILES:
        path = TATQA_DIR / name
        if path.exists() and not force:
            print(f"TAT-QA: {name} already present")
            continue
        url = f"{TATQA_BASE}/{name}"
        try:
            print(f"Downloading TAT-QA {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  Failed: {e}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download RAG datasets (FinQA, TAT-QA).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["finqa", "tatqa"],
        choices=["finqa", "tatqa"],
        help="Which datasets to download (default: finqa tatqa)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    if "finqa" in args.datasets:
        results["FinQA"] = download_finqa(force=args.force)
    if "tatqa" in args.datasets:
        results["TAT-QA"] = download_tatqa(force=args.force)

    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
