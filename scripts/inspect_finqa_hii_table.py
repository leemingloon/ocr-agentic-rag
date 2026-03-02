#!/usr/bin/env python3
"""
Inspect FinQA source table for HII/2018/page_64 (backlog portion sample).
Checks whether the Ingalls row has both 2018 and 2017 columns in the source data.

Usage (from repo root):
  python scripts/inspect_finqa_hii_table.py
  python scripts/inspect_finqa_hii_table.py --train_qa data/rag/FinQA/train/train_qa.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Inspect HII/2018/page_64 table in FinQA train_qa.json")
    parser.add_argument(
        "--train_qa",
        default=REPO_ROOT / "data" / "rag" / "FinQA" / "train" / "train_qa.json",
        type=Path,
        help="Path to train_qa.json",
    )
    args = parser.parse_args()
    path = args.train_qa if args.train_qa.is_absolute() else REPO_ROOT / args.train_qa
    if not path.exists():
        print(f"Not found: {path}")
        print("Download train_qa.json (e.g. scripts/download_rag_datasets.py) first.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        print("Unexpected format: expected list or {data: list}")
        sys.exit(1)

    # Find all entries for HII/2018/page_64 (same doc can appear as pdf-1, pdf-2, pdf-3, pdf-4 for different QA)
    hii_entries = []
    for e in data:
        eid = e.get("id") or e.get("filename") or ""
        if "HII" in eid and "page_64" in eid:
            hii_entries.append(e)
    if not hii_entries:
        print("No entry found with id/filename containing 'HII' and 'page_64'")
        print("Sample ids:", [str((e.get("id") or e.get("filename")))[:60] for e in data[:5]])
        sys.exit(1)

    # Dedup check: if multiple entries share the same table (same doc, different QA), dedup keeps first corpus_id only
    print(f"Entries with HII and page_64: {len(hii_entries)}")
    for e in hii_entries:
        print(f"  - {e.get('id') or e.get('filename')}")
    if len(hii_entries) > 1:
        first_table = (hii_entries[0].get("table") or hii_entries[0].get("table_ori") or [])
        first_hash = hashlib.sha256(json.dumps(first_table, sort_keys=True).encode()).hexdigest()[:16]
        same_table = all(
            (e.get("table") or e.get("table_ori") or []) == first_table for e in hii_entries
        )
        print(f"  Same table content across all: {same_table} (hash={first_hash})")
        if same_table:
            first_id = hii_entries[0].get("id") or hii_entries[0].get("filename")
            print("  -> DEDUP: build_finqa emits one set of table chunks per entry; dedup keeps FIRST occurrence (by data order).")
            print(f"     Table chunks will have corpus_id={first_id!r} only.")
            if "pdf-4" in str(first_id):
                print("     So retrieval for corpus_id=...pdf-4 WILL return table chunks (pdf-4 is first). Dedup is NOT the cause for pdf-4.")
            else:
                print("     So retrieval for corpus_id=...pdf-4 will NOT return those table chunks (they have another id). Dedup WOULD be the cause.")
    print()

    entry = hii_entries[0]  # use first for table inspection
    table = entry.get("table") or entry.get("table_ori") or []
    if not table:
        print(f"Entry {entry.get('id') or entry.get('filename')} has no 'table' or 'table_ori'")
        sys.exit(1)

    print(f"Entry id: {entry.get('id') or entry.get('filename')}")
    print(f"Table: {len(table)} rows")
    print()

    # Header (first row)
    headers = table[0] if table else []
    n_cols = len(headers)
    print(f"Header ({n_cols} columns):")
    for i, h in enumerate(headers):
        print(f"  [{i}] {repr(h)}")
    print()

    # Find Ingalls row (row label typically first column)
    ingalls_row = None
    ingalls_idx = None
    for i, row in enumerate(table[1:], start=1):
        if not row:
            continue
        label = (row[0] or "").strip().lower()
        if "ingalls" in label:
            ingalls_row = row
            ingalls_idx = i
            break
    if ingalls_row is None:
        print("No row with 'ingalls' in first column found. All row labels (first col):")
        for i, row in enumerate(table[1:], start=1):
            if row:
                print(f"  row {i}: {repr((row[0] or '')[:60])}")
        sys.exit(0)

    print(f"Ingalls row (table row index {ingalls_idx}, {len(ingalls_row)} columns):")
    for i, cell in enumerate(ingalls_row):
        header = headers[i] if i < len(headers) else ""
        print(f"  [{i}] {repr(header)}: {repr(cell)}")
    print()

    # Check for 2017 in headers and report GT reconciliation
    headers_str = " ".join(str(h).lower() for h in headers)
    has_2017 = "2017" in headers_str
    has_2018 = "2018" in headers_str
    print(f"Header contains '2017': {has_2017}")
    print(f"Header contains '2018': {has_2018}")
    print(f"Row has {len(ingalls_row)} columns (header has {n_cols})")
    if len(ingalls_row) != n_cols:
        print("  -> Column count mismatch: row may be truncated or header longer than row.")
    # GT 0.37399 = Ingalls 2017 total / 2017 total = 7991/21367. Check if row has 7991.
    if ingalls_row and len(ingalls_row) >= 7:
        # Last column is typically "total backlog" for second year (2017)
        total_2017_ingalls = ingalls_row[6].replace(",", "").replace("$", "").strip() if len(ingalls_row) > 6 else ""
        print(f"  -> Last column (total backlog, 2017): {repr(ingalls_row[6])} -> parsed '{total_2017_ingalls}'")
        if total_2017_ingalls == "7991" or total_2017_ingalls == "7,991":
            print("  -> SOURCE HAS 2017 Ingalls total (7991). GT 0.37399 = 7991/21367. Chunking should include full row.")
        else:
            print("  -> (Check if 7991 appears elsewhere in row for 2017 Ingalls total.)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
