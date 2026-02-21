#!/usr/bin/env python3
"""
Reset perfect 1.0 evaluation proofs
Removes any JSON in data/proof where all values == 1.0
"""

import os
import json

PROOF_DIR = os.path.join("data", "proof")

if not os.path.exists(PROOF_DIR):
    print("No proof directory found, nothing to do")
    exit(0)

deleted_files = 0
for fname in os.listdir(PROOF_DIR):
    if fname.endswith(".json"):
        fpath = os.path.join(PROOF_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping corrupt proof file {fname}")
                continue
        # Check if all values are 1
        if isinstance(data, list) and all(float(v) == 1.0 for v in data):
            os.remove(fpath)
            deleted_files += 1
            print(f"Deleted perfect score file {fname}")

print(f"✅ Deleted {deleted_files} perfect score files")
