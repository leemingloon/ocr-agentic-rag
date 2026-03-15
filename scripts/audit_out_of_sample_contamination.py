#!/usr/bin/env python3
"""
Audit: Ensure finqa_train_samples.json and tatqa_test_samples.json are not contaminated
by QA from the out-of-sample sources (test.json, tatqa_dataset_dev.json).

- finqa_train_samples.json must contain ONLY sample_ids that appear in train_qa.json
  and ZERO from data/rag/FinQA/test/test.json.
- tatqa_test_samples.json must contain ONLY sample_ids (UIDs) that appear in
  tatqa_dataset_test_gold.json and ZERO from data/rag/TAT-QA/tatqa_dataset_dev.json.

Exit 0 if no contamination; exit 1 and print details if any contamination.
Run from repo root. Used for academic paper integrity.
"""

import json
import sys
from pathlib import Path


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def finqa_id(entry: dict, idx: int) -> str:
    return (
        entry.get("id")
        or (entry.get("filename") and f"{entry['filename']}-{idx}")
        or f"doc_{idx}"
    )


def tatqa_uids(data: list) -> set:
    uids = set()
    if not data or not isinstance(data, list):
        return uids
    for doc in data:
        for q in doc.get("questions", []):
            uid = q.get("uid")
            if uid:
                uids.add(uid)
    return uids


def main():
    repo = Path(__file__).resolve().parent.parent
    ok = True

    # --- FinQA ---
    train_qa_path = repo / "data" / "rag" / "FinQA" / "train" / "train_qa.json"
    test_path = repo / "data" / "rag" / "FinQA" / "test" / "test.json"
    proof_train_path = repo / "data" / "proof" / "rag" / "finqa" / "train" / "finqa_train_samples.json"

    train_qa = load_json(train_qa_path)
    test_data = load_json(test_path)
    proof_train = load_json(proof_train_path)

    train_ids = set()
    if train_qa:
        data = train_qa if isinstance(train_qa, list) else train_qa.get("data", [])
        for i, e in enumerate(data):
            train_ids.add(finqa_id(e, i))

    test_ids = set()
    if test_data:
        data = test_data if isinstance(test_data, list) else test_data.get("data", [])
        for i, e in enumerate(data):
            test_ids.add(finqa_id(e, i))

    proof_train_ids = {r["sample_id"] for r in proof_train if isinstance(r, dict)} if proof_train else set()
    in_test = proof_train_ids & test_ids
    not_in_train = proof_train_ids - train_ids

    print("=== FinQA: finqa_train_samples.json vs train_qa.json / test.json ===")
    print(f"  train_qa.json IDs: {len(train_ids)}, test.json IDs: {len(test_ids)}")
    print(f"  finqa_train_samples sample_ids: {len(proof_train_ids)}")
    if in_test:
        print(f"  CONTAMINATION: {len(in_test)} sample_ids from test.json: {sorted(in_test)[:20]}")
        ok = False
    else:
        print("  OK: 0 sample_ids from test.json")
    if not_in_train:
        print(f"  WARN: {len(not_in_train)} sample_ids not in train_qa.json: {sorted(not_in_train)[:10]}")
        ok = False
    print()

    # --- TAT-QA ---
    dev_path = repo / "data" / "rag" / "TAT-QA" / "tatqa_dataset_dev.json"
    test_gold_path = repo / "data" / "rag" / "TAT-QA" / "tatqa_dataset_test_gold.json"
    proof_test_path = repo / "data" / "proof" / "rag" / "tatqa" / "test" / "tatqa_test_samples.json"

    dev_data = load_json(dev_path)
    test_gold_data = load_json(test_gold_path)
    proof_test = load_json(proof_test_path)

    dev_uids = tatqa_uids(dev_data)
    test_gold_uids = tatqa_uids(test_gold_data)
    proof_test_ids = {r["sample_id"] for r in proof_test if isinstance(r, dict)} if proof_test else set()
    in_dev = proof_test_ids & dev_uids
    not_in_test_gold = proof_test_ids - test_gold_uids

    print("=== TAT-QA: tatqa_test_samples.json vs test_gold / dev ===")
    print(f"  tatqa_dataset_test_gold UIDs: {len(test_gold_uids)}, tatqa_dataset_dev UIDs: {len(dev_uids)}")
    print(f"  tatqa_test_samples sample_ids: {len(proof_test_ids)}")
    if in_dev:
        print(f"  CONTAMINATION: {len(in_dev)} sample_ids from tatqa_dataset_dev.json: {sorted(in_dev)[:20]}")
        ok = False
    else:
        print("  OK: 0 sample_ids from tatqa_dataset_dev.json")
    if not_in_test_gold:
        print(f"  WARN: {len(not_in_test_gold)} sample_ids not in test_gold: {sorted(not_in_test_gold)[:10]}")
        ok = False
    print()

    if ok:
        print("Audit passed: no contamination.")
    else:
        print("Audit FAILED: see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
