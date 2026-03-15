#!/usr/bin/env python3
"""
Verify that the TAT-QA index contains both pension documents:
- 054153aec5a8b7066b1083f3ec3515ed (asset allocation table; sample's corpus_id)
- dc0d8f2313478f7e229f7f76985d90ee (funding status table: PBO, ABO, fair value of plan assets)

Uses only chunks_meta.pkl (no embedding model). Run from repo root:
  python scripts/verify_tatqa_pension_corpus.py
"""
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "data" / "rag" / "TAT-QA" / "tatqa_retriever_index"

ALLOCATION_CORPUS = "054153aec5a8b7066b1083f3ec3515ed"
FUNDING_CORPUS = "dc0d8f2313478f7e229f7f76985d90ee"


def main():
    meta_path = INDEX_DIR / "chunks_meta.pkl"
    if not meta_path.exists():
        print(f"Index not found at {INDEX_DIR}; run build_tatqa_embeddings_colab.py first.")
        return 1
    with open(meta_path, "rb") as f:
        chunks_meta = pickle.load(f)
    cid_to_count = {}
    cid_to_preview = {}
    for m in chunks_meta:
        cid = (m.get("metadata") or {}).get("corpus_id") or ""
        cid_to_count[cid] = cid_to_count.get(cid, 0) + 1
        if cid in (ALLOCATION_CORPUS, FUNDING_CORPUS) and cid not in cid_to_preview:
            cid_to_preview[cid] = (m.get("text") or "")[:400]
    print("TAT-QA index: corpus_id -> chunk count (sample)")
    for cid in (ALLOCATION_CORPUS, FUNDING_CORPUS):
        n = cid_to_count.get(cid, 0)
        label = "allocation table (sample's doc)" if cid == ALLOCATION_CORPUS else "funding status table (PBO/ABO/fair value)"
        print(f"  {cid}: {n} chunks ({label})")
        if cid in cid_to_preview:
            preview = cid_to_preview[cid].replace("\n", " ")[:200]
            print(f"    preview: {preview}...")
    if cid_to_count.get(FUNDING_CORPUS, 0) > 0:
        print("\nConclusion: Funding status table IS in the index under corpus_id dc0d8f... .")
        print("The 'missing table' is due to corpus_id scoping: retrieval for the sample uses 054153... only.")
    else:
        print("\nWarning: No chunks found for funding corpus dc0d8f... ; check index build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
