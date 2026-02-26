#!/usr/bin/env python3
"""
Simulate FinQA retrieval without calling Claude. Use this to verify what chunks
the LLM would see before spending API credits.

Example:
  python scripts/simulate_finqa_retrieval.py "what was the total operating expenses in 2018 in millions" "AAL/2018/page_13.pdf-2" --debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate FinQA RAG retrieval (no LLM)")
    ap.add_argument("query", help="Question (e.g. FinQA query)")
    ap.add_argument("corpus_id", help="Document id (e.g. AAL/2018/page_13.pdf-2)")
    ap.add_argument("--debug", action="store_true", help="Use debug corpus (first 80 entries)")
    ap.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve (default 10)")
    args = ap.parse_args()

    # Add repo root for imports
    sys.path.insert(0, str(REPO_ROOT))

    # Reuse eval_runner's retriever (builds index if needed)
    from eval_runner import _get_rag_retriever_for_dataset

    print("Building or loading FinQA retriever (this may take a minute with embeddings)...")
    retriever = _get_rag_retriever_for_dataset("FinQA", debug=args.debug)
    print("Retrieving...")
    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        corpus_id=args.corpus_id,
    )
    print(f"\nRetrieved {len(results)} chunks for corpus_id={args.corpus_id!r}\n")
    print("=" * 60)
    for i, (chunk, score) in enumerate(results, 1):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        print(f"\n--- Chunk {i} (score={score:.4f}) ---\n{text[:1200]}")
        if len(text) > 1200:
            print("... [truncated]")
    print("\n" + "=" * 60)
    if not results:
        print("No chunks returned. Check that corpus_id exists in the index (e.g. id in train_qa.json).")
        # Hint: with --debug only first 80 entries are indexed; AAL/2018 might be after that
        train_qa = REPO_ROOT / "data" / "rag" / "FinQA" / "train" / "train_qa.json"
        if train_qa.exists():
            with open(train_qa, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(entries, list):
                for idx, e in enumerate(entries):
                    if str(e.get("id")) == args.corpus_id:
                        print(f"  corpus_id found at entry index {idx}. With --debug only first 80 entries are indexed.")
                        break
                else:
                    print(f"  corpus_id {args.corpus_id!r} not found in train_qa.json.")


if __name__ == "__main__":
    main()
