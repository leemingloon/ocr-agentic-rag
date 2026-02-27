#!/usr/bin/env python3
"""
Build TAT-QA retriever index on GPU (e.g. Google Colab) and save a bundle for local use.

Uses the same chunking logic as eval_runner._build_tatqa_corpus_chunks.
Place the output folder at: data/rag/TAT-QA/tatqa_retriever_index/

Usage (from repo root):
  python scripts/build_tatqa_embeddings_colab.py --output data/rag/TAT-QA/tatqa_retriever_index

Optional:
  --batch_size N    Embedding batch size (use EMBED_BATCH_SIZE env or default 140)
  --cpu             Use CPU instead of GPU
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_tatqa_chunks(tatqa_dir: Path):
    """Build TextNode chunks from TAT-QA train and dev JSONs (same logic as eval_runner)."""
    from rag_system.chunking import DocumentChunker

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    for split, filename in [("train", "tatqa_dataset_train.json"), ("dev", "tatqa_dataset_dev.json")]:
        path = tatqa_dir / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for doc_idx, doc in enumerate(data):
            table = doc.get("table") or {}
            table_rows = table.get("table") if isinstance(table, dict) else table
            if isinstance(table_rows, list):
                table_str = "\n".join(" | ".join(str(c) for c in row) for row in table_rows)
            else:
                table_str = str(table_rows or "")
            paragraphs = doc.get("paragraphs") or []
            para_str = "\n".join(
                p.get("text", "") for p in paragraphs if isinstance(p, dict)
            )
            doc_text = f"{table_str}\n\n{para_str}".strip()
            if not doc_text:
                continue
            corpus_id = table.get("uid") if isinstance(table, dict) else None
            if not corpus_id:
                corpus_id = f"tatqa_{split}_{doc_idx}"
            chunks = chunker.chunk_document(
                doc_text,
                metadata={"entry_id": doc_idx, "source": f"tatqa_{split}", "corpus_id": corpus_id},
            )
            all_chunks.extend(chunks)
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Build TAT-QA index bundle on GPU for Colab.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "rag" / "TAT-QA" / "tatqa_retriever_index",
        help="Output directory for the index bundle",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("EMBED_BATCH_SIZE", "140")),
        help="Embedding batch size",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    tatqa_dir = REPO_ROOT / "data" / "rag" / "TAT-QA"
    if not tatqa_dir.exists():
        print(f"TAT-QA dir not found: {tatqa_dir}. Run scripts/download_rag_datasets.py --datasets tatqa first.")
        sys.exit(1)

    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print("Building TAT-QA chunks...")
    chunks = build_tatqa_chunks(tatqa_dir)
    if not chunks:
        print("No chunks produced. Ensure tatqa_dataset_train.json (and optionally dev) exist under data/rag/TAT-QA/.")
        sys.exit(1)
    n_docs = len({(getattr(c, "metadata", None) or {}).get("corpus_id") for c in chunks})
    print(f"Got {len(chunks)} chunks from {n_docs} documents.")

    from rag_system.retrieval import HybridRetriever

    device = "cpu" if args.cpu else "cuda"
    print(f"Loading retriever (device={device}, batch_size={args.batch_size})...")
    retriever = HybridRetriever(embedding_model="BAAI/bge-m3", device=device)
    retriever.build_index(chunks, batch_size=args.batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save_index_bundle(str(output_dir), embedding_model="BAAI/bge-m3")
    print(f"\nDone. Copy the folder to your local repo:\n  {output_dir}")
    print("  -> data/rag/TAT-QA/tatqa_retriever_index/")
    print("Then run eval_runner.py; it will load this index instead of building from scratch.")


if __name__ == "__main__":
    main()
