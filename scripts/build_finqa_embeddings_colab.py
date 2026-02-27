#!/usr/bin/env python3
"""
Build FinQA retriever index on GPU (e.g. Google Colab) and save a bundle for local use.

Run this on Colab with Runtime > Change runtime type > T4 GPU, then download the
output folder and place it at: data/rag/FinQA/train/finqa_retriever_index/

Usage (from repo root):
  python scripts/build_finqa_embeddings_colab.py --output data/rag/FinQA/train/finqa_retriever_index

Optional:
  --train_qa PATH   Path to train_qa.json (default: data/rag/FinQA/train/train_qa.json)
  --batch_size N    Embedding batch size (default: 256 on GPU, 48 on CPU)
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_finqa_chunks(train_qa_path: Path):
    """Build TextNode chunks from FinQA train_qa.json (same logic as eval_runner)."""
    from rag_system.chunking import DocumentChunker
    from llama_index.core.schema import TextNode

    if not train_qa_path.exists():
        raise FileNotFoundError(f"train_qa.json not found: {train_qa_path}")
    with open(train_qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        return []

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    for idx, entry in enumerate(data):
        pre = entry.get("pre_text") or []
        post = entry.get("post_text") or []
        table = entry.get("table") or entry.get("table_ori") or []
        pre_str = "\n".join(pre) if isinstance(pre, list) else str(pre)
        post_str = "\n".join(post) if isinstance(post, list) else str(post)
        table_str = ""
        if table:
            for row in table:
                if isinstance(row, (list, tuple)):
                    table_str += " | ".join(str(c) for c in row) + "\n"
                else:
                    table_str += str(row) + "\n"
        doc_text = f"{pre_str}\n\n{table_str}\n\n{post_str}".strip()
        if not doc_text:
            continue
        corpus_id = entry.get("id") or entry.get("filename", str(idx))
        chunks = chunker.chunk_document(
            doc_text,
            metadata={"entry_id": idx, "source": "finqa_train", "corpus_id": corpus_id},
        )
        all_chunks.extend(chunks)
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Build FinQA index bundle on GPU for Colab.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "rag" / "FinQA" / "train" / "finqa_retriever_index",
        help="Output directory for the index bundle",
    )
    parser.add_argument(
        "--train_qa",
        type=Path,
        default=REPO_ROOT / "data" / "rag" / "FinQA" / "train" / "train_qa.json",
        help="Path to train_qa.json",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size (GPU: 256–512)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    train_qa_path = args.train_qa if args.train_qa.is_absolute() else REPO_ROOT / args.train_qa
    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print("Building FinQA chunks...")
    chunks = build_finqa_chunks(train_qa_path)
    if not chunks:
        print("No chunks produced. Check train_qa.json path and content.")
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
    print("  -> data/rag/FinQA/train/finqa_retriever_index/")
    print("Then run eval_runner.py; it will load this index instead of building from scratch.")


if __name__ == "__main__":
    main()
