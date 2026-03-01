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


def build_finqa_chunks(train_qa_path: Path, table_aware: bool = False):
    """Build TextNode chunks from FinQA train_qa.json.

    If table_aware=True, tables are serialized with serialize_table_to_rows so each row
    is self-contained (row label + column: value). One chunk per table row (+ pre/post
    text chunks). Avoids GS/2014-style misattribution when the chunker splits at column
    boundaries. Requires re-running the full index build and replacing the pre-built bundle.
    """
    from rag_system.chunking import DocumentChunker, serialize_table_to_rows
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
        corpus_id = entry.get("id") or entry.get("filename", str(idx))
        meta = {"entry_id": idx, "source": "finqa_train", "corpus_id": corpus_id}

        if table_aware and table and all(isinstance(r, (list, tuple)) for r in table):
            # Level 3: one self-contained chunk per table row (headers embedded)
            row_strings = serialize_table_to_rows(table, first_row_is_header=True)
            for row_str in row_strings:
                if not row_str.strip():
                    continue
                all_chunks.append(TextNode(text=row_str, metadata={**meta}))
            # Pre/post as separate chunks so retrieval can still hit context
            if pre_str.strip():
                all_chunks.extend(chunker.chunk_document(pre_str, metadata=meta))
            if post_str.strip():
                all_chunks.extend(chunker.chunk_document(post_str, metadata=meta))
            continue

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
        chunks = chunker.chunk_document(doc_text, metadata=meta)
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
    parser.add_argument(
        "--table_aware",
        action="store_true",
        help="Serialize each table row with column headers inline (Level 3). One chunk per row; avoids row-split misattribution (see RAG_LESSONS.md). Requires replacing the pre-built index.",
    )
    args = parser.parse_args()

    train_qa_path = args.train_qa if args.train_qa.is_absolute() else REPO_ROOT / args.train_qa
    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print("Building FinQA chunks..." + (" (table_aware=row-level serialization)" if args.table_aware else ""))
    chunks = build_finqa_chunks(train_qa_path, table_aware=args.table_aware)
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
