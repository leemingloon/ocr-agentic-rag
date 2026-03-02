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


def _page_from_finqa_id(corpus_id: str):
    """Extract page number from FinQA id (e.g. GS/2014/page_134 -> 134, ADI/2009/page_49.pdf-1 -> 49)."""
    import re
    m = re.search(r"page_(\d+)", str(corpus_id), re.I)
    return int(m.group(1)) if m else None


def _footnote_block_from_post(post_str: str, max_chars: int = 800) -> str:
    """Extract a footnote block from post_text (e.g. (1) ... (2) ... or Note 1 ...) for appending to table chunk."""
    if not post_str or not post_str.strip():
        return ""
    import re
    # Match lines that look like footnote starters: (1), (2), Note 1, etc.
    lines = post_str.strip().split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\(\d+\)\s*|\s*note\s+\d+\s*[\.\-:]", line.strip(), re.I):
            start = i
            break
    if start is None:
        return ""
    block = "\n".join(lines[start:]).strip()
    return block[:max_chars] if len(block) > max_chars else block


def build_finqa_chunks(train_qa_path: Path, table_aware: bool = False):
    """Build TextNode chunks from FinQA train_qa.json.

    If table_aware=True, tables are serialized with serialize_table_to_rows so each row
    is self-contained (row label + column: value). One chunk per table row (+ pre/post
    text chunks). Avoids GS/2014-style misattribution when the chunker splits at column
    boundaries. Requires re-running the full index build and replacing the pre-built bundle.

    Returns (chunks, context_by_corpus, page_by_corpus) for preprocess_chunks_for_index.
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
        return [], {}, {}

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    context_by_corpus = {}
    page_by_corpus = {}
    for idx, entry in enumerate(data):
        pre = entry.get("pre_text") or []
        post = entry.get("post_text") or []
        table = entry.get("table") or entry.get("table_ori") or []
        pre_str = "\n".join(pre) if isinstance(pre, list) else str(pre)
        post_str = "\n".join(post) if isinstance(post, list) else str(post)
        corpus_id = entry.get("id") or entry.get("filename", str(idx))
        meta = {"entry_id": idx, "source": "finqa_train", "corpus_id": corpus_id}
        # Full doc context for section/unit inference (pre+post so table rows get doc-level section and units)
        context_by_corpus[corpus_id] = f"{pre_str}\n\n{post_str}".strip()
        page_num = _page_from_finqa_id(corpus_id)
        if page_num is not None:
            page_by_corpus[corpus_id] = page_num

        if table_aware and table and all(isinstance(r, (list, tuple)) for r in table):
            # Level 3: one self-contained chunk per table row (headers embedded)
            row_strings = serialize_table_to_rows(table, first_row_is_header=True)
            table_row_chunks = []
            for row_idx, row_str in enumerate(row_strings):
                if not row_str.strip():
                    continue
                row_meta = {**meta, "row_index": row_idx, "chunk_type": "table"}
                table_row_chunks.append(TextNode(text=row_str, metadata=row_meta))
            # Append footnotes to last table row chunk (parent table chunk)
            footnote_block = _footnote_block_from_post(post_str)
            if footnote_block and table_row_chunks:
                last = table_row_chunks[-1]
                last.text = (last.text or "").rstrip() + "\n\nFootnotes:\n" + footnote_block
            all_chunks.extend(table_row_chunks)
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
    return all_chunks, context_by_corpus, page_by_corpus


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
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Skip content-hash deduplication (keep all chunks). Use when reindexing if you suspect dedup causes missing chunks for some corpus_ids (e.g. multiple entries sharing the same table). Default: deduplicate.",
    )
    args = parser.parse_args()

    train_qa_path = args.train_qa if args.train_qa.is_absolute() else REPO_ROOT / args.train_qa
    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print("Building FinQA chunks..." + (" (table_aware=row-level serialization)" if args.table_aware else ""))
    chunks, context_by_corpus, page_by_corpus = build_finqa_chunks(train_qa_path, table_aware=args.table_aware)
    if not chunks:
        print("No chunks produced. Check train_qa.json path and content.")
        sys.exit(1)
    n_docs = len({(getattr(c, "metadata", None) or {}).get("corpus_id") for c in chunks})
    print(f"Got {len(chunks)} raw chunks from {n_docs} documents.")

    # Pre-index steps: section tagging, unit parsing, provenance (page, table_id), content hash, dedup
    from rag_system.index_preprocess import preprocess_chunks_for_index
    chunks = preprocess_chunks_for_index(
        chunks,
        context_by_corpus=context_by_corpus,
        page_by_corpus=page_by_corpus,
        table_id_prefix_by_corpus={cid: cid for cid in context_by_corpus},
        dedup=not args.no_dedup,
    )
    print(f"After section/units/provenance/dedup: {len(chunks)} chunks.")

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
