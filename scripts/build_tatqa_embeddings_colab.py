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


def build_tatqa_chunks(tatqa_dir: Path, table_aware: bool = False):
    """Build TextNode chunks from TAT-QA train, dev, and test JSONs.

    The index is built from all splits (train, dev, test) so the full corpus is
    available for retrieval. Evaluation uses the test split only; having train/dev
    in the index avoids missing context when documents overlap or when you want
    a single consistent index.

    Splits and files: train (tatqa_dataset_train.json), dev (tatqa_dataset_dev.json),
    test (tatqa_dataset_test_gold.json). Missing files are skipped.

    If table_aware=True, tables are serialized with serialize_table_to_rows so each row
    is self-contained (row label + column: value). One chunk per table row, then
    paragraph chunks. Same Level 3 logic as FinQA (see RAG_LESSONS.md).

    Returns (chunks, context_by_corpus) for preprocess_chunks_for_index.
    """
    from rag_system.chunking import DocumentChunker, serialize_table_to_rows
    from llama_index.core.schema import TextNode

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)
    all_chunks = []
    context_by_corpus = {}
    splits_and_files = [
        ("train", "tatqa_dataset_train.json"),
        ("dev", "tatqa_dataset_dev.json"),
        ("test", "tatqa_dataset_test_gold.json"),
    ]
    for split, filename in splits_and_files:
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
            corpus_id = table.get("uid") if isinstance(table, dict) else None
            if not corpus_id:
                corpus_id = f"tatqa_{split}_{doc_idx}"
            meta = {"entry_id": doc_idx, "source": f"tatqa_{split}", "corpus_id": corpus_id}
            paragraphs = doc.get("paragraphs") or []
            para_str = "\n".join(
                p.get("text", "") for p in paragraphs if isinstance(p, dict)
            )
            table_str = "\n".join(" | ".join(str(c) for c in row) for row in table_rows) if isinstance(table_rows, list) else ""
            context_by_corpus[corpus_id] = f"{table_str}\n\n{para_str}".strip()

            if table_aware and isinstance(table_rows, list) and all(
                isinstance(r, (list, tuple)) for r in table_rows
            ):
                row_strings = serialize_table_to_rows(table_rows, first_row_is_header=True)
                for row_idx, row_str in enumerate(row_strings):
                    if not row_str.strip():
                        continue
                    row_meta = {**meta, "row_index": row_idx, "chunk_type": "table"}
                    all_chunks.append(TextNode(text=row_str, metadata=row_meta))
                if para_str.strip():
                    all_chunks.extend(chunker.chunk_document(para_str, metadata=meta))
                continue

            if isinstance(table_rows, list):
                table_str = "\n".join(" | ".join(str(c) for c in row) for row in table_rows)
            else:
                table_str = str(table_rows or "")
            doc_text = f"{table_str}\n\n{para_str}".strip()
            if not doc_text:
                continue
            chunks = chunker.chunk_document(doc_text, metadata=meta)
            all_chunks.extend(chunks)
    return all_chunks, context_by_corpus


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
    parser.add_argument(
        "--table_aware",
        action="store_true",
        help="One chunk per table row with headers inline (Level 3). Same as FinQA build.",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Skip content-hash deduplication (keep all chunks).",
    )
    args = parser.parse_args()

    tatqa_dir = REPO_ROOT / "data" / "rag" / "TAT-QA"
    if not tatqa_dir.exists():
        print(f"TAT-QA dir not found: {tatqa_dir}. Run scripts/download_rag_datasets.py --datasets tatqa first.")
        sys.exit(1)

    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    print("Building TAT-QA chunks from all splits (train, dev, test)..." + (" (table_aware=row-level serialization)" if args.table_aware else ""))
    chunks, context_by_corpus = build_tatqa_chunks(tatqa_dir, table_aware=args.table_aware)
    if not chunks:
        print("No chunks produced. Ensure data/rag/TAT-QA/ has at least one of: tatqa_dataset_train.json, tatqa_dataset_dev.json, tatqa_dataset_test_gold.json (run section 2 to download).")
        sys.exit(1)
    n_docs = len({(getattr(c, "metadata", None) or {}).get("corpus_id") for c in chunks})
    print(f"Got {len(chunks)} raw chunks from {n_docs} documents.")

    # Fallback corpus_id consistency: when table has no uid we use tatqa_{split}_{doc_idx}.
    # Adapter and eval_runner._build_tatqa_corpus_chunks must use the same split order and enumerate(data).
    # This assert is a build-time sanity check (each fallback doc got at least one chunk), not a correctness
    # guarantee — it won't catch wrong-id assignment from split-order mismatch. Real check: single-sample rerun.
    fallback_cids = {c for c in context_by_corpus if isinstance(c, str) and c.startswith("tatqa_")}
    n_fallback_docs = len(fallback_cids)
    n_fallback_chunks = sum(1 for c in chunks if (getattr(c, "metadata", None) or {}).get("corpus_id") in fallback_cids)
    if n_fallback_docs > 0:
        print(f"Fallback corpus_id: {n_fallback_docs} docs, {n_fallback_chunks} chunks (tatqa_<split>_<doc_idx>). Keep split/file/doc order in sync with TATQAAdapter.")
    assert n_fallback_chunks >= (n_fallback_docs if n_fallback_docs else 0), "Fallback-id chunks should cover all fallback docs."

    # Pre-index steps: section tagging, unit parsing, provenance (table_id), content hash, dedup
    from rag_system.index_preprocess import preprocess_chunks_for_index
    chunks = preprocess_chunks_for_index(
        chunks,
        context_by_corpus=context_by_corpus,
        page_by_corpus=None,
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
    print("  -> data/rag/TAT-QA/tatqa_retriever_index/")
    print("Then run eval_runner.py; it will load this index instead of building from scratch.")


if __name__ == "__main__":
    main()
