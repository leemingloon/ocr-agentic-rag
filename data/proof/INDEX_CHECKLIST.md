# Pre-index checklist (one Colab run)

Use this before building FinQA and TAT-QA indexes on Google Colab so a **single run** includes all indexing steps and you don’t miss anything (no false negatives, no second pass).

**Policy:** Fix indexing or pre-index preprocessing when retrieval fails (e.g. table row split, missing section context). Do not maintain a stored "chunking failure" label; the eval pipeline no longer uses one.

---

## 1. Header-per-row table-aware chunking

- **What:** Each table row is one chunk with **column headers inline** (e.g. `row_label | col1: val1 | col2: val2`). No row is split across chunks; no value is attributed to the wrong row.
- **Where:** `rag_system/chunking.py`: `serialize_table_row`, `serialize_table_to_rows` (first row = header, one string per data row).
- **Build:** Pass **`--table_aware`** to both scripts.
- **Notebook:** Set `TABLE_AWARE = True` in sections 4 and 4b.

**Verify:** Build log says “table_aware=row-level serialization” and chunk count is higher than without table_aware (more chunks = one per row + pre/post).

---

## 1b. Cross-page references and multi-level headers (chunking quality)

- **Cross-page:** Chunk text is scanned for "page N", "see page 5", "p. 12"; `metadata["references_pages"]` is set. At retrieval, when a chunk references pages not already in the result set, a second retrieve run fetches chunks from those pages (same corpus) and merges them.
- **Multi-level headers:** Document context is scanned for section headers; `metadata["header_hierarchy"]` is the ordered list. The generator prompt gets "Document outline (section hierarchy): ..." so the model sees where each chunk sits.

Both in `index_preprocess`; cross-page expansion in `retrieval_tools._rag_retrieval`; outline in orchestrator generator.

---

## 2. Section tagging at index time

- **What:** Every chunk gets `section_type` in metadata: `income_statement` | `balance_sheet` | `notes` | `unknown`, from document context (pre_text + post_text for FinQA; table + paragraphs for TAT-QA).
- **Where:** `rag_system/index_preprocess.py`: `infer_section_type`, `add_section_and_units` (uses `context_by_corpus`).
- **Build:** Automatic; no flag. FinQA uses pre+post as context; TAT-QA uses table+paragraphs.

**Verify:** After build, load bundle and inspect a chunk: `metadata["section_type"]` is set (e.g. `notes` for “Note 5” context).

---

## 3. Unit parsing at chunk ingestion

- **What:** Units (millions, thousands, per_share, quarterly, billions) are detected and stored in chunk `metadata["units"]`. Detection uses **chunk text + document context** so table rows get doc-level units (e.g. “in millions” in pre_text).
- **Where:** `rag_system/index_preprocess.py`: `detect_units`, `add_section_and_units` (merges units from context and chunk text).
- **Build:** Automatic.

**Verify:** Chunks from a doc that says “in millions” have `metadata["units"]` including `"millions"` (including table-row chunks from that doc).

---

## 4. Assembly-time unit normalisation

- **What:** When building the prompt for the model, a **canonical-unit note** is added from retrieved chunk metadata (e.g. “All dollar figures in the retrieved context are in millions unless stated otherwise (e.g. per share, thousands).”) so the model reasons in a consistent scale.
- **Where:** `rag_system/agentic/orchestrator.py`: `_units_note_from_results`, injected into `numbers_hint` before generation.
- **Build:** No build step; uses metadata from the index at query time.

**Verify:** Run a query with RAG_DEBUG=1 and check the prompt includes the unit note when retrieved chunks have `units` metadata.

---

## 5. Hash, deduplicate, and provenance at index build

- **What:**
  - **Hash:** Each chunk’s content is hashed (`content_hash`); stored in `metadata["content_hash"]`.
  - **Deduplicate:** Duplicate content (same hash) is merged: **first occurrence kept**, later ones dropped; kept chunk gets `metadata["duplicate_count"]` (incremented for each duplicate). Index is built over the deduplicated list.
  - **Provenance:** `page_number` (from FinQA id, e.g. page_134 → 134), `section_type`, `table_id` (e.g. `corpus_id_row0`) for table rows. Retriever **prefers primary source**: when RRF scores tie, chunks with lower `duplicate_count` rank first.
- **Where:** `rag_system/index_preprocess.py`: `content_hash`, `deduplicate_chunks`, `add_provenance`, `preprocess_chunks_for_index`. `rag_system/retrieval.py`: sort in `_reciprocal_rank_fusion` by `(score desc, duplicate_count asc)`.
- **Build:** Dedup is default; use `--no_dedup` only to keep all chunks (e.g. debugging).

**Dedup and multi-entry docs (FinQA):** When several entries share the same table (e.g. page_64.pdf-1 … pdf-4), dedup keeps the first occurrence only; that chunk's `corpus_id` is the first entry's id. Retrieval by another id then misses those table chunks. Use `scripts/inspect_finqa_hii_table.py` to see which id won. For HII/page_64, pdf-4 is first in train_qa order, so **dedup did not cause** missing table chunks for that sample.

**Verify:** Build log shows “After section/units/provenance/dedup: N chunks” with N ≤ raw chunk count. Chunks have `content_hash`, `duplicate_count`, and (where applicable) `page_number`, `table_id`.

---

## Single Colab run (no missing steps)

| Step | FinQA script | TAT-QA script | Notebook |
|------|--------------|---------------|----------|
| Table-aware (header-per-row) | `--table_aware` | `--table_aware` | `TABLE_AWARE = True` |
| Section tagging | ✅ in preprocess | ✅ in preprocess | automatic |
| Unit parsing (chunk + context) | ✅ in preprocess | ✅ in preprocess | automatic |
| Dedup + provenance | ✅ default | ✅ default | don’t pass `--no_dedup` |
| Assembly unit note | — | — | uses loaded index metadata |

**Commands (run once on Colab):**

```bash
# FinQA (with dedup; use --no_dedup if you suspect missing chunks for some corpus_ids)
python scripts/build_finqa_embeddings_colab.py --output data/rag/FinQA/train/finqa_retriever_index --table_aware --batch_size 256

# TAT-QA
python scripts/build_tatqa_embeddings_colab.py --output data/rag/TAT-QA/tatqa_retriever_index --table_aware --batch_size 170
```

To **reindex without dedup** (e.g. to rule out missing chunks): add `--no_dedup` to the FinQA command above. Index will be larger; every corpus_id keeps its own copy of shared table chunks.

Then download the bundles (notebook section 9) and replace your local index folders so eval and local runs use the same index.

---

## Final check: lessons covered (no indexing false negatives)

All of the following are implemented; this is the final audit against FinQA/TAT-QA and financial RAG literature.

| Lesson | Status | Where |
|--------|--------|--------|
| **Table-aware chunking** | Done | Header-per-row in `chunking.py`; `--table_aware` in both builds |
| **Cross-page references** | Done | `references_pages` in preprocess; cross-page expansion in `retrieval_tools._rag_retrieval` |
| **Multi-level headers** | Done | `extract_header_hierarchy` → `header_hierarchy`; "Document outline" in generator |
| **Footnotes (FinQA)** | Done | `_footnote_block_from_post` appended to last table row in FinQA build |
| **Section tagging** | Done | `infer_section_type` in preprocess; `section_types` filter at retrieval |
| **Unit parsing (chunk + context)** | Done | `detect_units`, `add_section_and_units`; assembly unit note in orchestrator |
| **Dedup + provenance** | Done | `content_hash`, `deduplicate_chunks`, `add_provenance`; RRF prefers lower `duplicate_count` |
| **Reranker** | Done | Cross-encoder in agentic path; `RAG_SKIP_RERANKER` to disable |
| **Query intent / primers** | Done | `classify_query_intent`; intent-driven primer selection in orchestrator |
| **Numerical grounding** | Done | `financial_constants.py` + missing-constant hint in generator |
| **Multi-hop (section filter)** | Done | `section_types` + `_infer_section_types_for_query` |
| **Negative retrieval / abstain** | Done | `RAG_RELEVANCE_THRESHOLD`; abstain when max reranker &lt; threshold |
| **Lost-in-the-middle** | Done | `_apply_bookends_order` (top-1 and top-2 at start/end) |

**Chunk size:** 512 tokens, 128 overlap (DocumentChunker) — matches literature (512–1024 for financial docs; 10–20% overlap). Narrative gets 512/128; table rows are one chunk each (no split).

**Supporting facts (FinQA):** Gold supporting-fact indices are for **evaluation** (context recall), not for indexing. No extra indexing step required.

**Optional / future:** (1) **FinBERT** (or domain embeddings) instead of generic embeddings for better financial term similarity. (2) **Two-stage retrieval:** document → page → chunk if you need to reduce noise at scale. (3) **TAT-QA footnote block:** FinQA has explicit post_text footnote extraction; TAT-QA uses table + paragraphs (footnotes may be in paragraphs); add a footnote block for TAT-QA only if you see retrieval misses on footnote values.

---

## Quick code references

- **Chunking (header-per-row):** `rag_system/chunking.py` → `serialize_table_to_rows`, `serialize_table_row`
- **Pre-index pipeline:** `rag_system/index_preprocess.py` → `preprocess_chunks_for_index`
- **Build entry points:** `scripts/build_finqa_embeddings_colab.py`, `scripts/build_tatqa_embeddings_colab.py` (both call `preprocess_chunks_for_index` before `build_index`)
- **Retrieval (prefer primary):** `rag_system/retrieval.py` → `_reciprocal_rank_fusion` (sort by duplicate_count)
- **Assembly unit note:** `rag_system/agentic/orchestrator.py` → `_units_note_from_results`
