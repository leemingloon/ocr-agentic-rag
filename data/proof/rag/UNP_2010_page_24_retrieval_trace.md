# Code trace: UNP/2010/page_24 — missing chunk (receivables securitization 400)

**Sample:** `UNP/2010/page_24.pdf-1`  
**Query:** "in 2010 what was the percent of the cash provided by operations that was from re[ceivables securitization facility]"  
**Gold:** `divide(400, 4505)` → 0.08879  
**Observed:** Only 2 chunks retrieved; chunk containing **400** (receivables securitization facility row) not in context → model cannot compute gold program.

---

## 1. Data source: train_qa.json

- **Path:** `data/rag/FinQA/train/train_qa.json`
- Entry keyed by `id` (e.g. `UNP/2010/page_24.pdf-1`) has:
  - `pre_text`, `post_text`, `table` (and/or `table_ori`)
  - Table rows include:
    - "cash provided by operating activities" | 2010: $ 4105 | ...
    - "receivables securitization facility [a]" | 2010: 400 | 2009: 184 | ...
    - "cash provided by operating activities adjusted for the receivables securitization facility" | 2010: 4505 | ...
- All of this is concatenated into one document per entry and then chunked.

**Reference:** `scripts/build_finqa_embeddings_colab.py` → `build_finqa_chunks(train_qa_path)` reads `train_qa.json` and builds one doc per entry.

---

## 2. Indexing: chunks from document

**File:** `scripts/build_finqa_embeddings_colab.py` — `build_finqa_chunks()`

- For each entry, `doc_text = pre_str + "\n\n" + table_str + "\n\n" + post_str`.
- `chunker.chunk_document(doc_text, metadata=meta)` with `chunk_size=512`, `chunk_overlap=128` produces multiple chunks. Metadata: `entry_id`, `source`, `corpus_id` (e.g. `UNP/2010/page_24.pdf-1`). **No `section_type` at this stage.**
- `context_by_corpus[corpus_id] = pre_str + "\n\n" + post_str` (no table) — used later for section inference.

So the **receivables securitization facility** row (400) is part of `table_str` and thus in `doc_text`; it will appear in one or more chunks. **Chunk 1614** in the index diagnostic is the chunk containing that row; it **is** in the index.

---

## 3. Preprocessing: section_type assignment

**File:** `rag_system/index_preprocess.py`

- `preprocess_chunks_for_index()` calls `add_section_and_units(chunks, context_by_corpus=context_by_corpus)`.
- For each chunk, `section_type = infer_section_type(text, context)` where `context = context_by_corpus.get(corpus_id, "")` = **pre_text + post_text** (no table).
- **`infer_section_type(combined)`** (combined = context + "\n" + text):
  - Returns `SECTION_INCOME_STATEMENT` only if combined matches:
    - `statement(s)? of operations`, `consolidated statement(s)? of operations`, `income statement`, `statements? of comprehensive income`, **`results? of operations`**
  - No match for bare "operations" or "operating activities".
  - For UNP/2010/page_24, pre/post text is about fuel, free cash flow, etc.; no "statement of operations" or "results of operations". So **all chunks for this document get `section_type = SECTION_UNKNOWN`.**

So at index time: **every chunk for UNP/2010/page_24 has `section_type = "unknown"`**. The chunk with 400 is in the index and tagged `unknown`.

---

## 4. Retrieval: section filter from query

**File:** `rag_system/agentic/retrieval_tools.py`

- `_rag_retrieval()` calls `_infer_section_types_for_query(query)`.
- **`_infer_section_types_for_query(query)`**:
  - `sections = []`
  - If `re.search(r"income\s+statement|statement(s)?\s+of\s+operations|operations", q)` → appends **`SECTION_INCOME_STATEMENT`**.
  - The regex includes **bare `"operations"`**. The query contains **"cash provided by operations"** → match → `section_types = ["income_statement"]`.

So for this query we pass **`section_types=['income_statement']`** to the retriever even though the question is about **cash flow** (cash provided by operations), not the income statement.

---

## 5. Retrieval: corpus_id + section filter → 0 chunks

**File:** `rag_system/retrieval.py` — `retrieve()`

- `corpus_id = 'UNP/2010/page_24.pdf-1'` → `corpus_id_indices` = all chunk indices whose `metadata.corpus_id` equals that (e.g. 10 chunks).
- `section_types = ['income_statement']` → `section_filter` = all chunk indices in the **whole index** with `metadata.section_type == 'income_statement'`.
- **`idx_set = corpus_id_indices & section_filter`**. For UNP/2010/page_24, every chunk has `section_type == 'unknown'`, so **no chunk is in section_filter** → **idx_set is empty**.
- Sparse and dense retrieval run over `idx_set` → **0 results**.
- **Hybrid fallback** (lines 357–368): when corpus-restricted returns 0, do **global** sparse + dense retrieval, then **filter to chunks belonging to this corpus_id**. So we get back only chunks that (1) rank in the global top list and (2) belong to UNP/2010/page_24. That yields **2 chunks** in the observed run. The chunk containing "receivables securitization facility | 2010: 400" (index 1614) does **not** appear in that small global top for the query, so it is **never returned**.

---

## 6. Summary: why chunk 1614 was “left out”

| Stage | What happens |
|--------|----------------|
| **Indexing** | Chunk 1614 (receivables securitization 400) **is** in the index. It is not “left out” of indexing. |
| **Section tag** | All UNP/2010/page_24 chunks get `section_type = "unknown"` because pre/post text does not match "statement of operations" etc. |
| **Query section** | Query contains "operations" → `_infer_section_types_for_query` returns `["income_statement"]` (over-broad). |
| **Retrieval** | Restrict to `corpus_id` + `section_type == 'income_statement'` → **0 chunks** (all are `unknown`). |
| **Fallback** | Global retrieval then filter to doc → only **2 chunks**; chunk 1614 is not in the global top → **not retrieved**. |

So the “missing chunk” is a **retrieval** effect: the **section filter** (triggered by the word "operations" in the query) zeroes out the document, and the hybrid fallback returns too few chunks to include the row with 400.

---

## 7. Recommended fix

**Query-side (surgical):** In `_infer_section_types_for_query`, **do not** add `SECTION_INCOME_STATEMENT` when the query is clearly about **cash flow** (e.g. "cash provided by operations", "cash from operations"). Those phrases refer to the statement of cash flows, not the income statement. Options:

- Exclude queries that contain "cash provided by operations" or "cash from operations" from adding `income_statement`, or
- Require a more specific phrase for income statement (e.g. "statement of operations", "income statement") and remove the bare word "operations" from the regex so "operations" alone does not trigger the filter.

**Implementation:** Tighten the regex so that "operations" in the sense of "cash provided by operations" does not set `section_types = ['income_statement']` (e.g. require "statement(s)? of operations" or "results? of operations", and do not add income_statement when "cash" and "operations" appear together as in cash flow context).
