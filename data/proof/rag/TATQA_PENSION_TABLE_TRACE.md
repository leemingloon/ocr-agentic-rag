# TAT-QA pension “missing table” — code trace and root cause

This document traces why the pension **funding status** table (Projected benefit obligation, Accumulated benefit obligation, Fair value of plan assets) is not in the retrieved context for the question *“Which years does the table provide information for the projected benefit obligation, accumulated benefit obligation and fair value of plan assets for the company's Pension Plans?”* — and shows that the cause is **dataset structure + corpus_id scoping**, not indexing dropping the table.

---

## 1. Source dataset: `tatqa_dataset_test_gold.json`

TAT-QA is an array of **documents**. Each document has exactly **one table** and a list of paragraphs and questions:

- `doc["table"]` → one table (with `uid`)  
- `doc["paragraphs"]` → list of text blobs  
- `doc["questions"]` → list of QA pairs that reference **this** table and these paragraphs  

So **one document = one table**. The pension **asset allocation** table and the pension **funding status** table are in **two different documents**.

### Document A — asset allocation only (sample’s document)

| Field | Value |
|-------|--------|
| **Table uid** | `054153aec5a8b7066b1083f3ec3515ed` |
| **Table content** | Target Allocations 2020, Percentage of Plan Assets at December 31 (2019, 2018), Asset Category, Equity securities, Debt securities, Other, Total |
| **Question uid** | `227a0357e3487e7cd5ff15b8d86b1045` |
| **Question** | *“Which years does the table provide information for the projected benefit obligation, accumulated benefit obligation and fair value of plan assets for the company's Pension Plans?”* |
| **Answer** | `["2020", "2019", "2018"]` |

So in the **gold file**, this question is attached to the **allocation**-table document. The answer is the column years of **that** table (2020, 2019, 2018), even though the question text asks about PBO/ABO/fair value.

### Document B — funding status table (different document)

| Field | Value |
|-------|--------|
| **Table uid** | `dc0d8f2313478f7e229f7f76985d90ee` |
| **Table content** | As of December 31 (2019, 2018); **Projected benefit obligation**; **Accumulated benefit obligation**; **Fair value of plan assets** (with values) |
| **Question uid** | `9f83bb30c7281446998767f7bff3b72e` |
| **Same question text** | *“Which years does the table provide information for the projected benefit obligation, accumulated benefit obligation and fair value of plan assets…”* |
| **Answer** | `["2019", "2018"]` |

So the **funding status table is in the dataset** but in a **different** document (different table uid). It was **not** “left out” of the source file.

---

## 2. Adapter: how `corpus_id` is set

**File:** `eval_dataset_adapters.py` (TATQAAdapter).

For each sample, the adapter yields:

- `ground_truth["corpus_id"] = table.get("uid")` of the **document that contains that question**.

So for question uid `227a0357e3487e7cd5ff15b8d86b1045`:

- It lives in the document whose table uid is `054153aec5a8b7066b1083f3ec3515ed`.
- So **corpus_id = `054153aec5a8b7066b1083f3ec3515ed`** (the allocation-table doc).

Relevant code (same split/file order as indexer):

```python
for doc_idx, doc in enumerate(data):
    table = doc.get("table", {})
    corpus_id = table.get("uid") if isinstance(table, dict) else None
    if not corpus_id:
        corpus_id = f"tatqa_{split}_{doc_idx}"
    for q in doc.get("questions", []):
        ...
        yield { ..., "ground_truth": { "corpus_id": corpus_id, ... }, ... }
```

So every QA pair from Document A gets **corpus_id = 054153aec5a8b7066b1083f3ec3515ed**.

---

## 3. Indexing: how documents become chunks

**Files:**  
- `eval_runner.py` → `_build_tatqa_corpus_chunks()` (when prebuilt index is missing)  
- `scripts/build_tatqa_embeddings_colab.py` → `build_tatqa_chunks()`

**Logic (same in both):**

1. For each split: `tatqa_dataset_train.json`, `tatqa_dataset_dev.json`, `tatqa_dataset_test_gold.json`.
2. For each **document** in the list:
   - `corpus_id = table["uid"]` (or `tatqa_{split}_{doc_idx}` if no uid).
   - `doc_text = table_str + "\n\n" + para_str` (single concatenation of **that** doc’s table + paragraphs).
   - Chunks = `DocumentChunker(chunk_size=512, chunk_overlap=128).chunk_document(doc_text, metadata={"corpus_id": corpus_id, ...})`.
   - All chunks for that doc get the **same** `corpus_id`.

So:

- Document A → one concatenated text (allocation table + its 3 paragraphs) → N chunks, all with **corpus_id = 054153aec5a8b7066b1083f3ec3515ed**.
- Document B → one concatenated text (funding status table + its 2 paragraphs) → M chunks, all with **corpus_id = dc0d8f2313478f7e229f7f76985d90ee**.

**Nothing is dropped:** both documents are indexed. The funding status table **is** in the index, under **corpus_id = dc0d8f2313478f7e229f7f76985d90ee**.

---

## 4. Retrieval: why the funding table never appears

**File:** `rag_system/retrieval.py` (and orchestrator passing `corpus_id` from ground truth).

At eval time, for this sample:

- **corpus_id** passed to retrieval = **054153aec5a8b7066b1083f3ec3515ed** (from the adapter, i.e. the document that contains the question).
- Retrieval **restricts** to chunks whose `metadata["corpus_id"] == "054153aec5a8b7066b1083f3ec3515ed"`.
- So only chunks from **Document A** (allocation table + its paragraphs) are considered. Chunks from Document B (funding status table) are **never** retrieved because they have a different **corpus_id**.

So the “missing table” is **not** missing from the index; it is in the index under another document. It is **excluded by corpus_id scoping**.

---

## 5. Root cause summary

| Layer | What happens |
|-------|-----------------------------|
| **Dataset** | The question “which years … PBO, ABO, fair value” is placed in the **allocation-table** document (uid 054153…). The **funding status** table (PBO/ABO/fair value) is in a **different** document (uid dc0d8f…). |
| **Adapter** | `corpus_id` = table uid of the doc that contains the question → **054153aec5a8b7066b1083f3ec3515ed**. |
| **Indexing** | Each doc (one table + paragraphs) is chunked; **both** docs are indexed with their respective **corpus_id**. No table is dropped. |
| **Retrieval** | Eval passes this sample’s **corpus_id** to the retriever; only chunks from 054153… are returned. The funding table (dc0d8f…) is never searched for this sample. |

So the failure is **not** an indexing bug (table left out). It is:

1. **Dataset design:** One table per document; the question is attached to the allocation doc, while the semantically correct table (funding status) is in another doc.  
2. **Corpus_id scoping:** Retrieval is limited to the document that contains the question, so the other document (with the funding table) is never retrieved.

---

## 6. Verification (run script)

From repo root:

```bash
python scripts/verify_tatqa_pension_corpus.py
```

This reads `data/rag/TAT-QA/tatqa_retriever_index/chunks_meta.pkl` and reports:

- **054153aec5a8b7066b1083f3ec3515ed**: 6 chunks (allocation table — sample’s corpus_id).
- **dc0d8f2313478f7e229f7f76985d90ee**: 5 chunks (funding status table: PBO, ABO, fair value of plan assets).

So the funding table **is** in the index; the “missing table” is due to **corpus_id scoping** (retrieval for this sample only uses 054153…).

With `RAG_DEBUG=1`, the log line `RAG index diagnostic: corpus_id=054153aec5a8b7066b1083f3ec3515ed ... doc_chunks=6` confirms only the allocation doc’s 6 chunks are considered.

---

## 7. Possible fixes (product/design)

- **Don’t scope by corpus_id for this question type** (e.g. allow cross-document retrieval for “which years does the table provide … PBO/ABO/fair value”) so the model can see both pension tables.  
- **Dataset/annotation:** If the intended answer is from the funding status table, attach this question to the funding-status document (or add a multi-document link).  
- **Accept current behavior:** With corpus_id scoping, the model correctly says the required table is not in the retrieved context; the “FAIL” is then due to retrieval scope + dataset design, not reasoning.
