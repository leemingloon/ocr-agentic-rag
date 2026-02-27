# RAG / FinQA: Lessons from Prediction Failures

Summary of issues encountered and fixes applied so that future runs and readers don’t repeat the same mistakes.

---

## 1. Metrics: Don’t Mark Wrong Answers as Correct

**Issue:** For numeric answers, “exact_match” was 1.0 even when the model’s number was wrong (e.g. 2.4M vs GT 3.8M). Reason: the metric used “any number in the prediction within 5% of reference” (e.g. 3.9 in the passage counted as matching 3.8).

**Fix:** In `eval_postprocess_utils.py`, **FinQAUtils** now uses **strict numerical match** for `exact_match` when the reference is numeric: only the extracted answer number is compared to the reference. So wrong numbers get 0.

**Takeaway:** For numerical QA, treat `numerical_exact_match` (and the strict `exact_match` for numerics) as the main metric; avoid “any number in text within 5%” for grading.

---

## 2. First Step Must Be RAG (Retrieval)

**Issue:** The agent sometimes chose the calculator for “Step 1” and passed the **plan phrase** (e.g. “Locate the company’s 2012 financial statements”) as the calculator expression → “Could not parse expression” → no context → model said “I cannot access the data.”

**Fix:** In `rag_system/agentic/orchestrator.py`, **step 0 is forced to RAG retrieval** using the **user query** (not the plan step). So the model always gets document context first.

**Takeaway:** For single-query RAG, always retrieve first; only then use calculator/SQL on extracted numbers or expressions.

---

## 3. Pass `corpus_id` So Retrieval Is Scoped

**Issue:** Retrieval searched the **entire** FinQA corpus. FinQA questions are tied to a specific document (e.g. `AAL/2018/page_13.pdf-2`); without scoping, the right doc can be outranked.

**Fix:** `ground_truth.corpus_id` is passed from the eval runner into `rag.query(query, corpus_id=...)`. The retriever filters (or ranks by) chunks whose `metadata.corpus_id` matches. Chunk metadata is set when building the index from `train_qa.json`.

**Takeaway:** When the benchmark has a document id per question, pass it and scope retrieval to that document.

---

## 4. Give the LLM Clear Chunk Text, Not a Raw Dict

**Issue:** The generator was sending the RAG tool result as a **stringified dict** (e.g. `{'chunks': [{'text': '...'}, ...]}`). Hard for the model to use and easy to truncate.

**Fix:** In the orchestrator’s generator, when the tool result is a RAG result with `chunks`, we now format context as **“Retrieved documents: [Document 1] … [Document 2] …”** with plain chunk text.

**Takeaway:** Format RAG results as clear, numbered document text in the prompt.

---

## 5. More Chunks When One Document Is Enough

**Issue:** With `corpus_id` set we only retrieved 5 chunks; for a long document the table with the answer could be in chunk 6+.

**Fix:** In `retrieval_tools.py`, when `corpus_id` is set we request more chunks (e.g. up to 10–20) so the model sees more of that document.

**Takeaway:** When retrieval is scoped to one doc, use a larger `top_k` so key tables/paragraphs aren’t missed.

---

## 6. Numerical Hint in the Prompt

**Issue:** The model had the numbers (e.g. fuel $9,896M, 23.6% of total) but didn’t infer “total = 9896 / 0.236”.

**Fix:** The generator prompt now includes a short instruction: for financial tables, if you have a component and its “percent of total,” you can compute total = component / (percent/100).

**Takeaway:** For formula-heavy benchmarks (e.g. FinQA), a one-line hint in the prompt can reduce “I cannot determine” when the data is present.

---

## 7. Calculator: Don’t Feed It Natural Language

**Issue:** When the calculator was chosen, it received a **sentence** (e.g. “Determine the value of…”) and tried to `eval()` it → parse error.

**Fix:** Step 0 is always RAG (above). The calculator now rejects clearly non-math input and returns a short message: “Calculator expects a mathematical expression… Use RAG retrieval first.”

**Takeaway:** Only send the calculator a cleaned math expression (e.g. from the LLM or a dedicated step), not the user or plan text.

---

## 8. Re-evaluating Existing Samples (Metrics vs Predictions)

**Issue:** Existing rows in `*_per_sample_*.json` were produced **before** the metric fix (and possibly before RAG/generator fixes). So stored “exact_match”/“f1” can still be wrong for old rows.

**Fix:** Metrics are computed at **evaluation time**. To refresh metrics for existing samples you must **re-run the model** for those samples (e.g. delete or rename the per_sample file so they are not skipped, or add a “re-evaluate” option). Simply re-running with the same samples **skipped** only refreshes aggregates from existing rows; it does not recompute metrics from the updated FinQAUtils.

**Takeaway:** To get correct metrics after code fixes, re-run inference for the samples (or re-evaluate from stored predictions if the runner supports it).

---

## 9. Debugging “Cannot Access Table Data” (FinQA)

**Symptom:** The model often says it cannot access the data or the specific table values, even though the document contains them.

**Causes:** (1) **corpus_id mismatch** — ground_truth uses `id` (e.g. `AAL/2018/page_13.pdf-2`) but the index was built with `filename` only, so filter returns 0 chunks. (2) **Too few chunks** — table lives in a chunk that ranks after the top-k, so it never reaches the generator. (3) **Chunking** — the table is split across chunks and the key row is in a chunk that isn’t retrieved.

**Debug with `--debug`:** Run `python eval_runner.py --category rag --dataset FinQA --debug ...`. You’ll see:
- `[DEBUG] RAG query corpus_id=... query=...` — confirms the corpus_id and query sent to RAG.
- `[DEBUG] RAG FinQA: building index ... sample corpus_ids: [...]` — check that eval sample corpus_ids appear in this list.
- `[DEBUG] retrieval: corpus_id=... before_filter=... after_filter=...` — if **after_filter=0**, the document isn’t in the index or the id doesn’t match (fallback tries prefix match).
- `[DEBUG] _rag_retrieval: ... num_chunks=... first_corpus_id=... has_table_like_content=...` — if num_chunks is 0, retrieval failed; if has_table_like_content is False, the top chunks may be narrative-only (table chunk ranked lower).
- `[DEBUG] RAG result: N chunks retrieved; first_chunk_preview=...` — inspect what the model actually saw.

**Fixes applied:** (1) Adapter uses `corpus_id = entry.get("id") or entry.get("filename")` so it matches the index. (2) Retrieval fallback: if exact corpus_id match gives 0 chunks, try matching by document prefix (e.g. `AAL/2018/page_13.pdf`). (3) When corpus_id is set we request more chunks (up to 30) so table content is less likely to be cut off. (4) Re-run samples (clear or move the per_sample file) so new runs hit the pipeline and you can see the debug logs.

---

## 10. Debug Run: Avoid Double-Loading (Segfault)

**Issue:** With `--debug --category rag`, after the dataset run the runner also ran **adversarial** tests, which created a **new** retriever and **new** reranker. Loading the large reranker (and embeddings) twice in one process can cause OOM or **segmentation fault** on 16GB.

**Fix:** When `--debug` is set, **adversarial is skipped** so the reranker (and heavy models) are not loaded a second time. Run adversarial without `--debug` when you have enough memory.

**Takeaway:** In resource-constrained debug runs, avoid running a second heavy pipeline (e.g. adversarial) in the same process.

---

## 11. Program Synthesis + Execution (FinQA / Kaggle Best Practice)

**Why:** FinQA is hard because answers require **multi-step arithmetic** (e.g. total = fuel_expense / (percent/100)). Letting the LLM output a number in prose is error-prone; the benchmark is designed for **program execution accuracy**: generate a program (add/subtract/multiply/divide), execute it, compare to `exe_ans`.

**What we did:**  
- **`rag_system/finqa_program_executor.py`**: Executes FinQA-style programs: `divide(9896, 23.6%)`, `subtract(a,b), divide(#0, c)` with percentage normalization and step references.  
- **Generator prompt**: We ask the model to optionally output a one-line program (e.g. `divide(9896, 23.6%)`) that we will execute.  
- **Post-process**: If the model’s reply contains an executable program, we run it and append **Numerical answer (from program execution): &lt;value&gt;** so evaluation’s numerical extraction and `numerical_exact_match` use the precise result.

**Takeaway:** For numerical QA over tables, program-synthesis + execution (as in official FinQA and SOTA like APOLLO) gives more reliable scores than free-form number-in-text.

---

## 12. Best Practices from Kaggle and Industry (FinQA / Multi-Hop Numerical QA)

- **Execution accuracy over program accuracy:** Optimize for “does the predicted program produce the right number?” (execution accuracy); exact program match (program accuracy) is harder and not necessary for correctness.  
- **Number-aware retrieval:** SOTA (e.g. APOLLO) uses number-aware negative sampling so the retriever prefers facts that contain the relevant numbers; we scope by `corpus_id` and use more chunks when scoped.  
- **Program format:** FinQA uses `add(a,b)`, `subtract(a,b)`, `multiply(a,b)`, `divide(a,b)` and step references `#0`, `#1`. Supporting this format and executing it avoids rounding/typo errors from the LLM.  
- **Percentage handling:** In programs, `23.6%` should be interpreted as 0.236 (e.g. `divide(9896, 23.6%)` → 9896/0.236). Our executor does this.  
- **Retriever–generator pipeline:** Standard is two-stage: retriever selects supporting facts (we use hybrid + corpus_id scoping); generator produces program or answer from those facts. Always retrieve first (we force step 0 to RAG).  
- **Consistency and augmentation:** Top methods (APOLLO) use consistency-based RL and program augmentation; we don’t train, but prompting the model to output executable programs and then executing them gives a similar benefit at inference time.

---

## What to Do Next

- **Re-run with fixes:** To see improved predictions and correct metrics, run without skipping: e.g. temporarily move or clear the FinQA per_sample file for the split you care about, then run again so those samples are re-evaluated (and metrics recomputed with FinQAUtils + new RAG/generator behavior).
- **Simulate before spending API credits:** Use `scripts/simulate_finqa_retrieval.py` to check what chunks the model would see for a given query and `corpus_id`.
- **Primary metric for FinQA:** Use **numerical_exact_match** (and the strict exact_match for numeric answers) when reporting or comparing runs.
- **Faster embedding (Colab GPU):** Use **notebooks/demo_agentic_rag_eval.ipynb** on Google Colab (free T4 GPU): it downloads FinQA + TAT-QA, builds the FinQA index on GPU, runs RAG eval, and displays results. You can also download the `finqa_retriever_index` zip and unzip into `data/rag/FinQA/train/finqa_retriever_index/` for local runs.

---

## Research-Backed Techniques to Increase FinQA Accuracy

Summary of techniques from recent papers and benchmarks (2024–2025) that can improve our RAG pipeline’s numerical accuracy on FinQA. Ordered by impact and feasibility for our current setup.

### 1. Number-aware retrieval (APOLLO-style)

**Idea:** Train or bias the retriever so it prefers facts that contain the **numbers** needed for the answer, not just semantically similar text. Standard retrieval treats all facts equally; FinQA needs the specific table cells and figures that hold the operands.

**Papers:** APOLLO (LREC-COLING 2024) uses number-aware negative sampling for the retriever; MultiFinRAG uses modality-aware similarity thresholds (e.g. 80% text, 65% tables/images) so numerical/table chunks aren’t drowned out.

**What we can do without retriever training:**
- **Boost chunks that contain numbers** when the query is numerical: at retrieval time, optionally re-rank or filter chunks by “has numbers” (e.g. regex or simple NER) and prefer those when the gold answer or query suggests a calculation.
- **Separate table vs text in indexing:** If we have table-structured content (e.g. from FinQA’s gold “table” facts), index table rows/cells with a `modality=table` tag and use a lower similarity threshold for table chunks so they are not excluded by a single high bar.
- **Larger top_k when corpus_id is set:** We already do this; consider 20–30 for long documents so multi-hop numerical facts (e.g. two different table rows) are both retrieved.

### 2. Preserve table structure in chunks (MultiFinRAG, SQuARE, TaCube)

**Idea:** Don’t flatten tables into plain text; keep headers, row/column relationships, and units so the model can “see” which number is in which cell.

**Papers:** MultiFinRAG converts tables (and figures) via a multimodal LLM into **structured JSON + short summaries**, then indexes both. SQuARE uses structure-preserving chunking (header hierarchy, time labels, units). TaCube pre-computes aggregates (sum, average) and attaches them to the table context.

**What we can do:**
- **Chunk tables as units:** When building the FinQA corpus from `train_qa.json`, if we have access to table snippets or gold “table” facts, store each table (or each table row group) as **one chunk** with a clear header row and aligned columns (e.g. markdown or CSV-like lines), instead of splitting mid-table.
- **Add units and column names:** Ensure chunk text includes column headers and units (e.g. “($ millions)”, “% of total”) so the model knows how to interpret and combine numbers.
- **Optional: pre-computed aggregates:** For tables we control, we could add one line per table like “Row sum: X, Column sum: Y” (TaCube-style) to reduce arithmetic errors when the question asks for totals or averages.

### 3. Program synthesis + execution (we already do this; strengthen it)

**Idea:** FinQA is designed for **execution accuracy**: the model outputs a program (e.g. `divide(9896, 23.6%)`), we execute it, and we compare the result to the gold answer. This is more reliable than free-form number-in-text.

**Papers:** Official FinQA evaluation uses execution accuracy; APOLLO uses consistency-based RL and program augmentation so programs that execute to the same answer are not penalized for differing from the gold program.

**What we can do:**
- **Stronger prompt:** Explicitly ask the model to output **one line** in FinQA format: `add/subtract/multiply/divide` and step refs `#0`,`#1`, and that we will execute it. Give 1–2 examples in the system or user message.
- **Program augmentation at eval:** We don’t train; at inference we can still try small variants (e.g. divide then round) if the first execution is close to the gold (for analysis only; for reporting stick to one program per query).
- **Fallback:** If the model doesn’t output an executable program, keep using our current “extract number from text + program executor” path; ensure the executor handles percentage and step refs robustly (we already have `finqa_program_executor`).

### 4. Tiered / modality-aware retrieval (MultiFinRAG)

**Idea:** First retrieve high-confidence **text** chunks; if too few hits or low similarity, escalate to **table** and **image** chunks and merge contexts. Use different similarity thresholds per modality so tables (often more “keywordy”) aren’t dropped.

**What we can do:**
- **Two-phase retrieval:** Phase 1: retrieve text chunks with current threshold. Phase 2: if top score &lt; 0.7 or count &lt; 3, retrieve again from table-tagged chunks (or all chunks) with a lower threshold (e.g. 0.5) and append to context.
- **Tag chunks by type:** When building the index, set `chunk_type: "text" | "table"` from FinQA gold (e.g. “table” vs “text” in the supporting facts). Query logic can then prefer or add table chunks when the query looks numerical (e.g. “what was the total”, “percentage”, “how much”).

### 5. Semantic chunking and merging (MultiFinRAG)

**Idea:** Avoid splitting in the middle of a sentence or table. Use sentence-level segmentation, then sliding windows, then **merge** chunks that are very similar (e.g. cosine &gt; 0.85) to reduce redundancy and keep coherent units.

**What we can do:**
- **Chunk on sentence boundaries:** When we build FinQA corpus from narrative + table text, split on sentences first, then form chunks of N sentences with overlap; avoid cutting in the middle of “$9,896” or “23.6%”.
- **Merge near-duplicate chunks:** After building chunks, merge any two with embedding similarity above 0.85 to shrink context size and avoid diluting the prompt (MultiFinRAG reports ~40–60% chunk reduction).

### 6. Retriever fine-tuning (APOLLO, ReasonIR, RAG-IT)

**Idea:** Fine-tune the retriever (e.g. contrastive or supervised) so it retrieves facts that lead to **correct answers** (e.g. execution accuracy) rather than only semantic similarity. APOLLO uses number-aware negative sampling; ReasonIR trains on synthetic reasoning queries; RAG-IT does retrieval-augmented instruction tuning for financial analysis.

**What we can do (larger effort):**
- **Collect training signal:** For FinQA train split, we have (query, gold program, gold answer, supporting facts). Use gold facts as positives and other facts from the same doc as negatives (number-aware: negatives that lack the key numbers).
- **Fine-tune our embedder:** Add a contrastive head or use a framework (e.g. sentence-transformers training) to fine-tune BGE/MiniLM on (query, positive_chunk, negative_chunks) and re-index.
- **ReasonIR-style:** If we adopt a retriever trained for “reasoning” (e.g. ReasonIR-8B), we could use it for the FinQA index; reported +6.4% on MMLU, +22.6% on GPQA when used for RAG.

### 7. Generator improvements (without full fine-tuning)

**Idea:** Help the LLM use the retrieved context and output executable programs.

**What we can do:**
- **Few-shot examples:** Add 1–2 FinQA examples (query → supporting numbers → one-line program → answer) in the prompt so the model sees the expected format.
- **Explicit “supporting numbers” section:** Before “Generate your answer”, list “Numbers you may use: …” extracted from the retrieved chunks (e.g. all numbers with units) to reduce hallucination and focus the model on the right values.
- **Verification step:** After the model outputs a program and we execute it, optionally add a reflector step: “The program yielded X. Is this consistent with the retrieved context? If not, try again with a different program.” (Can be limited to one retry to control cost.)

### 8. How Kaggle / Community Address Our Three Debugging Issues

The three recurring problems (corpus_id mismatch, too few chunks, chunking) are well studied in FinQA/TAT-QA and long-document financial QA. Below is how top solutions and the literature handle them.

---

#### Issue 1: Corpus_id / document scoping mismatch (wrong doc or 0 chunks)

**What happens:** Eval uses a per-question document id (e.g. `AAL/2018/page_13.pdf-2`); if the index uses a different key (e.g. filename only) or the id format differs, filtering returns 0 chunks and the model “cannot access” the data.

**How Kaggle / community handle it:**

- **Gold document restriction (oracle / two-stage):** Official FinQA and many solutions assume the **gold document is known at eval** (e.g. one question per document excerpt). The retriever’s job is then to select **supporting facts inside that document**, not to find the document in a large corpus. So the pipeline is: (1) restrict the search space to the gold document (our `corpus_id` does this), (2) retrieve within that document. Ensuring index and eval use the **same** id (e.g. `entry.get("id") or entry.get("filename")`) is standard; we added a prefix fallback when exact match gives 0 chunks.
- **Hierarchical retrieval (doc → page → chunk):** For long filings (e.g. FinanceBench, SEC), “Decomposing Retrieval Failures in RAG for Long-Document Financial QA” (arXiv:2602.17981) shows that **document-level success does not imply page/chunk-level success**. They evaluate at three levels: document recall, page recall, chunk-level overlap (ROUGE-L/BLEU to gold evidence). So “corpus_id” is the first filter (right document); then page-level or chunk-level ranking inside that document is the next step. FinGEAR uses **Item-level** (SEC section) and **dual hierarchical indices** (Summary Trees, Question Trees) so retrieval is scoped by regulatory structure. **Takeaway:** Use a single, consistent document id (corpus_id) and, for long docs, consider a second level (e.g. page or section) so retrieval is “right doc → right page/region → right chunk.”
- **Oracle analysis:** The same paper uses **oracle document** (candidates = gold filing only) and **oracle page** (candidates = gold filing + gold pages) to separate “wrong document” from “right document, wrong chunk.” That diagnostic is directly applicable: if with oracle document your accuracy jumps, the bottleneck is document scoping; if it doesn’t, the bottleneck is within-document retrieval or chunking.

---

#### Issue 2: Too few chunks (table or key passage not in top-k)

**What happens:** The correct document is in the index and corpus_id matches, but the chunk(s) containing the answer or table rank below the top-k, so the generator never sees them.

**How Kaggle / community handle it:**

- **Larger k when scoped to one document:** When the search space is already restricted to one doc (our corpus_id case), top solutions retrieve **more** chunks (e.g. 20–30+) so that tables and multi-hop evidence are included. We increased to up to 30 chunks when corpus_id is set.
- **Page as intermediate unit:** “Decomposing Retrieval Failures” introduces a **domain fine-tuned page scorer**: rank **pages** first, then retrieve chunks only from top pages. That way, even with limited top-k chunks, those chunks are drawn from the right pages. For FinQA we don’t have page ids in the index today, but we could add a “page” or “region” in metadata and do two-phase: (1) score regions/pages, (2) retrieve chunks only from top regions.
- **Hierarchical indices (FinGEAR):** Dual hierarchical indices (e.g. Summary Tree, Question Tree) and **two-stage cross-encoder reranking** improve recall; they report large F1 gains over flat retrieval. So “too few chunks” is often addressed by **better ranking** (reranker, page-level or section-level scoring) rather than only increasing k.
- **Oracle page / chunk:** The paper shows that oracle page retrieval (candidates = gold doc + gold pages) gives an upper bound on “if we had perfect within-doc retrieval.” The gap between oracle document and oracle page tells you how much is lost at page/chunk level; the gap between oracle page and standard retrieval tells you how much is lost to ranking/chunking.

**Takeaway:** For “too few chunks”: (1) use a larger k when scoped to one doc; (2) consider page- or section-level scoring before chunk retrieval; (3) use reranking (we have BGE reranker); (4) use oracle document/page metrics to see where the gap is.

---

#### Issue 3: Chunking (table split across chunks, structure lost)

**What happens:** The table is in the right document but is split across chunks by a generic sentence/token splitter, or flattened so that row/column structure and units are lost. The model then doesn’t see a coherent table or the right cells.

**How Kaggle / community handle it:**

- **Cell-level vs row-level retrieval (FinQA):** Published work on FinQA (e.g. 2206.08506) shows that **retrieving full table rows** adds noise: cells in the same row share context (row name, header, value), so unrelated cells in that row can hurt the generator. A **cell-level retriever** that retrieves only **gold cells** (cells that appear in the annotated program and table rows) reduces noise and improves execution accuracy (e.g. 69.79% on FinQA private test). So the design choice is: either chunk so that **table rows or cells are retrievable units** and train/select at that granularity, or keep larger chunks but bias retrieval toward table/cell content.
- **Table as a single unit / structure-preserving chunking:** MultiFinRAG, SQuARE, and financial chunking papers (e.g. “Financial Report Chunking for Effective RAG”) keep **tables intact** (one chunk per table or per coherent table block) and preserve **headers, units, and row/column alignment**. Some use larger chunk sizes (e.g. 1024 tokens, 128 overlap) for financial RAG so that tables are less often split. So: **chunk on sentence boundaries**, avoid splitting mid-table, and either keep the full table in one chunk or chunk by rows with the header repeated.
- **TAT-QA hybrid context:** TAT-QA stresses that tables and **associated paragraphs** must be processed together; understanding a cell often requires the surrounding text. So chunking should keep “table + its describing text” as a unit where possible, rather than splitting table and text into unrelated chunks.
- **gold_inds / supporting facts:** FinQA provides **gold_inds** (indices of supporting text/table spans). Top pipelines use these to train retrievers (positive = gold spans, negative = other spans from the same doc) or to **diagnose** whether the failure is retrieval (gold not in top-k) vs generation (gold in top-k but wrong answer). We don’t train yet, but we can log whether the gold evidence would be in our top-k (e.g. by embedding gold text and comparing to retrieved chunk set) to separate chunking/retrieval issues from generator issues.

**Takeaway:** For chunking: (1) keep tables as single chunks or chunk by table rows with header; (2) preserve headers and units; (3) consider cell-level or row-level retrieval instead of arbitrary text chunks; (4) use gold_inds / gold evidence to measure retrieval recall and to separate retrieval vs generation failures.

---

#### Summary table

| Our issue | Kaggle / community approach | What we did / can do |
|-----------|-----------------------------|----------------------|
| **Corpus_id mismatch** | One document id per question; hierarchical doc → page → chunk; oracle doc/page analysis | Use `id` or `filename` consistently; prefix fallback; add page/section later for long docs |
| **Too few chunks** | Larger k when scoped; page-level scoring; hierarchical indices; reranking; oracle metrics | Increased to 30 chunks when corpus_id set; BGE reranker; optional page-level phase |
| **Chunking** | Cell-level retrieval; table as unit; structure-preserving chunking; gold_inds for diagnosis | Keep table intact or chunk by rows; preserve headers/units; consider gold_inds recall |

---

### 9. Adversarial / distribution gap (Kaggle FinQA)

**Idea:** FinQA train vs test can have distribution shift. Top Kaggle solutions used adversarial validation to detect and mitigate this; ensembles of models also helped (71.93% execution accuracy).

**What we can do:**
- **Don’t overfit to train:** If we ever tune hyperparameters (e.g. top_k, thresholds), use a held-out dev set or the official FinQA test split for reporting.
- **Ensemble:** If we have multiple retrievers or multiple program parsers, we could combine answers (e.g. majority vote on the executed number); only worth it once the single-pipeline ceiling is clear.

---

## Prioritized Next Steps for Our Pipeline

1. **Low effort, high impact**
   - Add **1–2 few-shot FinQA examples** (query → program → answer) to the generator prompt.
   - **List “Numbers you may use”** in the prompt from retrieved chunks (regex or simple extraction) to focus the model on the right operands.
   - Ensure **table facts are not split mid-row** when building the corpus; one chunk per table or per coherent table block.

2. **Medium effort**
   - **Two-phase retrieval:** Retrieve text first; if weak, add table-tagged chunks with a lower similarity threshold. Requires tagging chunks by type when building the index (from FinQA gold “table” vs “text”).
   - **Number-aware re-ranking:** After retrieval, boost or re-rank chunks that contain numbers when the query is numerical (e.g. “total”, “percentage”, “how much”).

3. **Larger effort (when aiming for SOTA)**
   - **Retriever fine-tuning:** Number-aware negative sampling (APOLLO) or contrastive training on (query, gold_fact, negative_facts) using FinQA train.
   - **Semantic chunking:** Sentence-boundary chunking + similarity-based merging before indexing to avoid fragmented numerical context.
   - **Structured table indexing:** Store tables as structured JSON or markdown and index table summaries + cell values; use modality-aware thresholds (MultiFinRAG-style).
