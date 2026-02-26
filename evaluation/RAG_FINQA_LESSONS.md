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

## 9. Debug Run: Avoid Double-Loading (Segfault)

**Issue:** With `--debug --category rag`, after the dataset run the runner also ran **adversarial** tests, which created a **new** retriever and **new** reranker. Loading the large reranker (and embeddings) twice in one process can cause OOM or **segmentation fault** on 16GB.

**Fix:** When `--debug` is set, **adversarial is skipped** so the reranker (and heavy models) are not loaded a second time. Run adversarial without `--debug` when you have enough memory.

**Takeaway:** In resource-constrained debug runs, avoid running a second heavy pipeline (e.g. adversarial) in the same process.

---

## What to Do Next

- **Re-run with fixes:** To see improved predictions and correct metrics, run without skipping: e.g. temporarily move or clear the FinQA per_sample file for the split you care about, then run again so those samples are re-evaluated (and metrics recomputed with FinQAUtils + new RAG/generator behavior).
- **Simulate before spending API credits:** Use `scripts/simulate_finqa_retrieval.py` to check what chunks the model would see for a given query and `corpus_id`.
- **Primary metric for FinQA:** Use **numerical_exact_match** (and the strict exact_match for numeric answers) when reporting or comparing runs.
