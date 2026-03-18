# RAG pipeline: Improvement roadmap

Carry-forward list of enhancements to reduce retrieval/reasoning failures (FinQA, TAT-QA). Cross-references existing code where relevant.

---

## Chunking

**Table-aware chunking** — Detect table boundaries, preserve row/column structure. Carry forward section headers as chunk metadata. Handle footnotes by appending them to their parent table chunk. Use overlap on prose but not on tables.

- **Current:** Level 3 row-level serialization with `--table_aware`. Section header in metadata via `index_preprocess.extract_section_header(context)`. Footnotes: FinQA build appends post_text footnote block to last table row chunk (`_footnote_block_from_post`). Overlap only on prose: `DocumentChunker` uses sentence splitter (with overlap) for narrative segments; table/header segments are single chunks with no overlap.
- **Next:** Optional footnote-to-specific-row attachment (when reference markers are parseable).

---

## Retrieval ranking

**Cross-encoder reranker** — Add a cross-encoder reranker after embedding retrieval. The reranker sees the full query+chunk pair jointly, so "net derivative liabilities 2013" and "collateral posted 2013" score differently even if their embeddings are close.

- **Current:** `rag_system/reranking.py` has `BGEReranker` (BAAI/bge-reranker-v2-m3). It is **skipped** when `RAG_DEBUG=1` or `RAG_SKIP_RERANKER` is set (see `eval_runner` and agentic retrieval path). Agentic flow in `rag_system/agentic/retrieval_tools.py` uses the retriever then optionally reranker.
- **Next:** (1) Enable reranker in debug runs (or a separate flag) so evaluation uses it. (2) Ensure reranker is always used in production eval; calibrate top_k after rerank. (3) Optionally add a lightweight second-stage reranker tuned for “row label + year” disambiguation (e.g. query “net derivative liabilities 2013” vs chunk “collateral posted … 2013”).

---

## Query understanding

**Query intent classifier** — Classify query intent (absolute change vs. % change vs. ratio) before formula selection. Extend the existing primer trigger system with a dedicated query intent classifier — rule-based first, then a lightweight model if patterns proliferate.

- **Current:** Dedicated query intent classifier in `orchestrator.py`: `classify_query_intent(query)` returns a list of intent labels (e.g. `absolute_change`, `percent_change`, `percent_reduction`, `table_year`, `table_total_across_columns`, `compensation`, `totals_prefer_direct`, `event_scoped`, `cashflow_financing`, `locom_growth`, `yes_no`). Primer selection is driven by intents (e.g. `needs_growth_rate = RAG_INTENT_PERCENT_CHANGE in intents`). The classifier is implemented via the existing `_needs_*_primer(query)` helpers so behaviour is unchanged; intents are logged when `RAG_DEBUG=1`.
- **Next:** (1) Optionally pass intent-based suggested formula line into the generator prompt. (2) If rules proliferate, replace the rule-based classifier with a small model trained on (query, correct_formula) from FinQA.

---

## Numerical grounding

**Missing-number detection and constants** — Explicitly detect when a required number is missing from retrieved chunks (e.g. statutory tax rate) and either prompt the model to state the assumption or inject a small structured knowledge base of standard financial constants.

- **Current:** No explicit “required number missing” detection. Model sometimes says “cannot be determined” (see row-label disambiguation primer).
- **Next:** (1) Define a small KB of standard constants (e.g. statutory tax rates by jurisdiction, risk-free rate conventions). (2) After retrieval, if the query mentions a concept (e.g. “effective tax rate”) and no chunk contains a matching number, optionally inject a one-line hint from the KB or add an instruction: “If the document does not state X, state your assumption (e.g. statutory rate 21%) or say INSUFFICIENT_DATA.”

---

## Multi-hop retrieval

**Section-tagged retrieval** — Tag chunks with their document section (income statement, balance sheet, notes) at index time. On multi-hop queries, retrieve from each required section independently, then assemble before reasoning.

- **Current:** Chunks have `corpus_id`, `source`, `entry_id`. No section/source-type tag (e.g. income_statement, balance_sheet, notes).
- **Next:** (1) At index build, label each chunk with section type (e.g. from PDF outline, or heuristics: “statement of operations” → income_statement, “note 5” → notes). (2) Store in chunk metadata. (3) In retrieval, if the query or planner indicates multiple sections (e.g. “compare note 5 to the income statement”), run retrieval filtered by section and merge results.

---

## Negative retrieval

**Relevance threshold and abstention** — Score retrieved chunk relevance before passing to the reasoning step. If max relevance score falls below a threshold, route to abstention rather than generation. Calibrate the threshold on the labeled failure set.

- **Current:** Reranker returns scores; they are not compared to a threshold. Generator always runs.
- **Next:** (1) After rerank, take `max_score = max(scores)`. (2) If `max_score < threshold`, return a structured abstention (e.g. “INSUFFICIENT_RELEVANCE: max score 0.32 below 0.5”) instead of calling the generator. (3) Tune threshold on labeled failures (e.g. from `chunking_failures.json` and numerical_exact_match=0 samples) to maximise correct abstentions without over-abstaining.

---

## Unit/scale normalisation

**Parse and normalise units** — Parse units at chunk ingestion time and store as metadata (millions, per-share, quarterly). At assembly time, normalise all numbers to a canonical unit before the model reasons over them.

- **Current:** No unit metadata on chunks. Numbers hint in the generator lists raw numbers; model reasons over mixed units (e.g. “in millions” in text).
- **Next:** (1) At chunking/index time, run a lightweight unit detector (e.g. “$ in millions”, “per share”, “quarterly”) and attach `units` to chunk metadata. (2) When assembling context for the generator, optionally normalise mentioned numbers to a canonical unit (e.g. millions) and add a one-line note: “All dollar figures in millions unless stated otherwise.” (3) Or inject unit into the numbers_hint: “35764 (millions), 22176 (millions)”.

---

## Context length / lost-in-the-middle

**Reorder chunks for importance** — After reranking, reorder chunks so the top-1 and top-2 chunks sit at the start and end of the context window, with lower-ranked chunks in the middle.

- **Current:** After rerank, `_apply_bookends_order(chunks)` in `retrieval_tools` reorders so top-1 and top-2 are at start and end: `[chunk_0, chunk_2, chunk_3, ..., chunk_1]`.
- **Next:** Optional env `RAG_CONTEXT_ORDER=bookends|top_first`.

---

## Deduplication

**Chunk dedup and provenance** — At index build time, hash chunk content and deduplicate. Track provenance (page, section, table ID) per chunk so the retriever can prefer the primary source over condensed repetitions.

- **Current:** No dedup; same content can appear in multiple chunks (e.g. table repeated in summary). Chunk metadata has `corpus_id`, `entry_id`, `source`.
- **Next:** (1) At index build, compute a content hash per chunk; skip or merge duplicates (keep one, record “duplicate_of” or count). (2) Add `page`, `section`, `table_id` to metadata when available. (3) At retrieval, optionally prefer chunks with `table_id` or “primary” section when scores are tied.

---

## Summary table

| Area                | Status / existing hook                              | Next step |
|---------------------|-----------------------------------------------------|-----------|
| Chunking            | Table-aware, section_header, footnotes, prose overlap | Footnote-to-row attachment |
| Reranker            | Wired in agentic path; rerank_with_scores            | Tune top_k; optional 2nd-stage |
| Query intent        | `classify_query_intent` → primer selection          | Suggested formula line; optional model |
| Numerical grounding | KB + detect_missing_constant; hint in prompt        | Extend KB; program injection |
| Multi-hop           | section_type filter; infer_section_types_for_query   | Planner-driven; per-section top-k |
| Negative retrieval  | RAG_RELEVANCE_THRESHOLD; abstain in generator        | Calibrate on failures |
| Units               | Ingest metadata; assembly note                       | Arithmetic normalisation |
| Lost-in-middle      | Bookends order in retrieval_tools                    | Optional RAG_CONTEXT_ORDER |
| Dedup               | Hash, dedup, provenance; prefer primary              | duplicate_of pointer |
