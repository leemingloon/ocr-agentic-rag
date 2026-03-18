# RAG Lessons: TAT-QA

Lessons learned from TAT-QA RAG evaluation and fixes. Use for interviews and future tuning.

---

## 1. Multi-account balances (Topic 606)

- **Always use the right column.** When the query asks for balances **"without adoption of Topic 606"**, use **"As Adjusted - Without Adoption of Topic 606"** or **"Balances without Adoption of Topic 606"** from the documents for each account. Do not use "As Reported" when the question specifies without Topic 606.
- **Units.** When the query says "in millions", use or convert to millions (e.g. divide by 1,000 if values are in thousands). Round to one decimal. Output only numeric values; no units or commentary.

---

## 2. “Respectively” vs sum

- **Per-account when "respectively".** If the query contains **"respectively"**, **"each"**, or **"per account"**, return **each account’s balance separately** in the order asked (comma-separated). Do **not** sum in that case.
- **Sum only when asked.** Sum only when the query explicitly asks for "total", "combined", or "sum" of the accounts. Otherwise, when multiple accounts are listed without "respectively", return per-account values unless the context clearly implies a total is expected.

## 2b. Preserve source formatting (gold-blinded)

- **Exact match needs exact format.** For "respectively" (and per-account) answers, numeric exact-match evaluation compares strings. So return values **exactly as in the source**: preserve **commas** in thousands (e.g. `1,568.6`) and **decimals** as stated (e.g. `690.5`). Do not round to integers or strip commas (e.g. `1568.0` ≠ `1,568.6`).
- **Implementation (gold-blinded).** Extract the substring after the adjustment label (e.g. "Balances without Adoption of Topic 606: ") with a regex that captures the raw number including commas and decimals, e.g. `([0-9,]+(?:\.[0-9]+)?)`. Use that string as the answer segment; do not parse to float and re-format. Assemble the final answer as comma-separated values in query order, e.g. `1,568.6, 690.5`.

---

## 3. Percentage of adjustment

- **Intent.** When the query asks for the **percentage** (or **percent**) **of adjustment** to an "as reported" balance (e.g. prepaid expenses), the expected answer is a percentage in 0–100, not a raw fraction.
- **Formula.** From the table/text: find **As Reported** and **Adjustments** (parentheses = negative). Compute:
  - `(|adjustment| / as_reported) × 100`
  - Round to one decimal (e.g. **17.7**).
- **Example.** As Reported: 93.8, Adjustments: (16.6) → adjustment = −16.6 → 16.6/93.8 × 100 ≈ **17.7%**.
- **Implementation.** In the RAG pipeline, if the program returns a fraction (e.g. `divide(-16.6, 93.8)` → −0.177) and the query contains "percent" or "percentage", convert to percentage: `round(abs(result) * 100, 1)` before scoring.

---

## 4. Percent change (unified formula and adjusted vs reported)

- **Always use (new - old) / old.** Do not branch on direction (e.g. "decrease → (old-new)/new"). The sign of the result encodes increase/decrease. Never divide by **new**; the denominator is always **old**.
- **"After being adjusted" / "after adjustment" / "balance after adjustment"**: For the **same line item** (e.g. Inventories), use **old = As Reported**, **new = Balances without Adoption of Topic 606**. Do **not** use FIFO/LIFO hypotheticals or other narrative numbers (e.g. "would have been $178.4 million and $210.3 million higher"); use the table columns **As Reported** and **Balances without Adoption of Topic 606** for that row.
- **Year-over-year**: "from 2018 to 2019" → old = earlier year, new = later year, then (new - old) / old.
- **Output**: TAT-QA percent change in 0–100 scale, round to 1 decimal (e.g. -0.2). Program: subtract(new, old), divide(#0, old), multiply(#1, 100).

---

## 5. Parentheses as negative

- In financial tables, amounts in **parentheses** (e.g. (16.6)) denote **negative** values. When parsing or computing (e.g. adjustment / as reported), treat (16.6) as −16.6. The percentage of adjustment should use the **absolute** value for the magnitude (e.g. 17.7%), or the sign can be preserved for "increase" vs "decrease" if the question asks for direction.

---

## 6. Scale and format

- **Multi-value answers.** For "respectively" queries, preserve source formatting (see 2b) so the prediction string matches the reference (e.g. `1,568.6, 690.5`). When the reference is a single value but the model returns multiple, scoring may accept the **first** value as a match depending on evaluation rules.
- **Index vs evaluation split.** Build the TAT-QA retriever index from the **same split** used for evaluation (e.g. test-only) so that retrieval can surface the document that contains the gold answer. Mismatch (e.g. index from train+dev, eval on test) leads to "GT in_index=False" and retrieval misses.

---

## 7. Retrieval (recall) and fallbacks

- **Reranker.** Table-style finance QA benefits from cross-encoder reranking. Only skip when explicitly set (`RAG_SKIP_RERANKER=1`); do not disable by default for debug.
- **Retrieval depth.** Default `top_k` is 15 (overridable with `RAG_TOP_K`). Table rows often chunk separately; top-5 is too small and causes "GT in_index=True in_retrieved=False".
- **Query rewriting.** For "percent change" + "adjusted" (or "after being adjusted"), the embedding query is rewritten to append "As Reported Balances without Adoption Topic 606" so retrieval is more likely to return the right table chunks.
- **No 0.0 fallback.** When no program is executed and the growth-rate fallback would return 0.0 (or no fallback), do **not** output 0.0 as the numerical answer. Output **INSUFFICIENT_DATA** so the failure is attributed to retrieval/context, not a wrong number.

- **Document isolation (corpus_id).** TAT-QA questions are document-specific. Use **corpus_id** from the sample (per-doc table uid) so retrieval is restricted to that document only. Global retrieval (corpus_id=None) causes cross-document contamination: the model sees totals from other docs (e.g. 16,503 / 6,390) and uses them instead of the gold doc’s component sums (e.g. 831.7 + 1,571.7 + 93.8) / 691.6 = 3.61. The adapter must set **ground_truth.corpus_id** for each TAT-QA sample so the eval runner passes it to RAG and retrieval ranks only within that document.
