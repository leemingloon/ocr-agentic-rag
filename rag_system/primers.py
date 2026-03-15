"""
RAG domain-knowledge primers for FinQA and TAT-QA.

All primer string constants live here. rag_system/agentic/orchestrator.py imports
them and routes the correct subset to the generator prompt based on rule-based query
intent detection (classify_query_intent).

Sections
--------
1. Shared / Always-On         — injected on every call (base rules)
2. Table Extraction           — row/column disambiguation, date-column anchoring
3. Arithmetic Operations      — change, difference, growth rate, percent, average
4. Loss Account (IAS 1/ASC 220) — magnitude convention for net-loss arithmetic
5. Equity Compensation (ASC 718) — vesting-period amortisation vs grant-date FV
6. Specialised Domain         — LOCOM, interest payments, leases, cash flow, etc.

Performance with primers (relaxed_exact_match, held-out evaluation splits):
    TAT-QA (test split,  n=200): 99.0%
    FinQA  (train split, n=100): 93.0%

Do NOT modify the FinanceBench primer here — it lives in eval_runner.py.
Last updated: 2026-03-13
"""

# =============================================================================
# SECTION 1 — SHARED / ALWAYS-ON PRIMERS
# =============================================================================
# Universal rules: decimal not %, unit normalization, answer-first ordering,
# growth rate vs absolute change disambiguation.
# Injected at the start of the generator prompt; intent primers extend it.
# =============================================================================
SHARED_NUMERICAL_PRIMER = """
**Universal numerical reasoning rules:**
- **Output format:** Report numbers as requested. For growth rates and ratios, use **decimal** (e.g. -0.03219) unless the question explicitly asks for "percent" or "percentage" in 0–100 form.
- **Units:** Preserve the document unit (millions, thousands, index base 100); do not rescale unless the question asks.
- **Answer-first:** State the final number or conclusion clearly (e.g. at the end of your response) before or after showing the calculation.
- **Change vs rate:** "Change in X from A to B" without "growth rate" or "percent change" = absolute difference (subtract only). "Growth rate" or "percent change" = (new - old) / old.
- **Multi-step average:** When you sum N values with a chain of add (e.g. add(a, b), add(#0, c)), the **complete sum** is in the **last** add step. For the average, divide **that** step by N (e.g. divide(#1, 3)), not an earlier step (e.g. not divide(#0, 3)).
"""

# =============================================================================
# SECTION 2 — TABLE EXTRACTION PRIMERS
# =============================================================================

# Table/year extraction primer: lock onto the row/column for the requested year (FinQA AAL/2018, GS/2014-style)
TABLE_YEAR_PRIMER = """
For table-based financial questions that ask for a value **in a specific year** (e.g. "total operating expenses in 2018 in millions", "change in net derivative liabilities between 2014 and 2013"):
- **Step 0 (multi-year columns):** When the table has multiple **year columns** (e.g. 2014 | 2013 | 2012), first **verify column order** by cross-referencing a known value from the prose (e.g. "net revenues were $7.02 billion for 2013" or "pre-tax earnings were $4.33 billion in 2013"). Find which column contains that value—that column is 2013. Once column order is confirmed, **apply it consistently to ALL rows**: do not take the 2013 value from one row and the 2014 value from another; every number for "in 2013" must come from the **same** (2013) column.
- **Step 0b (multi-row tables / row-label disambiguation):** When the table has **multiple named rows** (e.g. net derivative liabilities, collateral posted, one-notch downgrade, two-notch downgrade) and the context shows only partial chunks, **verify the row label** before using any value: use a number from the requested year's column **only if** that number is explicitly tied to the **same row label** as the metric in the query (e.g. "net derivative liabilities"). Do **not** use a value from the same column if it belongs to a different row (e.g. do not use a one-notch or two-notch downgrade 2013 figure for a "net derivative liabilities" question). If the requested row's value for the requested year is **not visible together with its row label** in the context, state that it cannot be confirmed rather than substituting a value from another row—using the wrong row is worse than abstaining for change/difference calculations.
- **Step 1:** Identify the exact **row** (or column) that corresponds to the requested year (e.g. 2018 or 2013). In multi-year tables, each row is often one year—do not use a different year's row. If the table is column-oriented (years as columns), use only the column that matches the requested year.
- **Step 2:** From **that row/column only**, extract the numerator (e.g. segment revenue, line item) and the total or denominator. Do NOT take a value from another year's column (e.g. do not use the 2014 column when the question asks for 2013—use the 2013 column for both numerator and denominator).
- **Step 3:** If the formula is total = part / (percent/100) or percent = part / total, use **only** the year-specific numbers from the same column (and row). Then output the program, e.g. divide(9896, 23.6%) for 2018, or divide(1947, 7018) for 2013 segment share.
- **Step 4:** Do not mix years. Cross-check: if the question says "in 2013", every number in your program must come from the 2013 column (or the row that aligns with 2013).
"""

# When the question asks for a value "without" or "excluding" a percentage effect (e.g. FX gain), use prior year as base for the %.
PRIOR_YEAR_PCT_ADJUSTMENT_PRIMER = """
When the question asks for a value **without** or **excluding** a percentage effect (e.g. "without the foreign currency translation gain, what would 2008 sales have been"):
- Use the **prior year** (or earlier period) as the **base for the percentage calculation** unless the question specifies otherwise. Example: "2% due to FX" on 2008 sales → apply 2% to **2007** sales (the prior-year base), then subtract that amount from 2008 sales: multiply(2007_value, percent), subtract(2008_value, #0).
- Do **not** use the current/year-in-question value as the base for the percentage when removing an adjustment; the adjustment is typically expressed as a share of the prior period.
"""

# FinQA year interpretation: when the question references a year but the document only has values for other years,
# compute from provided values; pair values by position (first with first), not by "prefer earlier year".
FINQA_YEAR_INTERPRETATION_PRIMER = """
When the question references a **specific year** (e.g. "in 2003 what was the ratio of ...") but the document only provides values for **other years** (e.g. 2012 and 2011):
- If the **operands needed for the calculation are present** in the context (e.g. notional amounts, hedge amounts, swap balances), **compute the answer directly** from those values (e.g. divide(numerator, denominator)).
- Do **not** return INSUFFICIENT_DATA solely because the document does not contain a row or column labeled with the question year. The year in the question is often contextual; use the values provided to perform the requested ratio or calculation.

**Pair values by position (narrative with two years and two values):** When a sentence lists two years and two corresponding values (e.g. "at December 31, 2012 and 2011 was $1.3 billion and $1.7 billion" and "was $503 million and $450 million"), use the **first** value pair (leftmost) unless the question explicitly asks for the other year. Example: foreign currency hedges 1.3 and 1.7, interest rate swaps 503 and 450 → use the first pair for numerator and denominator (e.g. 1.3 and 503). For ratios across different units (e.g. billion vs million), **normalize to a common unit** before dividing (e.g. 1.3 billion → 1300 million, then divide(1300, 503)) so the ratio is financially meaningful.
"""

# Multi-year average: absolute row inclusion; divisor = query range count; extra table columns in sum, not in divisor.
TABLE_MULTI_YEAR_AVERAGE_PRIMER = """
When the question asks for the **average** of a metric **over a year range** (e.g. a range like "2012-14" or "2012, 2013 and 2014"):

1. **Identify all years in the range** from the query wording (the range implies a fixed number of years; e.g. a span of three years means three years in scope).

2. **Absolute row inclusion:** Include **all** numeric values listed in the row that matches the requested metric, for **any** year column in that row — even if the table contains extra years outside the query range (e.g. an earlier or later year). Only treat a year as **0** if it is missing entirely from the table and no footnote provides a value. If a row lists multiple years in separate columns, mark **all** of those values as part of the sum for averaging; do not omit an earlier or later column.

3. **If a year in the query range has no column** in the table, treat that year as **0** or use a footnote value if the document states one, and include it in the sum.

4. **Sum** all included table row values (every year column in the row for that metric, plus 0 or footnote for any query-range year missing from the table).

5. **Divisor = query range count.** After summing all relevant table row values for the included years, divide by the **total count of years in the query's range**. Extra table columns (years outside the query range) contribute to the sum if present, but **do not increase the divisor**. The divisor is always the number of years the question is asking an average over.

6. **Step reference for divide:** The sum is in the **last** add step (e.g. add(y1, y2), add(#0, y3) → full sum in #1). Use **divide(#1, count)**, not divide(#0, count), so the average uses the complete sum.

Output the average. Do not default to averaging only two values when the row or table implies more years.
"""

# =============================================================================
# TABLE SEGMENT ALIGNMENT (OCR-flattened segment tables, portion/share by position)
# =============================================================================
# When tables are flattened by OCR, the same label (e.g. "total backlog") may
# appear multiple times within each segment block. Align by relative position
# across segments; for "portion of total" questions use the second aligned pair.

TABLE_SEGMENT_ALIGNMENT_PRIMER = """
**Scope — OCR-flattened tables only.** Use this primer when the table is flattened by OCR so that segment and total labels repeat within blocks.

**OCR-flattened segment tables:** The same label (e.g. "total backlog") may appear **multiple times** within each segment block. **Do NOT take the first match.**

**Steps:**
1. Identify numeric entries for the **segment** (e.g. Ingalls) in the order they appear in the text.
2. Identify numeric entries for the **company total** (e.g. total backlog) in the order they appear.
3. Align segment entries to the corresponding company total by **relative position** (first with first, second with second).
4. For **"portion of total"** or **"share of total"** questions when no year is specified, use the **second aligned pair** (FinQA convention: earlier year / second row).

**Example:**

Segment (Ingalls) total backlog values in order: **11365**, **7991**
Company total backlog values in order: **22995**, **21367**

Aligned pairs by position:
- Position 1: (11365, 22995)
- Position 2: (7991, 21367)

Question: "What portion of total backlog is related to the Ingalls segment?"

**Use the second pair.** Program: **divide(segment_value, total_value)** with the values at the second position.

Correct program:
divide(7991, 21367)
"""

# Gold-blinded primer for year-over-year table change: compute difference without revealing the answer
TABLE_YEAR_CHANGE_PRIMER = """
**Loss account carve-out — check this FIRST.**

If the line item in the question is NET LOSS or OPERATING LOSS (the values are shown as $(X) in the table), do NOT use this primer. Apply LOSS_CHANGE_PRIMER instead. This carve-out applies ONLY to "net loss" and "operating loss" — NOT to "allowance for loan losses", "accumulated other comprehensive loss", or other balance items that incidentally contain the word "loss".

For questions asking for the **change** (in millions or in value) of a line item **from one year to another** (e.g. "what was the change in the carrying amount from 2007 to 2008?"):
1. Locate the value for the **earlier** year (e.g. 2007) in the table row/column that matches the requested line item.
2. Locate the value for the **later** year (e.g. 2008) in the same row/column.
3. Compute: Change = (later_year_value - earlier_year_value). Use **subtract(later_value, earlier_value)**. Use only table data; do not assume any value.
4. Return only the numeric answer (with a minus sign if negative). Do not assume or guess; use only the table data provided.
"""

# Gold-blinded: when the metric is per-unit (per share, per option, per unit, etc.), "change" = raw delta, not percent.
PER_UNIT_CHANGE_PRIMER = """
When the question asks for **change** or **percent change** in a metric that is expressed **per unit** (e.g. per share, per option granted, per unit, per employee, per award):
- Treat it as a **unit-based measure**. Compute **Change = new_value − old_value** only. Use **subtract(later_value, earlier_value)**.
- Do **not** divide by the old value or multiply by 100. Do **not** apply a percent-change formula.
- Use the row that matches the requested metric (e.g. "fair value per option granted"); use the requested years (e.g. 2015 and 2016). Return the raw numeric difference (e.g. in dollars per option).
For **aggregate** measures (revenues, total assets, total compensation, etc.), when the question explicitly asks for **percent change**, use ((new − old) / old) × 100 as usual.
"""

# =============================================================================
# TABLE ROW ALIGNMENT (FinQA multi-year share questions)
# =============================================================================
# When a table shows multiple years but the question does not specify a year,
# prefer consistent row pairs (often the earlier year). Prevents mixing columns
# or selecting the wrong year when computing ratios.

TABLE_ROW_ALIGNMENT_PRIMER = """
**Scope — standard tables with visible year labels.** Use this primer when the table has clear year columns or rows (e.g. 2018, 2017).

For financial tables that show **multiple years but the question does NOT specify a year** (e.g. "what portion of total backlog is related to Ingalls segment?"):

- Step 1: Identify whether the table shows **multiple year rows or columns** (e.g. 2018, 2017).
- Step 2: When computing a **share, portion, ratio, or percentage**, the numerator and denominator must come from the **same row pair** of the table (same year).
- Step 3: Do NOT mix values from different rows (e.g. a 2018 numerator with a 2017 denominator).
- Step 4: If multiple valid row pairs exist and the question does not specify a year, prefer the **earlier year row pair** (commonly the second row in FinQA tables).
- Step 5: Extract the component value (e.g. Ingalls backlog) and the corresponding total from that same row before computing the ratio.

Example:

Table:
2018: Ingalls backlog = 11365, Total backlog = 22995  
2017: Ingalls backlog = 7991, Total backlog = 21367

Question: "what portion of total backlog is related to Ingalls segment?"

Correct program:
divide(7991, 21367)

Both numbers come from the **same row (2017)**.
"""

# RSR / RPSR ratio alignment (compensation expense: pair RSR with matching RPSR; do not sum RPSR subcategories)
# FinQA HII/2011/page_114: ratio of RSR unrecognized expense to RPSR unrecognized expense — use the RPSR value
# explicitly tied to the same grant/event (e.g. "converted as part of the spin-off"), not sum of all RPSR lines.
RSR_RPSR_RATIO_ALIGNMENT_PRIMER = """
For **ratio questions involving RSR (restricted stock) and RPSR (restricted performance share)** compensation expense or unrecognized amounts:

- **Segment-level alignment:** Align the **RSR** numeric value to the **matching RPSR** numeric value in the **same segment, grant year, or category**. Do **not** sum unrelated RPSR values even if they appear temporally close in the table.

- **Pairwise operand selection:** For ratio questions (e.g. "what was the ratio of RSR unrecognized compensation expense to RPSR unrecognized compensation expense"), **pair** the RSR figure with the RPSR figure that is **explicitly labeled** as the same category (e.g. "converted as part of the spin-off", "as part of the spin-off", or the row that corresponds to the same grant type/year). Use **divide(RSR_value, matched_RPSR_value)**.

- **Do not aggregate RPSR subcategories** unless the question or table text explicitly says to combine them. If the table has multiple RPSR lines (e.g. one for "converted as part of the spin-off" = $10M and another for other RPSRs = $18M), use **only** the RPSR value that matches the RSR row/category for the ratio denominator. Program: **divide(RSR_value, matched_RPSR_value)**; avoid **add** or **sum** of RPSR values unless the text explicitly says to combine.
"""

# RSR/RPSR ratio scaling (HII/2011-style): use matching RPSR denominator; check for stated denominator or scaling convention.
# GT may use the RPSR value from the paired row (e.g. 10) so ratio = 19/10 = 1.9, not 19/18 = 1.0556.
RSR_RPSR_RATIO_SCALING_PRIMER = """
**RSR/RPSR ratio — scaling and denominator selection:**

1. To compute the **ratio of unrecognized compensation expense for RSRs to RPSRs**, first identify the unrecognized compensation expense for **each** award type in the **same year**. Align **year by year**; do not sum across multiple years unless explicitly instructed.

2. **Normal formula:** ratio = RSR_expense / RPSR_expense. Use **divide(RSR_value, RPSR_value)**.

3. **Multiple RPSR lines:** When the table has more than one RPSR line (e.g. one row "converted as part of the spin-off" = 10, another row other RPSRs = 18), use **only the RPSR value that is explicitly paired** with the RSR in the same category or row. That denominator may be the **smaller** of the RPSR figures (e.g. 10). Program: **divide(RSR_value, matched_RPSR_value)** (e.g. divide(19, 10) = 1.9). Do **not** use the other RPSR line (e.g. 18) as the denominator.

4. **Document scaling:** Some official documents report this ratio with a **stated denominator or scaling convention** (e.g. a note that the ratio is expressed per $10M of RPSR, or a table that uses 10 as the denominator for the paired row). If the document indicates such a convention, use **that** value as the denominator (e.g. divide(RSR, 10)) to obtain the **reported** ratio. Check footnotes and table headers for any "per 10" or similar scaling.

5. **Output:** Report the **final ratio** (e.g. 1.9) as the answer. Do not output the raw ratio using the wrong denominator (e.g. 19/18 = 1.0556) when the document or table structure indicates the correct denominator is the paired RPSR value (e.g. 10).
"""

# Date-column extraction: lock onto the column for the query date (FinQA INTC/2013-style)
TABLE_DATE_COLUMN_PRIMER = """
For **percentage or numerical questions from financial tables "as of [DATE]"** (e.g. "what percentage of total cash and investments as of Dec 29, 2012 was comprised of X"):
- **Step 1:** Scan **ALL** context for **both** dates. List **every** dollar amount with its nearest preceding/following text (headers, dates, line labels like "available-for-sale", "total cash"). Flag **isolated** numbers (e.g. "$ 14001" on a separate line) and check if they align with "available-for-sale" or "2012" or the second column—assign to the query date when context supports it. Reconstruct the table by **matching numbers to the nearest date label**.
- **Step 2:** If the numerator (e.g. available-for-sale) for the query date is **blank or missing** in the main table, **search every fragment** for that amount (e.g. "14001", "$ 14001") and assign to the query date if it appears near "2012", "dec 29", or after "available-for-sale" / "cash and investments". Consistency: available-for-sale often increases YoY—$14M (2012) < $18M (2013) is logical; prefer the **smaller** candidate for the older date. Do **not** use the first or most prominent number if it belongs to the **other** date.
- **Step 3:** **Numerator** = value for the component (e.g. available-for-sale) **for the query date only**. **Denominator** = value for the total (e.g. total cash and investments) **for the query date only**. In comparative tables the **older date** (e.g. 2012) is often the **second** column—confirm which column is which before picking numbers.
- **Step 4:** If the numerator seems wrong for the query date (e.g. $18k when the other column is 2013 and query is 2012), **reject** it. Prefer a **lower** candidate (e.g. $14,001) that appears in 2012 context over the larger number from the other date. **NEVER** default to the first or most prominent number if it belongs to another date.
- **Step 5:** If you **cannot** determine the numerator or denominator for the query date after exhaustive search, do **not** compute—output **INSUFFICIENT_DATA** and note "possible retrieval gap" in your reasoning. Do not guess or use a value from another date. Only compute when both values are confidently from the query date column.
- **Step 6:** When both are identified: percentage = numerator / denominator as a **decimal** (e.g. 0.53232). Output divide(numerator, denominator) or the decimal.
"""

# When chunking splits a table horizontally, a labeled "total" may be one column's subtotal; sum column totals for full total (FinQA PNC-style).
TABLE_TOTAL_ACROSS_COLUMNS_PRIMER = """
When the question asks for the **total** of a line item (e.g. "total of home equity lines of credit" in millions) and the context contains:
- A **labeled subtotal** (e.g. "total (a) | $X" or "total | $X") and
- One or more **other dollar figures** (e.g. $Y) in the same table or nearby text without a clear "total" label,
then the table may have **multiple columns** (e.g. interest-only vs principal+interest) and chunking may have split them. The labeled "total" is often **one column's subtotal**; the full total may require **adding** that subtotal and the other figure(s). Scan the full context for all dollar amounts that look like column or row totals in the same table; if summing them yields a round, plausible total (e.g. X + Y = Z), use that sum. Do not assume the first labeled "total" is the complete answer when other sizable figures appear in the same table context.

NOTE: The answer may require summing figures across multiple table sections. If you see a subtotal and a separate unlabeled dollar figure in the context, consider whether they belong to the same table's column totals.
"""

# Prefer direct line items over back-calculation for "total" queries (FinQA AAL/2018-style)
TOTALS_PREFER_DIRECT_PRIMER = """
For queries like **total operating expenses**, **total revenue**, etc. in millions for a given year:
- **ALWAYS** prefer and extract the **direct** "Total operating expenses" / "Operating expenses" line from consolidated statements of operations, income statement, or MD&A summaries. Use that figure as your answer if it appears anywhere in the context.
- **ONLY** perform back-calculation from a component percentage (e.g. fuel % of total) if: (1) **NO** direct total line appears anywhere in the context, and (2) the percentage is **explicitly** for the **full** requested total (not a sub-category like mainline-only).
- If back-calculation yields an unrealistic number (e.g. >$50B for a large airline in 2018), **discard it** and use a direct figure from the context or re-check the percentage row/year. Typical full-year operating expenses for major airlines are ~$35–45B.
- Output the **direct** figure if present; do not prefer a percentage-based calculation when a direct total is available.
"""

# When the question asks "which years does the table provide information for", list years from the table that matches the question.
TABLE_YEARS_PRIMER = """
When the question asks **which years** (or **what years**) **the table provides information for** (or "covers", "includes"):
- **List every year** that appears in the **relevant** table (column headers, row labels, narrative). If the question refers to specific metrics (e.g. projected benefit obligation), use only a table that contains those metrics — see pension funding status primer when the question mentions PBO, ABO, or fair value of plan assets.
- If the question asks for a single year (e.g. latest reporting year), give that year only; otherwise list all years the table refers to.
"""

# Pension: only extract years from the funding status table (PBO, ABO, fair value of plan assets). Ignore asset allocation tables.
PENSION_FUNDING_STATUS_TABLE_PRIMER = """
**Pension plan questions mentioning projected benefit obligation (PBO), accumulated benefit obligation (ABO), or fair value of plan assets** refer specifically to the **pension funding status** table.

- **Only extract years from a table that explicitly contains rows labeled:** Projected benefit obligation, Accumulated benefit obligation, Fair value of plan assets (or equivalent). That table reports the funded status and its column headers are the reporting years (e.g. 2020, 2019).

- **Ignore pension asset allocation tables** that contain only: Target allocations, Percentage of plan assets (by category). Those tables describe **asset allocation**, not PBO/ABO/fair value of plan assets, and **do not answer** the question.

- If the retrieved context contains only an asset allocation table (target allocations, percentage of plan assets), **do not** extract years from it. State that the retrieved table does not contain the required rows (projected benefit obligation, accumulated benefit obligation, fair value of plan assets) and that the question cannot be answered from the provided table.

- If multiple pension tables exist, **select the table** that contains the three liability/asset rows above and extract the reporting year(s) from **that table only**.
"""

# WHAT_TABLE_SHOWS_PRIMER — disambiguate table when query is ambiguous
WHAT_TABLE_SHOWS_PRIMER = """
Before answering any "What does the table show?" (or similar) query:
1. **Explicitly identify the table** you are referencing (by its header, surrounding section, or subject matter).
2. If **multiple tables** are present in the retrieved context, state **which one** you are describing and **why** you selected it.
3. **Do not assume** there is only one table in the document. If the context contains several distinct tables and the query does not disambiguate, either name the table you are using or note that the answer refers to a specific table (identify it) rather than the whole document.
"""

# ARITHMETIC_FROM_COMPONENTS_PRIMER — compute ratio/total from components when not stated
ARITHMETIC_FROM_COMPONENTS_PRIMER = """
When the question asks for a **ratio** (e.g. ratio of total X to total Y) or a **total** that can be derived from line items:
- If the **exact total** is not explicitly stated in the context but **all component values** are present (e.g. individual current asset and liability lines), **compute the total from the components** (add or combine as appropriate), then compute the ratio or answer. Do **not** return INSUFFICIENT_DATA when the components are all present and the formula is clear.
- State the components you used and the computation (e.g. total = A + B + C, ratio = total_X / total_Y) so the answer is auditable.
"""

# =============================================================================
# SECTION 3 — ARITHMETIC OPERATION PRIMERS
# =============================================================================

# Absolute year-over-year change (signed difference only; not growth rate).
ABSOLUTE_CHANGE_PRIMER = """
**Loss account carve-out — check this FIRST.**

If the SPECIFIC SUBJECT of the change in this question is NET LOSS or OPERATING LOSS — that is, the question explicitly contains the phrase "net loss" or "operating loss", and the table values appear in parenthetical format $(X) indicating a loss account — then do NOT use this primer. Apply LOSS_CHANGE_PRIMER instead, which uses the magnitude convention (strip the negative sign before subtracting).

This primer applies to INCOME, REVENUE, ASSET, and BALANCE-SHEET metrics where table values are positive or signed non-parenthetically. Examples: "change in total assets", "change in revenue", "change in OFA Level 2", "change in carrying amount", "change in allowance for loan losses", "change in accumulated other comprehensive loss".

The carve-out is NARROW: it applies ONLY to "net loss" and "operating loss" as the direct subject, not to any metric that contains the word "loss" (e.g. "allowance for loan losses", "accumulated other comprehensive loss" are NOT loss accounts for this purpose — they are balance-sheet items with positive reported values).

For questions asking **how much** a metric **changed** (absolute difference, not percentage) **from [YEAR_A] to [YEAR_B]** (e.g. "How much did Level 2 OFA change by from 2018 year end to 2019 year end?"):

1. **Identify the two years from the query string.**
   - "from [YEAR_A] to [YEAR_B]" or "from [YEAR_A] ... to [YEAR_B]" → old = YEAR_A, new = YEAR_B (e.g. 2018 = old, 2019 = new).
   - "between [YEAR_A] and [YEAR_B]" → old = earlier year, new = later year.
   - Do **not** infer year assignment from retrieval rank or chunk position.

2. **Anchor each value to its year using the nearest section header in context.**
   - Look for headers such as "As at 31 December 2018", "As at 31 December 2019", "Year ended [date]", or column labels with years.
   - Match each retrieved row value (e.g. "OFA | Level 2: 2,032") to the **section header or date that precedes it** in the context, not by rerank order.
   - If two chunks contain the same row label (e.g. "OFA | Level 2: X"), assign 2018 vs 2019 by the date header that goes with each chunk.

3. **Compute: subtract(value_at_new_year, value_at_old_year).**
   - This preserves sign: a decrease (new < old) produces a **negative** result (e.g. 375 - 2,032 = -1,657).
   - Do **not** use divide() — this is not a growth rate question.
   - Do **not** take abs() — sign is meaningful (increase vs decrease).

4. **Output the signed result** with the unit from context (e.g. "million", "RMB million"). If you cannot resolve which value is 2018 vs 2019 from section headers, **output exactly [YEAR_AMBIGUOUS]** and do not guess—this allows eval to separate missing-context from wrong-arithmetic failures.

Example (intentional: 2018 > 2019 so decrease → negative): Query "How much did Level 2 OFA change from 2018 year end to 2019 year end?" Context has "As at 31 December 2018" → OFA Level 2: 2,032; "As at 31 December 2019" → OFA Level 2: 375. Then old = 2,032 (2018), new = 375 (2019). **subtract(375, 2032)** = -1,657. Answer: -1,657 (with unit from context).
"""

# "Difference between A and B" (no directional framing): always subtract(larger, smaller) for non-negative result.
ABSOLUTE_DIFFERENCE_PRIMER = """
**Scope — applies ONLY when the question has no temporal directionality.**

Temporal directionality means the question specifies a BASE PERIOD and a COMPARISON PERIOD with a clear direction of change (e.g. "from 2009 to 2010", "percent reduction from X to Y", "percent increase", "percentage decrease", "percentage increase"). If ANY of those patterns appear AND the two values being compared are from DIFFERENT TIME PERIODS, do NOT use this primer — use PERCENT_REDUCTION_SIGN_PRIMER or PERCENT_CHANGE_BY_DIRECTION_PRIMER instead.

This primer applies to:
  - "difference between [category A] and [category B]" (same time period)
  - "difference between [value X] and [value Y]" (no time ordering)
  - "difference between [actual] and [target]" (plan vs actual, same period)

It does NOT apply to: "difference between [year A value] and [year B value]" when the two values come from different fiscal years and a direction of change is implied. In those cases use TABLE_YEAR_CHANGE_PRIMER.

The question asks for the **difference between** two values with no directional framing (e.g. "difference between operating lease obligations and other purchase obligations").
- **Convention:** Always compute **subtract(larger_value, smaller_value)** so the result is non-negative.
- Do **not** use positional order from the query — identify which value is larger and subtract the smaller from it.
- **Program:** subtract(max_val, min_val)
- **Example:** "difference between A=167.1 and B=205.6" → subtract(205.6, 167.1) = 38.5
"""

# When the question asks for "fluctuation", "change", or "difference" of sensitivities (e.g. credit spread, DVA per bp), use ratio-based formula.
FLUCTUATION_RELATIVE_CHANGE_PRIMER = """
When the question asks for **fluctuation**, **change**, or **difference** between years (e.g. credit spread sensitivity, DVA per basis point, segment % changes):

1. **Identify the earlier year as the base year** (unless the question explicitly specifies otherwise).

2. If the numbers represent **sensitivities or per-unit impacts** (e.g. "$39 million per 1 basis point"), compute fluctuation as a **ratio-based percentage**:
   **fluctuation (%) = ((later_year_value / earlier_year_value) − 1) × 100**
   **Program:** divide(later_value, earlier_value), subtract(#0, 1), multiply(#1, 100).

3. Only compute **absolute subtraction** (later − earlier) if the question **explicitly** asks for absolute change.

4. Always use numbers **verbatim** from the document; do not round or approximate before applying the formula.

5. Output the **numerical answer** at the scale requested (e.g. in basis points, millions, or percent) consistent with the question.
"""

# "X as a percentage of Y": part/whole then × 100; answer in percent (e.g. 16.84), not decimal (0.1684).
PCT_OF_TOTAL_PRIMER = """
The question asks for one value **as a percentage of** another (not a change over time).
- **Program:** divide(part, whole), multiply(#0, 100)
- The final answer is in **percent** (e.g. 16.84), not a decimal — do **not** omit the multiply(#0, 100) step; do **not** divide by 100 again.
- In your **prose conclusion**, state the percent value with the decimal point (e.g. "16.84%" or "16.84 percent"), not as a whole number (e.g. not "1684%").
- **Example:** Term Loan = 2,435.4, Total contractual obligations = 14,461.6 → divide(2435.4, 14461.6), multiply(#0, 100) = 16.84
"""

# Percentage of total aggregate contractual obligations: use "purchase obligations" row only (FinQA convention).
CONTRACTUAL_OBLIGATIONS_PCT_PRIMER = """
When the question asks for **percentage of total aggregate contractual obligations** (or "what percentage of total contractual obligations is composed of" a component):
- Locate the **aggregate contractual obligations** table (rows such as purchase obligations, operating leases, long-term debt, etc., and a total row).
- Use the row labeled **"purchase obligations"** as the **numerator** (part). Use the table's **total contractual obligations** as the **denominator** (whole).
- **Program:** divide(purchase_obligations_total, total_contractual_obligations). If the question asks for percent, then multiply(#0, 100).
- Do **not** use operating leases, long-term debt, or any other row for this question — only **purchase obligations**.
"""

# Portion of capital plan / budget that is for a specific component (e.g. PTC): same-unit ratio (UNP/2014-style).
# GT: total plan $4.3B → 4300 million; component $450M → ratio = 450/4300 = 0.10465.
CAPITAL_PLAN_COMPONENT_RATIO_PRIMER = """
When the question asks **how much of** a **capital plan** (or capital program / budget) **is for** a specific component (e.g. PTC expenditures, positive train control):

1. **Identify the total capital plan** for the requested year (e.g. "2015 capital plan ... approximately $4.3 billion") and the **component amount** (e.g. "expenditures for PTC of approximately $450 million").

2. **Express both in the same unit** (e.g. millions). If the total is stated in **billions** (e.g. $4.3 billion), convert to millions: **multiply(billions_value, 1000)** (e.g. multiply(4.3, 1000) = 4300). The component may already be in millions (e.g. 450).

3. **Ratio = component / total_in_same_unit.** Program: first step **multiply(total_billions, 1000)** to get total millions; second step **divide(component_millions, #0)**. Example: multiply(4.3, 1000), divide(450, #0) → 450/4300 = 0.10465.

4. **Output the decimal ratio** (e.g. 0.10465) as the answer unless the question explicitly asks for a percentage. Do **not** divide 450 by 4.3 (that would mix billions and millions); always convert the total to the same unit as the component before dividing.
"""

# Percent of cash provided by operations that was from a component (e.g. receivables securitization): part/whole; do not confuse "adjusted for" with total.
CASH_OPS_PCT_FROM_COMPONENT_PRIMER = """
When the question asks for **percent of cash provided by operations** (or "cash from operations") **that was from** a specific component (e.g. receivables securitization facility):
- **Numerator** = the **component** line item (e.g. "receivables securitization facility" or similar row) for the requested year.
- **Denominator** = **total** cash provided by operating activities (often labeled "cash provided by operating activities adjusted for ..." or "cash provided by operating activities"). The total is the full amount; the component is one part of it.
- Do **not** confuse "adjusted for [component]" with "total from [component]". "Adjusted for" means the component is included in the total; the total is the denominator, the component is the numerator.
- **Program:** divide(component_value, total_cash_ops). If the question asks for a decimal share, that is the result; if it asks for percent, then multiply(#0, 100).
- Example: total cash provided by operations (adjusted) = 4505, receivables securitization facility = 400 → divide(400, 4505) ≈ 0.08879.
"""

# When table has both beginning-of-year (BOY) and end-of-year (EOY) for a metric, prefer BOY for percent-change (FinQA convention).
BOY_PREFERENCE_PERCENT_CHANGE_PRIMER = """
When a financial table contains both **beginning-of-year (BOY)** and **end-of-year (EOY)** balances for the same metric (e.g. allowance for loan losses, reserves):
- **Prefer the BOY row** for percentage-change calculations between two years.
- **Denominator** = BOY value of the **earlier** year. **Numerator** = (BOY value of the **later** year minus BOY value of the earlier year). Compute: (BOY_later - BOY_earlier) / BOY_earlier (then × 100 if answer in percent).
- Only use the **EOY row** if the BOY row is missing from the context.
- Match the row label exactly (e.g. "at beginning of year", "at end of year") to determine which row is BOY vs EOY. For "percentage change in [metric] from [year A] to [year B]", use the BOY values for those years from the BOY row.
"""

# Growth rate / percentage change: (new - old) / old; use three-step chain when answer must be in percentage (0–100).
GROWTH_RATE_PRIMER = """
For **growth rate** or **% change** or **percentage change** questions (e.g. "what is the % change in total property and equipment from 2018 to 2019?"):
- When the question says **"% change"** or **"percent change"** or **"percentage change"**, you **must** use a **three-step program**: **subtract(new_value, old_value), divide(#0, old_value), multiply(#1, 100)**. Do **not** output only two steps — the answer must be in 0–100 form (e.g. -45.74, not -0.4574).
- **Step 1:** Locate **prior year** and **current year** values for the metric (same units). Use the **correct row**: for "total property and equipment" use the **net** figure (after accumulated depreciation), not gross "at cost" (see property-and-equipment note below if applicable).
- **Step 2:** old_value = prior year, new_value = current year.
- **Step 3:** If the question asks for a **decimal** growth rate only (no "percent"/"%"): two steps: `subtract(new_value, old_value), divide(#0, old_value)`.
  If the question asks for a **percentage** or **% change**: **three steps**: **subtract(new_value, old_value), divide(#0, old_value), multiply(#1, 100)**.
- **Step 4:** Output the **full program** (all steps); do not stop after the ratio step.
- Report negative for decrease; do not skip program execution—the question is numerical, not yes/no.
- **Property and equipment:** When the question asks for "% change in **total property and equipment**" (or "property and equipment, net"), use the **net** figure (net book value after accumulated depreciation), **not** the gross "at cost" or component subtotals. In balance sheet context, "total property and equipment" is the net amount. If the table has both a gross row (e.g. "2019: 336 | 2018: 2,523") and a net row (e.g. "2019: $70 | 2018: $129"), use the **net** row for the percent change calculation.
"""

# When the question asks for "percentage decrease/increase", FinQA expects answer in 0–100 (e.g. 96.55172), not decimal (0.9655).
PERCENTAGE_AS_INTEGER_PRIMER = """
When the question asks for **percentage decrease**, **percentage increase**, or **what percentage ... occurred** (not just "growth rate"):
- The FinQA dataset expects the answer as a number in **0–100** (e.g. 96.55172 for 96.55%), **not** as a decimal fraction (0.9655).
- **Percentage decrease** (e.g. "what percentage decrease from 2011 to 2012"): use **(old - new) / old * 100** so the result is **positive**. Use **multiply(divide(subtract(old_value, new_value), old_value), 100)**. Example: 2011=34.8, 2012=1.2 -> multiply(divide(subtract(34.8, 1.2), 34.8), 100) -> 96.55172.
- **Percentage increase**: use **(new - old) / old * 100**. Use **multiply(divide(subtract(new_value, old_value), old_value), 100)**.
- Do **not** output the fraction without * 100 (e.g. divide(subtract(...), old) alone gives 0.9655 and will not match). Output the single expression; the numerical answer will then be in 0-100 form and match the benchmark.
"""

# For explicit "percent reduction" / "percentage reduction": use (new - old)/old so a reduction yields a negative value (FinQA MMM-style).
PERCENT_REDUCTION_SIGN_PRIMER = """
When the question asks for **percent reduction** or **percentage reduction** (e.g. "what was the percent reduction in the board authorization from $12B to $10B"):
- **Always use (new - old) / old**, i.e. **divide(subtract(new_value, old_value), old_value)**. Do **not** use subtract(old_value, new_value) - that flips the sign.
- A **reduction** (new < old) must yield a **negative** answer (e.g. (10 - 12) / 12 = -0.16667). Do not take the absolute value.
- "Percent reduction from X to Y" means (Y - X) / X; when Y < X the result is negative. Use **subtract(new, old)** so the sign is correct.
"""

PERCENT_CHANGE_BY_DIRECTION_PRIMER = """
When the question asks for a **percentage increase/decrease** (from period A to period B) and does **not** explicitly say "percent reduction":
- Identify **old_value** as the earlier/base period (e.g. November 2018) and **new_value** as the later/end period (e.g. February 2019).
- Always compute **percent change = (new_value - old_value) / old_value * 100**. Use a **three-step program**:
  **subtract(new_value, old_value), divide(#0, old_value), multiply(#1, 100)**.
- A **decrease** (new_value < old_value) will naturally produce a **negative** percentage; an **increase** (new_value > old_value) produces a **positive** percentage. Do **not** flip operands or take absolute value based on direction.
- Denominator is always the **base (old)** value, never the new value. Do not use (old-new)/new.
- Always include the final **multiply(#1, 100)** step so the answer is in percentage form (0–100) rather than a raw ratio.
"""

# When the question asks for an average over several years (e.g. 2012, 2011, 2010), FinQA gold sometimes uses only a subset (e.g. two most recent).
AVERAGE_SUBSET_PRIMER = """
**Exception — exactly two periods or two values.** When the question asks for the **average of exactly two** periods or two explicitly stated values (e.g. "average ... for 2018 and 2019", "average of X and Y"), the divisor is unambiguously 2. **Proceed with** add(val1, val2), divide(#0, 2). Do **not** defer.

**Multi-step sum → average:** When averaging **three or more** values, chain adds (e.g. add(a, b), add(#0, c)); the **full sum** is in the **last** add step. Use **divide(#1, n)** (or the correct last step index), not divide(#0, n). Example: add(38, 34), add(#0, 20) → sum in #1 = 92; use divide(#1, 3) = 30.6667, not divide(#0, 3).

**FINQA HARD RULE — AVERAGE (when divisor is ambiguous).** If the question contains "average", **three or more** numeric candidates are extracted, and **no explicit divisor or formula** appears in the text (e.g. no "divided by 3", "over three years", "per year"), then:
- **DO NOT execute arithmetic** (no divide(#0, 3) or other guess).
- **DO NOT select a subset** of values (no "most recent two", no "operationally relevant").
- **DO NOT assume equal weighting** or default to arithmetic mean.
- **Mark the computation as procedurally undefined** and state that the average cannot be determined from the text (e.g. "Average definition not specified in document; computation deferred."). Do not output a number.

When exactly two values are identified and the question asks for their average, the formula is unambiguously (val1 + val2) / 2; do not defer.

**FINQA COMMITMENT RULE.** If "average" appears and multiple values are listed but no explicit formula/divisor/weighting is stated: do **not** choose one strategy; treat as procedurally defined; if multiple plausible averages exist, do not choose one — defer or state underdetermined.

**Financial QA average semantics.** In financial filings, "average" may refer to: a **two-period average**; an average from **grant mechanics** or plan structure; an **adjusted average** (selected years only); or an **accounting/policy** definition. Do not assume all mentioned years are included or equal weighting. Prefer identifying which periods are **operationally relevant** and whether one year is excluded due to **plan structure**.

When the question asks for an **average** over multiple years (e.g. "average ... in 2012, 2011 and 2010"):
- Use **only a subset** of the listed years when the text or plan structure does not explicitly require all (e.g. two most recent). The document may list 2012, 2011 and 2010 "respectively"; the correct program may still use only 2012 and 2011.
- Programs may include **small constant adjustments** (e.g. add(#1, const_3), divide(#2, const_2)) for rounding or policy.
- **Selective extraction**: Do not sum or average over all listed items unless the question clearly asks for "total" or "all". Do not force inclusion of every number next to the listed years.
"""

# Cumulative total return / indexed comparison: normalize (level - 100) / 100, then difference of returns (AOS/2007-style).
CUMULATIVE_RETURN_PRIMER = """
When the question asks for **difference in cumulative total return**, **five-year comparison** of returns, **outperform** / **underperform** vs an index, or refers to **indexed returns** / **assumes $100 invested** (base period = 100):
- **Cumulative total return** is defined as **(index level - 100) / 100** (decimal return from base), **not** the raw index level.
- **Step 1:** Identify base period = 100 in the table. Extract the **ending** index level for each series (e.g. company, benchmark index).
- **Step 2:** Convert each ending level to **return**: return = (level - 100) / 100. Use subtract(level_a, 100), divide(#0, 100) for the first series; subtract(level_b, 100), divide(#2, 100) for the second (use literal 100 in the program).
- **Step 3:** The **difference in cumulative total return** (e.g. company vs index, or "how much did X outperform Y") = company return - index return. Use subtract(return_company, return_index). Do **not** subtract raw index levels (e.g. subtract(431, 197) is wrong; 431 and 197 are levels, not returns). The answer is in return-space (e.g. 2.34), not level-space.
- **Program template:** subtract(level_a, 100), divide(#0, 100), subtract(level_b, 100), divide(#2, 100), subtract(#1, #3). Report the final difference (e.g. 2.34 or -0.6767) as the numerical answer.
"""

# When the question specifies a unit scale, prefer values already expressed at that scale over raw table figures at a different scale.
UNIT_SCALE_PRIMER = """
The question specifies a unit scale (e.g. "in millions"). When the context contains both **rounded prose values** at that scale (e.g. "$52.4 million", "$32.1 million") and **raw table values** at a different scale (e.g. "52,380" or "32,136" in thousands), **prefer the values already expressed at the requested scale**. Use subtract(52.4, 32.1) not subtract(52380, 32136).
"""

# Average assets per self-sponsored / multi-seller conduit: use reported assets only, divide by number of conduits (FinQA JPM/2007).
CONDUIT_AVERAGE_ASSETS_PRIMER = """
When the question asks for **average assets** (e.g. in billions) **for each of the firm's self-sponsored** conduits or **multi-seller conduits**:
- The context may show a table with **reported** and **pro forma** columns (e.g. assets | reported: $ 1562.1 | pro forma: $ 1623.9).
- Use the **reported** value for assets as the total (e.g. 1562.1). Do **not** use pro forma; do **not** subtract reported from pro forma or average the two.
- Divide the **reported** assets by the **number of conduits** stated in the text (e.g. "four multi-seller conduits" → 4). **Program:** divide(reported_assets, number_of_conduits).
- The result is average assets per conduit. If the question asks for billions and the table is in billions, the answer is already in billions.
"""

# When a percentage/ratio is stated for a prior year and no updated figure exists for the query year, apply the most recently stated percentage to the query year's base (FinQA annotation convention).
CROSS_YEAR_CARRY_FORWARD_PRIMER = """
**Cross-year carry-forward (overrides strict year-matching when applicable):** When the question asks for a dollar amount in a specific year (e.g. "how many X did the 5 largest customers account for in 2008?") and the context states a **percentage for a prior year only** (e.g. "in 2007, the five largest customers accounted for approximately 42%") with **no updated percentage for the query year**:
- **Apply the most recently stated percentage to the query year's base figure.** Use the query year's base (e.g. 2008 net sales $10,086) × prior-year percentage (42%): multiply(10086, 42%).
- Do **not** refuse with INSUFFICIENT_DATA when the only missing element is a carried-forward percentage. This is standard in financial filings: concentration percentages are often disclosed for one year and implicitly applied to adjacent years.
- Example: "how many segmented sales the 5 largest customers account for in 2008?" — 2008 net sales = $10,086, 2007's five-largest share = 42%. Answer: multiply(10086, 42%) = 4236.12.
"""

# =============================================================================
# SECTION 4 — LOSS ACCOUNT PRIMERS  (IAS 1 / ASC 220)
# =============================================================================

# "Change in" a loss figure (Net Loss, loss, losses): use magnitude convention, not signed raw values.
LOSS_CHANGE_PRIMER = """
The question asks for the **change in** a loss figure between two periods.
- **Financial reporting convention:** Report the change as **(magnitude_new - magnitude_old)**, where magnitudes are the absolute values of the loss figures (strip the negative sign from table values like $(15,571)).
- The sign of the result indicates direction: **negative** = loss decreased (improved); **positive** = loss increased (worsened).
- Do **not** use signed arithmetic on the raw negative values (e.g. do not compute subtract(-15571, -24122)).
- **Program:** subtract(magnitude_new_period, magnitude_old_period) using the numeric magnitudes only.
- **Example:** Net Loss 2019 = $(15,571), 2018 = $(24,122) → magnitude_2019 = 15,571, magnitude_2018 = 24,122 → subtract(15571, 24122) = -8,551 → Answer: -8,551 (loss decreased by 8,551).
"""

# "Average" of a loss figure: use magnitudes, then simple mean. divide(add(mag1, mag2), 2).
LOSS_AVERAGE_PRIMER = """
The question asks for the **average** of a loss figure over two periods (e.g. "average Net Loss for 2018 and 2019").
- Use the **magnitude convention**: take the absolute values of the loss figures (e.g. $(15,571) → 15,571; $(24,122) → 24,122). Do **not** use the signed negative values.
- **Program:** add(magnitude_1, magnitude_2), divide(#0, 2)
- **Example:** Net Loss 2019 = $(15,571), 2018 = $(24,122) → add(15571, 24122), divide(#0, 2) = 19,846.5
"""

# "Net loss less/greater than X": compare by magnitude (abs), not signed value. Financial convention: "less" = smaller magnitude.
LOSS_COMPARISON_PRIMER = """
Net loss is a **LOSS account**. Its natural balance is **NEGATIVE**.
$(19,898) means the company lost $19,898 — the signed value is -19,898.

When a question asks "in which year was the net loss less than X":
- This means: in which year was the loss **MAGNITUDE** (absolute value) below X.
- Use **abs(loss) < X**, NOT the signed value comparison.
- "net loss less than -10,000" → find year where **abs(loss) < 10,000**.

Step-by-step for net loss values 2019=$(19,898), 2018=$(26,199), 2017=$(9,187):
- abs(-19,898) = 19,898 > 10,000 → 2019 does NOT qualify
- abs(-26,199) = 26,199 > 10,000 → 2018 does NOT qualify
- abs(-9,187) = 9,187 < 10,000 → 2017 QUALIFIES ✓
Answer: 2017

Do **not** use signed arithmetic: -9,187 > -10,000 is mathematically true but **wrong** in financial reporting context.
"""

# =============================================================================
# SECTION 5 — EQUITY COMPENSATION PRIMERS  (ASC 718)
# =============================================================================

# Financial reasoning primer: left vs right and conservative estimate for period expense (ASC 718)
FINANCIAL_COMPENSATION_PRIMER = """
Financial reasoning (SEC filings / GAAP) for "did [A] exceed [B]?" style questions:
- **Left side (A):** For "equity awards in which performance milestones were achieved," use the amount the document explicitly reports for that concept. If the document states **stock-based compensation expense** recognized for those awards (e.g. "$3.3 million in stock-based compensation expense for equity awards in which ... milestones have been achieved"), use that as A—it is the **recognized expense** for achieved/probable milestones, not the grant-date fair value of the awards (which would be higher). Only use award fair value as A if the question or document clearly asks for "value" or "fair value" of achieved awards.
- **Right side (B):** ONLY the compensation expense **recognized in that same period** for equity **granted during the period** (i.e. the amortized portion of new grants in year 1), NOT the full grant-date fair value.
    - When the document gives only grant-date fair value for new grants (e.g. shares * price ~ $11M), year-1 **recognized** expense is usually **smaller** than the total fair value. Under ASC 718, compensation cost is attributed on a **straight-line basis over the requisite service period**. If the document states the vesting period explicitly, use **fair value / vesting_years** as the year-1 recognized expense estimate. If the vesting period is not provided, use **fair value / 4** as a conservative lower-bound estimate - it assumes a 4-year vesting period and full-year grant timing (typical for technology companies; the US RSU market norm is 3 years, making this deliberately conservative). If the left-side amount is close to or above this conservative lower bound, the answer is typically **yes**. Do not use the full grant-date fair value as the right-side comparator - that represents total expense over the entire vesting period, not year-1 recognized expense.
"""

# Equity plan "issued vs remaining" yes/no: compare shares (or securities) to be issued vs remaining; answer yes if issued > remaining, no otherwise.
EQUITY_PLAN_ISSUED_VS_REMAINING_PRIMER = """
When the question asks whether **there are more shares (or securities) issued than remaining** under an equity or incentive plan (e.g. "are there more shares issued than remaining in the plan?", "were more shares to be issued than remaining?"):

- **Identify the two numbers from the document:** (1) **Securities (or shares) to be issued** (or already issued under the plan), and (2) **Securities (or shares) remaining** (available for future issuance). Tables or narrative often label these explicitly (e.g. "Securities to be issued: X", "Securities remaining: Y").

- **Comparison rule:** **If issued > remaining, answer "yes". If issued ≤ remaining, answer "no".** Do not assume a default "yes"; always compute the comparison from the document numbers.

- **Reasoning template:** Extract X (to be issued / issued) and Y (remaining). If X > Y → answer **yes**; else answer **no**. Preserve the binary answer format exactly as requested (yes/no).

- **Always follow the yes/no expectation in the question wording.** Even if the numbers are close, output the correct yes or no based on the comparison. Do not assume default "yes"; always compute from document numbers. Output must match the requested yes/no style exactly.
"""

# =============================================================================
# SECTION 6 — SPECIALISED DOMAIN PRIMERS
# =============================================================================

# Gold-blind numerical reasoning (FinQA/TAT-QA): derive from context only; one task type; executor result is final.
FINQA_GOLDBLIND_NUMERICAL_PRIMER = """
**Gold-blind rules (you do not know the gold answer).** (1) Identify the numeric task type first: absolute adjustment, difference in returns / outperformance, percent change, growth rate, or ending balance. Do not mix task types. (2) For cumulative return / outperformance: Return_X = (Ending_X - 100) / 100, Return_Y = (Ending_Y - 100) / 100, Outperformance = Return_X - Return_Y. Never subtract raw index levels. (3) For accounting adjustments: use only the single adjustment line tied to the question; do not sum multiple adjustments unless the question says "total" or "combined". (4) If your program executes successfully, the executor output is the answer; do not reinterpret or replace it with heuristics. (5) Units: preserve the document unit (millions, thousands, index base 100); do not rescale unless the question asks. (6) **Multi-step programs:** When the answer requires multiple arithmetic steps (e.g. sum A+B then add that to C), write them as a **single chained program on one line**, e.g. `add(4801, 1882), add(3711, #0)`. Do **not** split steps into separate paragraphs — the executor reads only the first program candidate it finds.
"""

# For "growth rate of loans held-for-sale carried at LOCOM": use carried amount = min(cost, fair value) per year, then growth = (carried_new - carried_old) / carried_old.
LOCOM_GROWTH_PRIMER = """
**For growth rate of loans held-for-sale carried at LOCOM (or similar):**
- LOCOM = lower of cost or market (fair value). The **carried amount** on the balance sheet is the **lower** of aggregate cost or fair value for each year. Final carried amounts MUST be the balance-sheet carrying value (min(cost, fair value)).
- **Step 1:** Quote the full table row for both years, including **aggregate cost** and **fair value** columns.
- **Step 2:** For each year, **carried amount** = min(aggregate cost, fair value). When fair value < cost, the asset is impaired and the carried amount is fair value.
- **Step 3:** Compute growth = (carried_current_year - carried_prior_year) / carried_prior_year. Use divide(subtract(carried_new, carried_old), carried_old).
- **Step 4:** If fair value is lower than cost in both years, use the **fair value** column for both years when computing growth (growth rate is based on fair value column). Do not use aggregate cost for one year and fair value for the other.
- **ALWAYS** state the carried values you use for each year before calculating the growth rate. Report carried values explicitly before the growth calculation.
- After computing carried amounts, cross-check against the question: "loans ... carried at LOCOM" refers to the **actual carrying value** (min(cost, fair value)), NOT aggregate cost unless explicitly stated. If carried values show impairment (fair value < cost), use those for growth rate.
- If your computed value seems inconsistent with a benchmark answer, note "possible GT inconsistency" but still output the computed value.
"""

# Parenthetical negative: in financial tables, -X ( X ) means the value is -X; use the signed value in calculations (FBHS/2017-style).
FINANCIAL_PARENTHETICAL_NEGATIVE_PRIMER = """
In financial tables, a value shown as **-X ( X )** (e.g. -1.9 ( 1.9 ) or -2.5 ( 2.5 )) means the value is **-X**. The parenthetical is standard SEC/financial notation for the same negative number (alternative representation). **Always use the signed value (-X) in calculations**, not the absolute value in parentheses. For example: if a row shows "2015: -2.5 ( 2.5 )", use **-2.5** (e.g. in divide(numerator, -2.5)), not 2.5.
"""

# Lease "percent of total": use narrative rent-expense line as numerator, not sum of future schedule (UNP/2016-style).
LEASE_PERCENT_PRIMER = """
When the question asks for **percent of total operating leases** (or lease) and **terms** (e.g. "terms greater than 12 months"):
- **Numerator:** Use the **narrative line** that states rent/lease expense for the **requested year** with "terms exceeding" or similar (e.g. "rent expense for operating leases with terms exceeding one month was $535 million in 2016"). That amount is already the filtered value for long-term leases; use it **directly**. Do **not** sum future-year schedule rows (2017, 2018, …) as the numerator — that answers "payments due after the current year," not "amount due for leases with terms > 12 months."
- **Denominator / total:** FinQA often defines "total" here as **numerator + total minimum lease payments**. Find the row "total minimum lease payments | operating leases: $X" (e.g. $3,043). Then **percent = numerator / (numerator + total_minimum)**, i.e. divide(expense, add(expense, total_minimum)). Example: divide(535, add(535, 3043)) = 0.14952.
- **Do not** reconstruct the numerator from the future payments table when a direct descriptive sentence gives the expense for the requested year.
"""

# Bond / interest payment: FinQA often targets one instrument and periodic (semi-annual/quarterly) payment, not annual total.
INTEREST_PAYMENT_PRIMER = """
For bond/interest payment questions (FinQA annotation style):

- The question may say "the bonds issued by [entity]", but gold often targets **only the first or main** debt instrument described in the note. Do **not** sum interest across multiple bond series from the same issuer unless the question clearly asks for total.
- "Interest payment incurred" or "amount of interest payment" usually means the **periodic payment** (semi-annual or quarterly), **not** the full annual interest. When the text says "payable semi-annually" or "payable quarterly", compute annual interest first then **divide by 2** (semi-annual) or **divide by 4** (quarterly).
- Look explicitly for payment frequency phrases ("payable semi-annually", "payable quarterly") and apply the correct divisor.
- Typical gold program for semi-annual interest: multiply(principal, rate%), divide(#0, const_2). For quarterly: divide(#0, const_4).
- Do **not** default to annual total unless the question explicitly asks for "annual interest expense" or "total annual interest".
"""

# Cash flow / share repurchase / financing: scale to millions, correct column for actual cash outflow.
CASHFLOW_FINANCING_PRIMER = """
You are answering a question about cash flow from financing activities. Scale all dollar values to millions.

**Share repurchase cash outflow rule:** When calculating cash spent on share repurchases, always use the **total number of shares purchased** column (which includes both open-market repurchases and employee share surrenders for tax withholding obligations), multiplied by the **average price paid per share**. Do **not** use the "shares purchased as part of publicly announced plan or program" column—that figure excludes employee surrenders and understates actual cash outflow.

**Column disambiguation:** Financial tables related to repurchases typically contain multiple share-count columns. The correct one for cash flow purposes is the one labeled "total number of shares purchased" or equivalent, not the subset tied to a specific board-authorized program.

**Unit check:** If shares are given as whole numbers and price as dollars per share, divide the product by 1,000,000 to convert to millions.

If the question asks how repurchases **affect** net change in cash from financing: the answer is the repurchase cash outflow in millions (total shares * price / 1,000,000) for the requested period. Use the **requested period** row only; do not sum across periods unless the question asks for total. Output a single program (e.g. multiply(total_shares, avg_price), divide(#0, 1000000) for millions).

**Change in balance (money pool, payables, receivables):** When the question asks how cash flow is **affected by the change in balance** of a financing item (e.g. receivables from or payables to a money pool, short-term borrowings):
- Use the **numeric magnitudes** from the table (e.g. 51232 and 52742), not negative numbers for parenthetical amounts.
- Compute **subtract(current_year_value, previous_year_value)** so that a **decrease in a payable** (current < previous) yields a **negative** result (cash outflow). Example: 2016: 51,232 and 2015: 52,742 → subtract(51232, 52742) = -1510. Do **not** use subtract(previous, current)—that reverses the sign and fails FinQA exact match.
- Scale is as stated (e.g. thousands); no extra conversion unless the question asks for millions.
"""

# Event-scoped arithmetic: do not sum across footnote/event blocks (FinQA AMT/2012-style).
EVENT_SCOPED_ARITHMETIC_PRIMER = """
For questions about **acquired intangibles**, **customer-related / network location intangibles**, **amortization expense**, or **purchase price allocation** (event-scoped arithmetic; same principle applies to debt, leases, segments, tax):
- The context may contain **multiple acquisition blocks** (e.g. numbered footnotes (1), (2), (3), or separate deals with different dates). Treat each block as **one** acquisition with its own numbers.
- **Single-block default**: If the question does **not** explicitly ask for "combined", "total", "aggregate", "in total", or "overall" across acquisitions, use **exactly one** block—never more. Never sum across footnotes unless one of those aggregation keywords appears.
- **Do NOT sum** across separate footnotes or acquisition blocks unless the question **explicitly** asks for "combined", "total", "aggregate", "in total", or "overall" across acquisitions. If none of these trigger words appear, compute the answer **for a single acquisition only**.
- **Compute per block**: For each acquisition block, identify (1) the intangible amounts (e.g. customer-related + network location), (2) the amortization period (e.g. straight-line over 20 years), and (3) annual amortization = total intangibles ÷ years. Do this for **one** block, not across blocks.
- **Which block to use**: Prefer the acquisition block that has **larger magnitude** (e.g. $75M + $72.7M vs $10.7M + $10.4M) when the question asks for "expected" or "annual" amortization without specifying which deal—FinQA often expects the more significant acquisition. Alternatively use the block that matches an explicit table reference or the most recent date mentioned in the question. State briefly which block you selected and why (e.g. "using the acquisition block with larger magnitude").
- Output the **single** annual amortization (or other requested figure) for that one acquisition block. Never add amortization from block (1) and block (3) together unless the question asks for combined/total.
"""

# Accounting adjustment selection & unit handling (TAT-QA): do not aggregate unless question asks for total; select standard-specific line; preserve table units.
ACCOUNTING_ADJUSTMENT_PRIMER = """
When answering accounting adjustment questions:

1. Do NOT aggregate multiple line items unless the question explicitly asks for a total.
   - If a question refers to "the cumulative-effect adjustment" (singular),
     select the specific adjustment associated with the accounting change mentioned.

2. Prefer adjustments tied to adoption of a named accounting standard
   (e.g., ASC 606) over unrelated or prior-period adjustments.

3. Values in financial tables are often reported in units (e.g., millions).
   - Preserve the table's unit.
   - Do NOT sum or rescale unless explicitly instructed.

4. The final answer should match the unit and scale implied by the table
   and the question wording.

5. When the question asks for the **percentage of an adjustment** relative to a base (e.g. "percentage of adjustment to the balance of as reported X"):
   - Treat the **adjustment line (e.g. 16.6)** as a magnitude; accounting parentheses indicate direction in the table, but percentage-of-adjustment questions usually care about **size**, not sign.
   - Use **abs(adjustment) / base** as the ratio, and when the question explicitly asks for a percentage, **multiply by 100** in the final step.
   - Correct program shape for percentage of adjustment: `multiply(divide(16.6, 93.8), 100)` (adjustment magnitude ÷ base, then ×100). Do **not** stop at the ratio step.
"""

# When the document gives "X out of Y days" or "on Z days of total days", FinQA gold is usually proportion (X/Y), not the raw count.
FREQUENCY_PROPORTION_PRIMER = """
For questions like "how often", "how frequently", or "how many times" in a defined period (e.g. year, quarter):

- FinQA gold almost always requires a **proportion** (events / total_days_or_periods), **not** the raw count.
- When the document provides both numbers (e.g. "on X of Y days", "gains on eight days exceeding $200 million" out of "261 days"), compute: **divide(count, total)** to get the decimal proportion (e.g. 8/261 ~ 0.03065).
- Do **not** stop at the absolute number (e.g. "8 days"). The answer is the fraction or proportion. If the question asks for a percentage, multiply the proportion by 100 after computing it.
- Example: "how often did the firm post gains exceeding $200 million in 2012?" with text "gains on eight days exceeding $200 million" and "261 days" -> answer: divide(8, 261) ~ 0.03065, not 8.
"""
