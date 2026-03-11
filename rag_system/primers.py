"""
RAG primers for FinQA and TATQA.

Phase 2: SHARED_NUMERICAL_PRIMER is a base layer; query-intent primers (TABLE_YEAR,
GROWTH_RATE, etc.) extend it. Assembly: prompt = SHARED_NUMERICAL_PRIMER + "\n\n" + intent_specific_primer.

Do NOT modify FINANCEBENCH_PRIMER here — it lives in eval_runner.py and is production-stable at 94%.
Last updated: 2026-03-06
"""

# =============================================================================
# SHARED_NUMERICAL_PRIMER — base layer for FinQA and TATQA
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
"""

# =============================================================================
# WHAT_TABLE_SHOWS_PRIMER — disambiguate table when query is ambiguous
# =============================================================================
# For "What does the table show?" style queries: force explicit table identification
# to avoid answering about the wrong table when retrieval surfaces multiple tables.
# Gold-blinded: no reference to specific table content or gold answer.
# =============================================================================
WHAT_TABLE_SHOWS_PRIMER = """
Before answering any "What does the table show?" (or similar) query:
1. **Explicitly identify the table** you are referencing (by its header, surrounding section, or subject matter).
2. If **multiple tables** are present in the retrieved context, state **which one** you are describing and **why** you selected it.
3. **Do not assume** there is only one table in the document. If the context contains several distinct tables and the query does not disambiguate, either name the table you are using or note that the answer refers to a specific table (identify it) rather than the whole document.
"""

# =============================================================================
# ARITHMETIC_FROM_COMPONENTS_PRIMER — compute ratio/total from components when not stated
# =============================================================================
# For ratio/total questions (e.g. "ratio of total X to total Y"): if the required total
# is not explicitly stated but all component values are present, compute from components.
# =============================================================================
ARITHMETIC_FROM_COMPONENTS_PRIMER = """
When the question asks for a **ratio** (e.g. ratio of total X to total Y) or a **total** that can be derived from line items:
- If the **exact total** is not explicitly stated in the context but **all component values** are present (e.g. individual current asset and liability lines), **compute the total from the components** (add or combine as appropriate), then compute the ratio or answer. Do **not** return INSUFFICIENT_DATA when the components are all present and the formula is clear.
- State the components you used and the computation (e.g. total = A + B + C, ratio = total_X / total_Y) so the answer is auditable.
"""
