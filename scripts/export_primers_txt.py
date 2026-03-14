#!/usr/bin/env python3
"""
Export all RAG primers to a human-readable text file for documentation and
job application attachment.

Usage:  python scripts/export_primers_txt.py
Output: docs/primers_export.txt
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from rag_system.primers import (
        SHARED_NUMERICAL_PRIMER,
        WHAT_TABLE_SHOWS_PRIMER,
        ARITHMETIC_FROM_COMPONENTS_PRIMER,
        TABLE_YEAR_PRIMER,
        TABLE_YEAR_CHANGE_PRIMER,
        TABLE_DATE_COLUMN_PRIMER,
        TABLE_TOTAL_ACROSS_COLUMNS_PRIMER,
        TOTALS_PREFER_DIRECT_PRIMER,
        ABSOLUTE_CHANGE_PRIMER,
        ABSOLUTE_DIFFERENCE_PRIMER,
        PCT_OF_TOTAL_PRIMER,
        GROWTH_RATE_PRIMER,
        PERCENTAGE_AS_INTEGER_PRIMER,
        PERCENT_REDUCTION_SIGN_PRIMER,
        PERCENT_CHANGE_BY_DIRECTION_PRIMER,
        AVERAGE_SUBSET_PRIMER,
        CUMULATIVE_RETURN_PRIMER,
        UNIT_SCALE_PRIMER,
        CROSS_YEAR_CARRY_FORWARD_PRIMER,
        LOSS_CHANGE_PRIMER,
        LOSS_AVERAGE_PRIMER,
        LOSS_COMPARISON_PRIMER,
        FINANCIAL_COMPENSATION_PRIMER,
        FINQA_GOLDBLIND_NUMERICAL_PRIMER,
        FINANCIAL_PARENTHETICAL_NEGATIVE_PRIMER,
        LOCOM_GROWTH_PRIMER,
        INTEREST_PAYMENT_PRIMER,
        LEASE_PERCENT_PRIMER,
        CASHFLOW_FINANCING_PRIMER,
        EVENT_SCOPED_ARITHMETIC_PRIMER,
        ACCOUNTING_ADJUSTMENT_PRIMER,
        FREQUENCY_PROPORTION_PRIMER,
    )
except ImportError as e:
    print(f"[export_primers_txt] ERROR: Cannot import rag_system.primers — {e}")
    print("  Run this script from the repo root: python scripts/export_primers_txt.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Primer catalogue — each entry defines one primer's metadata and content.
# Order controls appearance in the output file.
# ---------------------------------------------------------------------------

SECTION_PREAMBLES: dict[int, str] = {
    4: """\
These three primers address a single systematic failure class: the base LLM does not
recognise that Net Loss carries a NEGATIVE natural balance (IAS 1 / ASC 220). This
causes three distinct failure modes on financial QA benchmarks.

INTERVIEW EXAMPLE A — Reasoning False Negative: ASC 718 vesting-period amortisation
  Failure mode: Model conflates grant-date fair value with the expense recognised in
  a given year under ASC 718. If RSUs worth $11M are granted with a 3-4 year vesting
  schedule, year-1 recognised expense is ~$2.5-3M (fair value / 4, conservative
  estimate). The model without a primer uses the full $11M as the right-side
  comparator in a "did A exceed B?" question, systematically producing false "no"
  answers. Primer fix: FINANCIAL_COMPENSATION_PRIMER (Section 5).

INTERVIEW EXAMPLE B — Reasoning False Negative: Anti-dilutive RSU exclusion from EPS
  Failure mode: When a company reports a net loss, adding RSU / option shares to the
  diluted EPS denominator makes the loss PER SHARE look smaller (less severe) — this
  is anti-dilutive. IAS 33 / ASC 260 explicitly prohibit including anti-dilutive
  shares in the diluted denominator. The related primer (LOSS_COMPARISON_PRIMER below)
  encodes the natural-balance reasoning that underlies the entire loss-account family
  of failures. The scorer also exposed a GT annotation gap: the model correctly
  retrieved BOTH years (383,000 and 750,000) while GT only captured the 2019 figure.

INTERVIEW EXAMPLE C — Reasoning False Positive: Net loss magnitude comparison
  Failure mode: Question: "In which year was the net loss less than -10,000?"
  Net loss values: 2019=$(19,898), 2018=$(26,199), 2017=$(9,187).
  Model applies signed arithmetic: -9,187 > -10,000 (true on number line) → wrongly
  excludes 2017. Correct reasoning: "net loss less than X" means magnitude (abs value)
  below X. abs(-9,187) = 9,187 < 10,000 → 2017 is the correct answer.
  Accounting standard: IAS 1 / ASC 220. Primer fix: LOSS_COMPARISON_PRIMER below.""",
}

PRIMER_CATALOGUE: list[dict] = [
    # ------------------------------------------------------------------
    # Section 1: Shared / Always-On
    # ------------------------------------------------------------------
    {
        "section": 1,
        "section_title": "SHARED / ALWAYS-ON PRIMERS",
        "name": "SHARED_NUMERICAL_PRIMER",
        "intent_label": "shared — injected on every call",
        "trigger": "Always injected as base layer; intent primers extend it",
        "standard": "General GAAP / IFRS numerical conventions",
        "rationale": (
            "Universal rules applied to all queries: output format (decimal vs percent), "
            "unit preservation, answer-first ordering, and change vs rate disambiguation. "
            "Reduces the most common low-level formatting errors without any query-specific logic."
        ),
        "content": SHARED_NUMERICAL_PRIMER,
    },
    # ------------------------------------------------------------------
    # Section 2: Table Extraction
    # ------------------------------------------------------------------
    {
        "section": 2,
        "section_title": "TABLE EXTRACTION PRIMERS",
        "name": "TABLE_YEAR_PRIMER",
        "intent_label": "table_year",
        "trigger": "Query mentions a 4-digit year AND numerical table keywords (total, revenue, expense, etc.)",
        "standard": "General multi-period financial table navigation",
        "rationale": (
            "Prevents year-column cross-contamination: model must confirm column order from "
            "a known prose anchor before extracting any value, and must not mix columns across rows."
        ),
        "content": TABLE_YEAR_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "TABLE_YEAR_CHANGE_PRIMER",
        "intent_label": "absolute_change (table variant)",
        "trigger": "Query asks for change from one year to another using table data",
        "standard": "General financial table arithmetic",
        "rationale": (
            "Gold-blinded year-over-year change: locate earlier and later year values, "
            "compute subtract(later, earlier), return signed result."
        ),
        "content": TABLE_YEAR_CHANGE_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "TABLE_DATE_COLUMN_PRIMER",
        "intent_label": "table_date_column",
        "trigger": "Query contains 'as of' or 'as at' plus a date pattern",
        "standard": "SEC filing column-oriented table extraction",
        "rationale": (
            "Financial tables often have two date columns; chunking may fragment them. "
            "This primer forces exhaustive context scan and rejects candidate values "
            "that belong to the wrong date column."
        ),
        "content": TABLE_DATE_COLUMN_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "TABLE_TOTAL_ACROSS_COLUMNS_PRIMER",
        "intent_label": "table_total_across_columns",
        "trigger": "Query asks for total of a line item when table may span multiple chunks",
        "standard": "Multi-column financial table reconstruction",
        "rationale": (
            "When chunking splits a multi-column table, the labelled 'total' is often only "
            "one column's subtotal. This primer instructs the model to sum across column "
            "subtotals when multiple plausible values exist."
        ),
        "content": TABLE_TOTAL_ACROSS_COLUMNS_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "TOTALS_PREFER_DIRECT_PRIMER",
        "intent_label": "totals_prefer_direct",
        "trigger": "Query asks for a named total (e.g. total operating expenses)",
        "standard": "Income statement / MD&A direct-line preference",
        "rationale": (
            "Model without this primer may back-calculate a total from a percentage when a "
            "direct total line is present, producing errors from rounding or percentage mismatch. "
            "Rule: always prefer the direct line."
        ),
        "content": TOTALS_PREFER_DIRECT_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "WHAT_TABLE_SHOWS_PRIMER",
        "intent_label": "what_table_shows",
        "trigger": "Query asks 'what does the table show?' or similar",
        "standard": "General financial document disambiguation",
        "rationale": (
            "When retrieval surfaces multiple tables, the model must explicitly identify "
            "which table it is describing before answering."
        ),
        "content": WHAT_TABLE_SHOWS_PRIMER,
    },
    {
        "section": 2,
        "section_title": None,
        "name": "ARITHMETIC_FROM_COMPONENTS_PRIMER",
        "intent_label": "arithmetic_from_components",
        "trigger": "Query asks for ratio or total when the explicit total is absent but components are present",
        "standard": "General financial arithmetic",
        "rationale": (
            "Prevents premature INSUFFICIENT_DATA when all component values are present "
            "and the formula is unambiguous. Forces the model to compute rather than defer."
        ),
        "content": ARITHMETIC_FROM_COMPONENTS_PRIMER,
    },
    # ------------------------------------------------------------------
    # Section 3: Arithmetic Operations
    # ------------------------------------------------------------------
    {
        "section": 3,
        "section_title": "ARITHMETIC OPERATION PRIMERS",
        "name": "ABSOLUTE_CHANGE_PRIMER",
        "intent_label": "absolute_change",
        "trigger": "Query contains 'change by', 'change from', 'changed by', 'changed from'",
        "standard": "General financial arithmetic — signed difference",
        "rationale": (
            "Anchors year assignment to section headers in context (not retrieval rank), "
            "computes subtract(new, old), preserves sign. Prevents growth-rate primer from "
            "firing on absolute-difference questions."
        ),
        "content": ABSOLUTE_CHANGE_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "ABSOLUTE_DIFFERENCE_PRIMER",
        "intent_label": "abs_difference",
        "trigger": "Query contains 'difference between' with no directional framing",
        "standard": "General financial arithmetic — non-negative difference",
        "rationale": (
            "Non-directional 'difference between A and B' always yields a non-negative "
            "result: subtract(larger, smaller). Prevents order-of-operands errors."
        ),
        "content": ABSOLUTE_DIFFERENCE_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "PCT_OF_TOTAL_PRIMER",
        "intent_label": "pct_of_total",
        "trigger": "Query contains 'as a percentage of' or 'percent of total'",
        "standard": "General financial ratio / percentage calculation",
        "rationale": (
            "Forces the final multiply(#0, 100) step so the answer is in percent form "
            "(e.g. 16.84), not decimal (0.1684). Prevents off-by-100 errors."
        ),
        "content": PCT_OF_TOTAL_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "GROWTH_RATE_PRIMER",
        "intent_label": "percent_change",
        "trigger": "Query contains 'growth rate', '% change', 'percent change', 'percentage change'",
        "standard": "General financial growth rate arithmetic",
        "rationale": (
            "Enforces three-step program (subtract, divide, multiply by 100) for percentage "
            "output. Includes property-and-equipment disambiguation (net vs gross)."
        ),
        "content": GROWTH_RATE_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "PERCENTAGE_AS_INTEGER_PRIMER",
        "intent_label": "percentage_0_100",
        "trigger": "Query asks for 'percentage decrease' or 'percentage increase'",
        "standard": "FinQA benchmark annotation — percentage in 0-100 form",
        "rationale": (
            "FinQA expects percentages as integers/decimals in 0-100 form, not raw ratios "
            "(0.9655 vs 96.55). This primer prevents the multiply step from being omitted."
        ),
        "content": PERCENTAGE_AS_INTEGER_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "PERCENT_REDUCTION_SIGN_PRIMER",
        "intent_label": "percent_reduction",
        "trigger": "Query contains 'percent reduction' or 'percentage reduction'",
        "standard": "Financial arithmetic — signed reduction convention",
        "rationale": (
            "Percent reduction must produce a negative result when new < old. "
            "Formula: subtract(new, old) / old — not subtract(old, new) / old."
        ),
        "content": PERCENT_REDUCTION_SIGN_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "PERCENT_CHANGE_BY_DIRECTION_PRIMER",
        "intent_label": "percent_change_by_direction",
        "trigger": "Query asks for percentage increase/decrease without 'percent reduction'",
        "standard": "General percent-change convention",
        "rationale": (
            "Fixes operand order: denominator is always the base (old) period. "
            "Three-step program with multiply(*100) enforced."
        ),
        "content": PERCENT_CHANGE_BY_DIRECTION_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "AVERAGE_SUBSET_PRIMER",
        "intent_label": "average_subset",
        "trigger": "Query contains 'average' over multiple periods",
        "standard": "FinQA average semantics — two-value exception and ambiguous divisor rule",
        "rationale": (
            "When exactly two values are present, divisor is unambiguous (divide by 2). "
            "When three or more values are present with no explicit formula, computation "
            "is deferred — FinQA GT may use a subset of listed years."
        ),
        "content": AVERAGE_SUBSET_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "CUMULATIVE_RETURN_PRIMER",
        "intent_label": "cumulative_return",
        "trigger": "Query references indexed returns, 'cumulative total return', '$100 invested', or outperformance",
        "standard": "Financial return calculation — base-100 indexed series",
        "rationale": (
            "Model without this primer subtracts raw index levels (e.g. 431 - 197) instead "
            "of computing (level - 100)/100 for each series then differencing the returns."
        ),
        "content": CUMULATIVE_RETURN_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "UNIT_SCALE_PRIMER",
        "intent_label": "unit_scale",
        "trigger": "Query specifies a unit scale ('in millions', 'in thousands')",
        "standard": "Financial reporting unit normalisation",
        "rationale": (
            "When prose gives values in millions and tables give raw thousands, model must "
            "prefer the unit that matches the question's requested scale."
        ),
        "content": UNIT_SCALE_PRIMER,
    },
    {
        "section": 3,
        "section_title": None,
        "name": "CROSS_YEAR_CARRY_FORWARD_PRIMER",
        "intent_label": "cross_year_carry_forward",
        "trigger": "Query asks for a dollar amount in year Y but context only states a percentage for year Y-1",
        "standard": "Financial filing concentration disclosure convention",
        "rationale": (
            "Prevents false INSUFFICIENT_DATA: when a percentage is disclosed for a prior year "
            "only, apply it to the query year's base. Standard practice in SEC filings."
        ),
        "content": CROSS_YEAR_CARRY_FORWARD_PRIMER,
    },
    # ------------------------------------------------------------------
    # Section 4: Loss Account Primers  (see SECTION_PREAMBLES[4])
    # ------------------------------------------------------------------
    {
        "section": 4,
        "section_title": "LOSS ACCOUNT PRIMERS  (IAS 1 / ASC 220)",
        "name": "LOSS_CHANGE_PRIMER",
        "intent_label": "loss_change",
        "trigger": "Query contains 'change in' AND any of: 'loss', 'net loss', 'losses'",
        "standard": "IAS 1 / ASC 220 — magnitude convention for period-over-period change in loss items",
        "rationale": (
            "Change in a loss account = subtract(magnitude_new, magnitude_old) using absolute "
            "values. Signed arithmetic reverses the direction: subtract(-15571, -24122) = +8551 "
            "(wrong); subtract(15571, 24122) = -8551 (correct: loss decreased by 8,551)."
        ),
        "content": LOSS_CHANGE_PRIMER,
    },
    {
        "section": 4,
        "section_title": None,
        "name": "LOSS_AVERAGE_PRIMER",
        "intent_label": "loss_average",
        "trigger": "Query contains 'average' AND any of: 'loss', 'net loss', 'losses'",
        "standard": "IAS 1 / ASC 220 — magnitude convention for averaging loss figures",
        "rationale": (
            "Composes with AVERAGE_SUBSET_PRIMER: the two-value exception unlocks arithmetic; "
            "this primer enforces that magnitudes (not signed negatives) are averaged. "
            "add(15571, 24122), divide(#0, 2) = 19,846.5 — not -19,846.5."
        ),
        "content": LOSS_AVERAGE_PRIMER,
    },
    {
        "section": 4,
        "section_title": None,
        "name": "LOSS_COMPARISON_PRIMER",
        "intent_label": "loss_comparison",
        "trigger": (
            "Query contains 'net loss' AND comparison language: "
            "'less than', 'greater than', 'exceed', 'below', 'above', 'smaller', 'larger', 'less '"
        ),
        "standard": (
            "IAS 1 / ASC 220 — natural negative balance of loss accounts; "
            "IAS 33 / ASC 260 — anti-dilutive EPS exclusion in loss periods"
        ),
        "rationale": (
            "Directly fixes Interview Example C (FP case above). 'Net loss less than X' "
            "in financial reporting means magnitude (abs value) below X, not signed comparison. "
            "Without this primer: -9,187 > -10,000 (mathematically true) → wrong answer 2019/2018. "
            "With primer: abs(-9,187) = 9,187 < 10,000 → correct answer 2017. "
            "Also relevant to anti-dilutive EPS exclusion (Example B): in a loss period, adding "
            "RSU shares to the diluted denominator makes loss per share look smaller — anti-dilutive "
            "and prohibited under IAS 33 / ASC 260."
        ),
        "content": LOSS_COMPARISON_PRIMER,
    },
    # ------------------------------------------------------------------
    # Section 5: Equity Compensation (ASC 718)
    # ------------------------------------------------------------------
    {
        "section": 5,
        "section_title": "EQUITY COMPENSATION PRIMERS  (ASC 718)",
        "name": "FINANCIAL_COMPENSATION_PRIMER",
        "intent_label": "compensation",
        "trigger": (
            "Query contains equity award / milestone language: "
            "'equity awards', 'performance milestones', 'stock-based compensation', "
            "'vesting', 'granted during'"
        ),
        "standard": "ASC 718 — Stock-Based Compensation; expense amortised over vesting period",
        "rationale": (
            "Directly fixes Interview Example A (FN case above). Under ASC 718-10-35-4, RSU "
            "grant-date fair value is amortised on a straight-line basis over the requisite "
            "service period. Year-1 recognised expense = fair_value / vesting_years (use "
            "document-stated period if available; default to / 4 as a conservative lower "
            "bound when not stated - US RSU norm is 3 years, so / 4 deliberately understates "
            "B). Model without this primer uses the full grant-date fair value (~$11M) as "
            "the right-side comparator, systematically overstating B and producing false "
            "'no' answers."
        ),
        "content": FINANCIAL_COMPENSATION_PRIMER,
    },
    # ------------------------------------------------------------------
    # Section 6: Specialised Domain
    # ------------------------------------------------------------------
    {
        "section": 6,
        "section_title": "SPECIALISED DOMAIN PRIMERS",
        "name": "FINQA_GOLDBLIND_NUMERICAL_PRIMER",
        "intent_label": "finqa_goldblind (FinQA dataset, always-on)",
        "trigger": "Injected on all FinQA calls alongside SHARED_NUMERICAL_PRIMER",
        "standard": "FinQA benchmark annotation conventions — multi-step chained programs",
        "rationale": (
            "Prevents gold-leakage heuristics: model must commit to a task type first. "
            "Enforces single-line chained programs (executor reads only the first candidate). "
            "Covers cumulative return, accounting adjustments, and unit preservation."
        ),
        "content": FINQA_GOLDBLIND_NUMERICAL_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "FINANCIAL_PARENTHETICAL_NEGATIVE_PRIMER",
        "intent_label": "financial_parenthetical (always-on for financial tables)",
        "trigger": "Always injected when processing financial table context",
        "standard": "SEC filing / GAAP parenthetical notation for negative values",
        "rationale": (
            "SEC filings show negative values as (X) in tables. Without this primer "
            "the model treats (X) as a positive absolute value and uses the wrong sign "
            "in calculations."
        ),
        "content": FINANCIAL_PARENTHETICAL_NEGATIVE_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "LOCOM_GROWTH_PRIMER",
        "intent_label": "locom_growth",
        "trigger": "Query contains 'growth rate' AND 'loans held-for-sale' or 'LOCOM'",
        "standard": "ASC 948 / GAAP — lower of cost or market measurement for held-for-sale loans",
        "rationale": (
            "Carried amount = min(aggregate cost, fair value) per LOCOM. Model must select "
            "the correct column for each year and compute growth on carried values, not cost."
        ),
        "content": LOCOM_GROWTH_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "INTEREST_PAYMENT_PRIMER",
        "intent_label": "interest_payment",
        "trigger": "Query asks for 'interest payment' or 'amount of interest' on a bond",
        "standard": "Bond interest arithmetic — semi-annual / quarterly payment frequency",
        "rationale": (
            "'Interest payment' means the periodic payment, not annual total. "
            "Divide annual interest by 2 (semi-annual) or 4 (quarterly) based on payment "
            "frequency stated in the note."
        ),
        "content": INTEREST_PAYMENT_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "LEASE_PERCENT_PRIMER",
        "intent_label": "lease_percent",
        "trigger": "Query asks for percent of total operating leases with specific term conditions",
        "standard": "ASC 840 / ASC 842 — operating lease expense and minimum lease payments",
        "rationale": (
            "Numerator is the narrative expense line for qualifying leases (not the sum of "
            "future payment schedule rows). Denominator = expense + total minimum lease payments."
        ),
        "content": LEASE_PERCENT_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "CASHFLOW_FINANCING_PRIMER",
        "intent_label": "cashflow_financing",
        "trigger": "Query asks about cash flow from financing, share repurchases, or net change in cash",
        "standard": "ASC 230 — Statement of Cash Flows; share repurchase column selection",
        "rationale": (
            "Total shares purchased (including employee surrenders for tax withholding) "
            "must be used, not just the board-authorised programme shares. Using the wrong "
            "column understates actual cash outflow."
        ),
        "content": CASHFLOW_FINANCING_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "EVENT_SCOPED_ARITHMETIC_PRIMER",
        "intent_label": "event_scoped",
        "trigger": "Query references acquired intangibles, amortization, or purchase price allocation",
        "standard": "ASC 805 — Business Combinations; event-scoped arithmetic",
        "rationale": (
            "When context contains multiple acquisition footnotes, use exactly one block "
            "unless the question explicitly asks for combined/total. Default to the block "
            "with larger magnitude when not specified."
        ),
        "content": EVENT_SCOPED_ARITHMETIC_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "ACCOUNTING_ADJUSTMENT_PRIMER",
        "intent_label": "accounting_adjustment",
        "trigger": "Query references cumulative-effect adjustment or accounting standard adoption",
        "standard": "ASC 250 — Accounting Changes and Error Corrections; ASC 606 transition",
        "rationale": (
            "Select only the single adjustment tied to the named standard. "
            "Percentage-of-adjustment questions use abs(adjustment)/base * 100. "
            "Preserve table units; do not rescale."
        ),
        "content": ACCOUNTING_ADJUSTMENT_PRIMER,
    },
    {
        "section": 6,
        "section_title": None,
        "name": "FREQUENCY_PROPORTION_PRIMER",
        "intent_label": "frequency_proportion",
        "trigger": "Query contains 'how often', 'how frequently', or 'how many times'",
        "standard": "FinQA benchmark — frequency as proportion, not raw count",
        "rationale": (
            "FinQA gold expects divide(count, total_days) not the raw count. "
            "If the question asks for a percentage, multiply the proportion by 100."
        ),
        "content": FREQUENCY_PROPORTION_PRIMER,
    },
]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

WIDE = "=" * 80
THIN = "-" * 80
MED  = "~" * 80

OVERVIEW = f"""\
OVERVIEW
{THIN}
This document contains the domain-specific reasoning primers used by the
Retrieval-Augmented Generation (RAG) pipeline in the ocr-agentic-rag project.

At inference time, the query is passed through a rule-based intent classifier
(classify_query_intent in orchestrator.py). The classifier fires one or more
intent labels, and the corresponding primers are appended to the generator
prompt. The base layer (SHARED_NUMERICAL_PRIMER) is always injected; intent
primers extend it.

Dataset coverage:
  These primers are shared across two financial numerical QA benchmarks: FinQA
  (Chen et al., EMNLP 2021) and TAT-QA (Zhu et al., ACL 2021). Both evaluate
  reasoning over SEC earnings report filings; the same domain-knowledge
  corrections apply to both.

  Credit risk memo generator (FinanceBench) shares the scoring methodology but
  uses a separate system-level primer (FINANCEBENCH_PRIMER in eval_runner.py)
  for model persona and answer style — that primer is not included here.

  Vision-language benchmarks (DocVQA, ChartQA, InfographicsVQA, MMMU) use
  separate evaluators with different normalisation tolerances suited to document
  text span and chart-read answer spaces. The primers in this file do not apply
  to vision evaluation.

Primer taxonomy:
  - Reasoning False Negatives (FN): model retrieves the correct document but
    applies an incorrect financial convention (e.g. ASC 718 expense vs. fair
    value, signed arithmetic on loss accounts, anti-dilutive EPS exclusion).
  - Reasoning False Positives (FP): model states a confident but wrong
    conclusion (e.g. interpreting $(9,187) net loss as positive in a signed
    comparison, missing natural-balance conventions).

Primers encode reasoning procedures. Where worked
examples appear in the primer text, they illustrate the reasoning procedure
using values drawn from the development corpus. The evaluation splits
(TAT-QA test, FinQA held-out) were held out from primer development — primers
were developed by analysing failure patterns on training examples, which is
methodologically equivalent to error-driven feature engineering in classical ML.
The intent classifier is purely rule-based (regex + keyword matching), ensuring
that evaluation metrics reflect retrieval and reasoning quality rather than
benchmark memorisation.

Total primers: {len(PRIMER_CATALOGUE)}
Source file:   rag_system/primers.py
Routing file:  rag_system/agentic/orchestrator.py
"""


def format_primer_block(entry: dict, position_in_section: int) -> str:
    lines = []
    lines.append(f"  Primer:     {entry['name']}")
    lines.append(f"  Intent:     {entry['intent_label']}")
    lines.append(f"  Trigger:    {entry['trigger']}")
    lines.append(f"  Standard:   {entry['standard']}")
    lines.append(f"  Rationale:  {entry['rationale']}")
    lines.append("")
    lines.append("  Prompt injected to generator:")
    lines.append("  " + MED[:76])
    # Indent primer content with 4 spaces
    for line in entry["content"].strip().splitlines():
        lines.append("    " + line)
    lines.append("  " + MED[:76])
    return "\n".join(lines)


def build_document(catalogue: list[dict], preambles: dict[int, str]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = []

    # Header
    parts.append(WIDE)
    parts.append("RAG FINANCIAL REASONING PRIMERS")
    parts.append("Credit Risk Document QA System — Ming Loon Lee")
    parts.append(f"Generated: {timestamp}")
    parts.append(WIDE)
    parts.append("")
    parts.append(OVERVIEW)

    # Group by section
    sections = {}
    for entry in catalogue:
        sections.setdefault(entry["section"], []).append(entry)

    for sec_num, entries in sorted(sections.items()):
        sec_title = next(e["section_title"] for e in entries if e["section_title"])
        parts.append("")
        parts.append(WIDE)
        parts.append(f"SECTION {sec_num} — {sec_title}")
        parts.append(WIDE)

        if sec_num in preambles:
            parts.append("")
            for line in preambles[sec_num].splitlines():
                parts.append(line)

        for i, entry in enumerate(entries):
            parts.append("")
            parts.append(THIN)
            parts.append(format_primer_block(entry, i))

    parts.append("")
    parts.append(WIDE)
    parts.append("END OF DOCUMENT")
    parts.append(f"Generated by: scripts/export_primers_txt.py")
    parts.append(f"Source:       rag_system/primers.py")
    parts.append(WIDE)
    parts.append("")

    return "\n".join(parts)


def main() -> None:
    output_path = Path("docs") / "rag_financial_domain_primers.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = build_document(PRIMER_CATALOGUE, SECTION_PREAMBLES)

    output_path.write_text(document, encoding="utf-8")
    size = output_path.stat().st_size
    print(
        f"[export_primers_txt] Wrote {output_path} "
        f"({len(PRIMER_CATALOGUE)} primers, {size:,} bytes)"
    )


if __name__ == "__main__":
    main()
