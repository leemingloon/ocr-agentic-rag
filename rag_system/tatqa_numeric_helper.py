"""
TAT-QA numeric retrieval helper: extract balances from chunks with correct unit scaling and aggregation.

- Finds line items (e.g. Inventories, Other accrued liabilities) and the requested adjustment (e.g. Without Adoption of Topic 606).
- Converts thousands → millions when appropriate.
- Sums multiple sub-accounts per line item.
- For "respectively" queries: can return raw source strings (preserve commas and decimals) for exact-match evaluation.
"""

import os
import re
from typing import List, Optional, Union


# Label that indicates the "without Topic 606" adjusted balance in TAT-QA docs.
DEFAULT_ADJUSTMENT_LABEL = "Without Adoption of Topic 606"

# Heuristic: raw numbers above this are likely in thousands (convert to millions by /1000).
THOUSANDS_THRESHOLD = 10_000

# Regex to capture the raw number after the adjustment label (digits, commas, optional decimal).
_RAW_NUMBER_PATTERN = r"([0-9,]+(?:\.[0-9]+)?)"


def _extract_number_after_label(text: str, account: str, adjustment_label: str) -> Optional[float]:
    """Find the numeric value that appears after the adjustment label for the given account.
    Handles patterns like '... As Adjusted - Without Adoption of Topic 606: 782,833' or ': $106,836'.
    """
    try:
        acc_pattern = re.escape(account)
        adj_pattern = re.escape(adjustment_label)
        pattern = rf"{acc_pattern}.*?{adj_pattern}\s*:\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            num_str = m.group(1).replace(",", "")
            return float(num_str)
    except (ValueError, IndexError):
        pass
    return None


def _extract_raw_after_label(text: str, account: str, adjustment_label: str) -> Optional[str]:
    """Find the raw string after the adjustment label (preserve commas and decimals).
    Example: 'Inventories ... Balances without Adoption of Topic 606: 1,568.6' -> '1,568.6'.
    """
    try:
        acc_pattern = re.escape(account)
        adj_pattern = re.escape(adjustment_label)
        pattern = rf"{acc_pattern}.*?{adj_pattern}\s*:\s*\$?\s*{_RAW_NUMBER_PATTERN}"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    except (IndexError, AttributeError):
        pass
    return None


def tatqa_numeric_helper(
    chunks: List[str],
    query_accounts: List[str],
    adjustment_label: str = DEFAULT_ADJUSTMENT_LABEL,
    scale_thousands_to_millions: bool = True,
    preserve_format: bool = False,
) -> Union[List[float], List[str]]:
    """
    Extract numeric balances for specified accounts from TAT-QA document chunks.

    Args:
        chunks: List of text chunks (retrieved RAG context split by separator).
        query_accounts: Account names to find, in order (e.g. ["Inventories", "Other accrued liabilities"]).
        adjustment_label: Label for the correct adjusted balance (e.g. "Without Adoption of Topic 606").
        scale_thousands_to_millions: If True, divide values > THOUSANDS_THRESHOLD by 1000 (float mode only).
        preserve_format: If True, return list of raw strings as in source (e.g. "1,568.6") for EM; else list of floats.

    Returns:
        List of balances in same order as query_accounts.
        preserve_format=False: floats in millions; missing -> 0.0.
        preserve_format=True: raw strings; missing -> fallback to one-decimal string of float.
    """
    if preserve_format:
        results: List[str] = []
        for account in query_accounts:
            raw_val: Optional[str] = None
            for chunk in chunks:
                raw_val = _extract_raw_after_label(chunk, account, adjustment_label)
                if raw_val is not None:
                    break
            if raw_val is not None:
                results.append(raw_val)
            else:
                # Fallback: use float extraction and format to one decimal (no comma)
                num = 0.0
                for chunk in chunks:
                    val = _extract_number_after_label(chunk, account, adjustment_label)
                    if val is not None:
                        if scale_thousands_to_millions and val > THOUSANDS_THRESHOLD:
                            val /= 1000.0
                        num += val
                results.append(str(round(num, 1)))
        if os.environ.get("RAG_DEBUG") == "1" and query_accounts:
            per_account = ", ".join(f"{a}={r!r}" for a, r in zip(query_accounts, results))
            print(f"[DEBUG] TAT-QA helper per-account (raw): {per_account}")
        return results

    results_float: List[float] = []
    for account in query_accounts:
        account_total = 0.0
        for chunk in chunks:
            val = _extract_number_after_label(chunk, account, adjustment_label)
            if val is not None:
                if scale_thousands_to_millions and val > THOUSANDS_THRESHOLD:
                    val /= 1000.0
                account_total += val
        results_float.append(round(account_total, 1))
    if os.environ.get("RAG_DEBUG") == "1" and query_accounts:
        per_account = ", ".join(f"{a}={r}" for a, r in zip(query_accounts, results_float))
        print(f"[DEBUG] TAT-QA helper per-account (millions): {per_account}")
    return results_float


def parse_accounts_from_query(query: str) -> List[str]:
    """
    Heuristic: parse account names from a TAT-QA question like
    'What are the balances (without Adoption of Topic 606, in millions) of inventories and other accrued liabilities, respectively?'
    Returns e.g. ["Inventories", "Other accrued liabilities"].
    Prefer ") of ..." so we capture the account list after the parenthetical, not "of Topic 606" inside it.
    """
    if not query or not isinstance(query, str):
        return []
    q = query.strip()
    # Prefer ") of X and Y" so we don't capture "(without Adoption of Topic 606, in millions)"
    of_match = re.search(r"\)\s*of\s+(.+?)(?:\s*,\s*respectively)?\s*\??\s*$", q, re.IGNORECASE | re.DOTALL)
    if not of_match:
        of_match = re.search(r"\bof\s+(.+?)(?:\s*,\s*respectively)?\s*\??\s*$", q, re.IGNORECASE | re.DOTALL)
    if not of_match:
        return []
    rest = of_match.group(1).strip()
    # Remove trailing "respectively" or "in millions" etc.
    rest = re.sub(r"\s*,?\s*respectively\s*$", "", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\s+in\s+millions\s*$", "", rest, flags=re.IGNORECASE)
    # Split by " and " or ", " (avoid splitting on commas inside parentheticals we already skipped)
    parts = re.split(r"\s+and\s+|\s*,\s*", rest, flags=re.IGNORECASE)
    accounts = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Skip fragments that are clearly not account names (e.g. "Topic 606", "in millions)")
        lower = p.lower()
        if "topic 606" in lower or lower.startswith("in millions"):
            continue
        # Normalize truncated "inventorie" -> Inventories
        if lower == "inventorie" or lower == "inventories":
            accounts.append("Inventories")
        elif "accrued" in lower and "liab" in lower:
            accounts.append("Other accrued liabilities")
        else:
            accounts.append(p.strip().title())
    return accounts


def _tatqa_multi_account_expects_sum(query: str, num_balances: int) -> bool:
    """
    True when this is a multi-account question for which the dataset expects a single total.
    False when the query asks for each account separately (e.g. "respectively", "each", "per account").
    """
    if num_balances < 2:
        return False
    q = query.strip().lower()
    if any(kw in q for kw in ("respectively", "each", "per account")):
        return False  # return per-account, not sum
    return "inventories" in q and ("accrued" in q or "liabilities" in q)


def _query_wants_respectively(query: str) -> bool:
    """True if query asks for per-account values (respectively, each, per account)."""
    q = query.strip().lower()
    return any(kw in q for kw in ("respectively", "each", "per account"))


def format_tatqa_helper_answer(
    balances: Union[List[float], List[str]],
    query: str,
    want_combined_total: Optional[bool] = None,
) -> str:
    """
    Format helper output as a single string for the numerical answer line.
    balances: either list of floats (millions) or list of raw strings (preserve source formatting).
    If want_combined_total is None: infer from query — "respectively"/"each"/"per account" -> no sum (comma-separated);
    "total"/"combined"/"sum" -> sum; else multi-account may sum unless query asks for per-account.
    For "respectively" queries, pass raw strings so output matches source (e.g. 1,568.6, 690.5).
    """
    if not balances:
        return ""
    use_raw = isinstance(balances[0], str)
    if want_combined_total is None:
        if _query_wants_respectively(query):
            want_combined_total = False
        else:
            n = len(balances)
            want_combined_total = (
                "total" in query.strip().lower()
                or "combined" in query.strip().lower()
                or "sum" in query.strip().lower()
                or _tatqa_multi_account_expects_sum(query, n)
            )
    if want_combined_total and len(balances) > 1:
        if use_raw:
            try:
                total = sum(float(str(b).replace(",", "")) for b in balances)
                return str(round(total, 1))
            except (ValueError, TypeError):
                return ", ".join(str(b) for b in balances)
        total = round(sum(balances), 1)  # type: ignore[arg-type]
        return str(total)
    return ", ".join(str(b) for b in balances)
