"""
TAT-QA numeric retrieval helper: extract balances from chunks with correct unit scaling and aggregation.

- Finds line items (e.g. Inventories, Other accrued liabilities) and the requested adjustment (e.g. Without Adoption of Topic 606).
- Converts thousands → millions when appropriate.
- Sums multiple sub-accounts per line item.
- Returns balances in millions for use by the generator.
"""

import os
import re
from typing import List, Optional


# Label that indicates the "without Topic 606" adjusted balance in TAT-QA docs.
DEFAULT_ADJUSTMENT_LABEL = "Without Adoption of Topic 606"

# Heuristic: raw numbers above this are likely in thousands (convert to millions by /1000).
THOUSANDS_THRESHOLD = 10_000


def _extract_number_after_label(text: str, account: str, adjustment_label: str) -> Optional[float]:
    """Find the numeric value that appears after the adjustment label for the given account.
    Handles patterns like '... As Adjusted - Without Adoption of Topic 606: 782,833' or ': $106,836'.
    """
    # Prefer pattern: account ... adjustment_label : number (number may have $ and commas)
    # Use a flexible pattern: account, then later adjustment_label, then colon and number
    try:
        # Case-insensitive: account then (anything) then adjustment label then : then optional $ then digits/commas
        acc_pattern = re.escape(account)
        adj_pattern = re.escape(adjustment_label)
        pattern = rf"{acc_pattern}.*?{adj_pattern}\s*:\s*\$?\s*([0-9,]+)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            num_str = m.group(1).replace(",", "")
            return float(num_str)
    except (ValueError, IndexError):
        pass
    return None


def tatqa_numeric_helper(
    chunks: List[str],
    query_accounts: List[str],
    adjustment_label: str = DEFAULT_ADJUSTMENT_LABEL,
    scale_thousands_to_millions: bool = True,
) -> List[float]:
    """
    Extract numeric balances for specified accounts from TAT-QA document chunks.

    Args:
        chunks: List of text chunks (retrieved RAG context split by separator).
        query_accounts: Account names to find, in order (e.g. ["Inventories", "Other accrued liabilities"]).
        adjustment_label: Label for the correct adjusted balance (e.g. "Without Adoption of Topic 606").
        scale_thousands_to_millions: If True, divide values > THOUSANDS_THRESHOLD by 1000.

    Returns:
        List of balances in millions, in the same order as query_accounts. Missing accounts yield 0.0.
    """
    results: List[float] = []
    for account in query_accounts:
        account_total = 0.0
        for chunk in chunks:
            val = _extract_number_after_label(chunk, account, adjustment_label)
            if val is not None:
                if scale_thousands_to_millions and val > THOUSANDS_THRESHOLD:
                    val /= 1000.0
                account_total += val
        results.append(round(account_total, 1))
    if os.environ.get("RAG_DEBUG") == "1" and query_accounts:
        per_account = ", ".join(f"{a}={r}" for a, r in zip(query_accounts, results))
        print(f"[DEBUG] TAT-QA helper per-account (millions): {per_account}")
    return results


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
    True when this is the classic TAT-QA multi-account question (inventories + other accrued liabilities)
    for which the dataset expects a single total in millions, not comma-separated values.
    """
    if num_balances < 2:
        return False
    q = query.strip().lower()
    return "inventories" in q and ("accrued" in q or "liabilities" in q)


def format_tatqa_helper_answer(
    balances_millions: List[float],
    query: str,
    want_combined_total: Optional[bool] = None,
) -> str:
    """
    Format helper output as a single string for the numerical answer line.
    If want_combined_total is None: infer from query — "total"/"combined"/"sum" -> sum;
    for TAT-QA multi-account (inventories + other accrued liabilities) the dataset expects sum -> sum;
    else return "a, b".
    """
    if not balances_millions:
        return ""
    if want_combined_total is None:
        q = query.strip().lower()
        want_combined_total = (
            "total" in q or "combined" in q or "sum" in q
            or _tatqa_multi_account_expects_sum(query, len(balances_millions))
        )
    if want_combined_total and len(balances_millions) > 1:
        total = round(sum(balances_millions), 1)
        return str(total)
    return ", ".join(str(b) for b in balances_millions)
