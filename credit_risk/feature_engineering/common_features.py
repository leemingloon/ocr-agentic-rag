"""
Common feature engineering for LendingClub-style PD (Probability of Default) prediction.

Shared by classical (XGBoost) and quantum (QSVM, VQC/QNN) PD models. Consumes either:
- A pandas DataFrame with LoanStats3a-style columns (from CSV), or
- A dict of raw fields (e.g. parsed from LendingClub benchmark query text).

Fifteen high-impact features: interactions, ratios, binning, domain-inspired.
Designed for PD prediction and compatible with streaming evaluation (one row at a time).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# CamelCase / variant names in benchmark query text -> canonical snake_case for feature pipeline
QUERY_FIELD_ALIASES = {
    "addressstate": "addr_state",
    "annualincome": "annual_inc",
    "delinquencyin2years": "delinq_2yrs",
    "employmentlength": "emp_length",
    "ficorangehigh": "fico_high",
    "ficorangelow": "fico_low",
    "grade": "grade",
    "homeownership": "home_ownership",
    "inquiriesin6months": "inq_last_6mths",
    "installment": "installment",
    "interestrate": "int_rate",
    "lastpaymentamount": "last_pymnt_amnt",
    "loanamount": "loan_amnt",
    "loanapplicationtype": "application_type",
    "loanpurpose": "purpose",
    "mortgageaccounts": "mort_acc",
    "openaccounts": "open_acc",
    "revolvingbalance": "revol_bal",
    "revolvingutilizationrate": "revol_util",
    "totalaccounts": "total_acc",
    "verificationstatus": "verification_status",
}


def _safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace(",", "").replace("%", "").replace("$", "")
    if not s or s in ("", "nan", "none", "n/a"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_lendingclub_query_text(query: str) -> Dict[str, Any]:
    """
    Parse LendingClub benchmark query string into a flat dict of raw fields.
    Example: "addressState: ga, annualIncome: 66400.00, ..." -> {"addr_state": "ga", "annual_inc": 66400.0, ...}
    """
    out: Dict[str, Any] = {}
    if not query or not isinstance(query, str):
        return out
    # Extract "key: value" pairs; key may be camelCase, value numeric or string
    pattern = r"(\w+)\s*:\s*([^,\n]+)"
    for m in re.finditer(pattern, query, re.IGNORECASE):
        key_raw = m.group(1).strip().lower()
        val_raw = m.group(2).strip().strip("'\"")
        canonical = QUERY_FIELD_ALIASES.get(key_raw, key_raw)
        if canonical in ("addr_state", "grade", "home_ownership", "verification_status", "purpose", "application_type"):
            out[canonical] = val_raw
        else:
            out[canonical] = _safe_float(val_raw)
    return out


def _emp_length_to_months(s: Any) -> float:
    """Map employment length string to numeric months (e.g. '10+Years' -> 120, '<1Year' -> 6)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip().lower()
    if not t:
        return np.nan
    if "10+" in t or "10 +" in t:
        return 120.0
    if "<1" in t or "n/a" in t:
        return 6.0
    m = re.search(r"(\d+)\s*year", t)
    if m:
        return float(m.group(1)) * 12.0
    return np.nan


def _grade_to_ordinal(s: Any) -> float:
    """Map LendingClub grade (A-G) to ordinal 1-7, NaN if missing."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    c = str(s).strip().upper()
    if len(c) >= 1 and c[0] in "ABCDEFG":
        return float(ord(c[0]) - ord("A") + 1)
    return np.nan


def build_features_from_dict(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Build 15 PD features from a single row (dict of raw fields).
    Used when streaming parquet rows or when inference receives one sample.
    Missing keys yield NaN; caller or downstream fills with 0 or median as needed.
    """
    get = lambda k: raw.get(k, np.nan)
    getf = lambda k: _safe_float(get(k))

    annual_inc = getf("annual_inc")
    loan_amnt = getf("loan_amnt")
    int_rate = getf("int_rate")
    installment = getf("installment")
    dti = getf("dti")  # may not be in benchmark query; allow from CSV
    delinq_2yrs = getf("delinq_2yrs")
    revol_util = getf("revol_util")
    revol_bal = getf("revol_bal")
    open_acc = getf("open_acc")
    total_acc = getf("total_acc")
    inq_last_6mths = getf("inq_last_6mths")
    fico_low = getf("fico_low")
    fico_high = getf("fico_high")
    emp_len_months = _emp_length_to_months(get("emp_length"))
    grade_ord = _grade_to_ordinal(get("grade"))

    # 1. Debt-to-income (use as-is if present)
    dti_ratio = dti if not np.isnan(dti) else np.nan

    # 2. Installment-to-income ratio
    if annual_inc and annual_inc > 0 and not np.isnan(installment):
        installment_to_income = installment / annual_inc
    else:
        installment_to_income = np.nan

    # 3. FICO mid (average of low/high)
    if not np.isnan(fico_low) and not np.isnan(fico_high):
        fico_mid = (fico_low + fico_high) / 2.0
    elif not np.isnan(fico_low):
        fico_mid = fico_low
    elif not np.isnan(fico_high):
        fico_mid = fico_high
    else:
        fico_mid = np.nan

    # 4. Revolving utilization (as decimal 0-1)
    revol_util_pct = revol_util / 100.0 if revol_util is not None and not np.isnan(revol_util) else np.nan

    # 5. Loan-to-income ratio
    if annual_inc and annual_inc > 0 and not np.isnan(loan_amnt):
        loan_to_income = loan_amnt / annual_inc
    else:
        loan_to_income = np.nan

    # 6. Grade ordinal (1-7)
    grade_num = grade_ord

    # 7. Employment length (months)
    emp_months = emp_len_months

    # 8. Delinquency flag (binary)
    delinq_flag = 1.0 if (delinq_2yrs is not None and not np.isnan(delinq_2yrs) and delinq_2yrs > 0) else 0.0

    # 9. Inquiries in 6m (count)
    inq_6m = inq_last_6mths if inq_last_6mths is not None and not np.isnan(inq_last_6mths) else 0.0

    # 10. Accounts utilization (open/total); avoid div by zero
    if total_acc and total_acc > 0 and open_acc is not None and not np.isnan(open_acc):
        acc_util = open_acc / total_acc
    else:
        acc_util = np.nan

    # 11. Interest rate bin (high >= 15%)
    int_rate_high = 1.0 if (int_rate is not None and not np.isnan(int_rate) and int_rate >= 15.0) else 0.0

    # 12. Revolving balance (log1p for scale)
    revol_bal_log = np.log1p(revol_bal) if revol_bal is not None and not np.isnan(revol_bal) and revol_bal >= 0 else np.nan

    # 13. FICO low bin (subprime < 660)
    fico_subprime = 1.0 if (fico_mid is not None and not np.isnan(fico_mid) and fico_mid < 660) else 0.0

    # 14. DTI high (>= 20% or missing as risky)
    dti_high = 1.0 if (dti_ratio is not None and not np.isnan(dti_ratio) and dti_ratio >= 20.0) else (1.0 if np.isnan(dti_ratio) else 0.0)

    # 15. Composite risk score (simple weighted sum for interpretability)
    comp = 0.0
    if not np.isnan(grade_num):
        comp += (grade_num - 1) / 6.0  # 0-1
    comp += 0.2 * delinq_flag
    comp += 0.1 * min(inq_6m / 5.0, 1.0) if not np.isnan(inq_6m) else 0
    comp += 0.2 * int_rate_high
    comp += 0.2 * fico_subprime
    comp += 0.15 * dti_high
    if not np.isnan(revol_util_pct):
        comp += 0.15 * min(revol_util_pct, 1.0)
    composite_risk = min(comp, 1.0)

    return {
        "dti_ratio": dti_ratio,
        "installment_to_income": installment_to_income,
        "fico_mid": fico_mid,
        "revol_util_pct": revol_util_pct,
        "loan_to_income": loan_to_income,
        "grade_num": grade_num,
        "emp_months": emp_months,
        "delinq_flag": delinq_flag,
        "inq_6m": inq_6m,
        "acc_util": acc_util,
        "int_rate_high": int_rate_high,
        "revol_bal_log": revol_bal_log,
        "fico_subprime": fico_subprime,
        "dti_high": dti_high,
        "composite_risk": composite_risk,
    }


def build_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 15 PD features from a DataFrame with LoanStats3a-style columns.
    Column names: snake_case (annual_inc, loan_amnt, int_rate, installment, dti,
    delinq_2yrs, revol_util, revol_bal, open_acc, total_acc, inq_last_6mths,
    emp_length, grade; fico from fico_range_low/fico_range_high or equivalent).
    """
    raw_list = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        raw_list.append(build_features_from_dict(raw))
    out = pd.DataFrame(raw_list)
    return out


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names produced by build_*."""
    return list(build_features_from_dict({}).keys())


def parse_query_to_features(query: str, fill_missing: float = 0.0) -> Dict[str, float]:
    """
    Parse LendingClub benchmark query text and return a feature dict suitable for predict_pd.
    Replaces NaN with fill_missing so the model always receives numeric values.
    """
    raw = _parse_lendingclub_query_text(query)
    feats = build_features_from_dict(raw)
    for k in feats:
        v = feats[k]
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            feats[k] = fill_missing
    return feats
