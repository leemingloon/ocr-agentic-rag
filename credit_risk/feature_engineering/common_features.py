"""
Common feature engineering for LendingClub-style PD (Probability of Default) prediction.

Shared by classical (XGBoost) and quantum (QSVM, VQC/QNN) PD models. Consumes either:
- A pandas DataFrame with LoanStats3a-style columns (from CSV), or
- A dict of raw fields (e.g. parsed from LendingClub benchmark query text).

Two modes:
- build_features_from_dict: 15 features (includes grade_num, int_rate_high; legacy/benchmark).
- build_features_from_dict_no_leakage: bank-grade, origination-only features (excludes grade,
  sub_grade, int_rate as "using the answer"; adds revol_util_bucket, credit_history_months,
  purpose_risk_code, home_ownership_risk_code, dti_to_income, payment_to_income_monthly,
  purpose_x_home_ownership). Use for training and production PD.
"""

from __future__ import annotations

import re
from datetime import datetime
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
    "earliestcrline": "earliest_cr_line",
    "issuedate": "issue_d",
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
    Benchmark format often has "Text: ' addressState: fl, annualIncome: 97000, ... '." — we extract
    only that segment so key: value pairs are parsed correctly (no spurious matches from "Answer:", etc.).
    """
    out: Dict[str, Any] = {}
    if not query or not isinstance(query, str):
        return out
    # Extract the feature block between "Text: '" and "'." or "'\n" if present
    text_match = re.search(r"Text:\s*['\"]([^'\"]+)['\"]", query, re.IGNORECASE | re.DOTALL)
    parse_text = text_match.group(1).strip() if text_match else query
    # Extract "key: value" pairs; key may be camelCase, value numeric or string
    pattern = r"(\w+)\s*:\s*([^,\n]+)"
    for m in re.finditer(pattern, parse_text, re.IGNORECASE):
        key_raw = m.group(1).strip().lower()
        val_raw = m.group(2).strip().strip("'\" ")
        canonical = QUERY_FIELD_ALIASES.get(key_raw, key_raw)
        # Keep string values for fields that need categorical/date/employment parsing
        if canonical in (
            "addr_state",
            "grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "application_type",
            "issue_d",
            "earliest_cr_line",
            "emp_length",  # e.g. "10+Years", "<1Year" -> _emp_length_to_months()
        ):
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
    m = re.search(r"(\d+)\s*years?", t, re.IGNORECASE)
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


def _parse_lendingclub_date(s: Any) -> Optional[datetime]:
    """Parse LendingClub date string (e.g. 'Dec-2018', 'Jan-2015') to datetime. Returns None if invalid."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip()
    if not t:
        return None
    try:
        return datetime.strptime(t, "%b-%Y")
    except ValueError:
        return None


# Purpose risk: higher = typically more default-prone (domain heuristic)
PURPOSE_RISK = {
    "debt_consolidation": 2,
    "credit_card": 2,
    "other": 1,
    "home_improvement": 1,
    "major_purchase": 1,
    "small_business": 2,
    "car": 1,
    "medical": 1,
    "wedding": 0,
    "moving": 1,
    "vacation": 0,
    "house": 1,
    "renewable_energy": 1,
    "educational": 1,
}

# Home ownership risk: rent > mortgage > own (typical ordering)
HOME_OWNERSHIP_RISK = {
    "rent": 2,
    "mortgage": 1,
    "own": 0,
    "any": 0,
    "none": 0,
    "other": 1,
}


def _purpose_risk_code(s: Any) -> float:
    """Map purpose string to numeric risk code (0-2). Missing -> 1."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 1.0
    key = str(s).strip().lower().replace(" ", "_")
    return float(PURPOSE_RISK.get(key, 1))


def _home_ownership_risk_code(s: Any) -> float:
    """Map home_ownership to numeric risk code. Missing -> 1."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 1.0
    key = str(s).strip().lower()
    return float(HOME_OWNERSHIP_RISK.get(key, 1))


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

    # 2. Installment-to-income ratio (annual: installment / annual_inc)
    if annual_inc and annual_inc > 0 and not np.isnan(installment):
        installment_to_income = installment / annual_inc
    else:
        installment_to_income = np.nan

    # 2b. Payment burden: monthly payment / monthly income = installment / (annual_inc/12)
    if annual_inc and annual_inc > 0 and not np.isnan(installment):
        payment_to_income_monthly = (installment * 12.0) / annual_inc
    else:
        payment_to_income_monthly = np.nan

    # 2c. DTI as leverage (already a ratio; normalize to decimal if stored as percent)
    dti_to_income = dti_ratio if not np.isnan(dti_ratio) else np.nan
    if dti_to_income is not None and not np.isnan(dti_to_income) and dti_to_income > 1.0:
        dti_to_income = dti_to_income / 100.0  # e.g. 18.5 -> 0.185

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
        "dti_to_income": dti_to_income,
        "installment_to_income": installment_to_income,
        "payment_to_income_monthly": payment_to_income_monthly,
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


# Ordered list of feature names for no-leakage (origination-only) PD model.
# New HF-compatible derived features: log_annual_inc, loan_to_income_bucket, fico_range,
# fico_bucket, emp_length_years, emp_home_interaction, revol_util_per_acc.
FEATURE_NAMES_NO_LEAKAGE = [
    "dti_ratio",
    "dti_to_income",
    "installment_to_income",
    "payment_to_income_monthly",
    "fico_mid",
    "revol_util_pct",
    "revol_util_bucket",
    "loan_to_income",
    "loan_to_income_bucket",
    "emp_months",
    "emp_length_years",
    "delinq_flag",
    "inq_6m",
    "acc_util",
    "revol_bal_log",
    "revol_util_per_acc",
    "fico_subprime",
    "fico_range",
    "fico_bucket",
    "dti_high",
    "credit_history_months",
    "purpose_risk_code",
    "home_ownership_risk_code",
    "purpose_x_home_ownership",
    "emp_home_interaction",
    "log_annual_inc",
]


def build_features_from_dict_no_leakage(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Build origination-only PD features (no grade, sub_grade, int_rate).
    Adds: revol_util_bucket, credit_history_months, purpose_risk_code, home_ownership_risk_code.
    Use for training and production to avoid leakage and "using the answer".
    """
    feats = build_features_from_dict(raw)
    get = lambda k: raw.get(k, np.nan)
    getf = lambda k: _safe_float(get(k))

    revol_util = getf("revol_util")
    revol_util_pct = feats["revol_util_pct"]
    if revol_util_pct is not None and not np.isnan(revol_util_pct):
        if revol_util_pct < 0.3:
            revol_util_bucket = 0.0
        elif revol_util_pct <= 0.6:
            revol_util_bucket = 1.0
        else:
            revol_util_bucket = 2.0
    else:
        revol_util_bucket = np.nan

    issue_d = _parse_lendingclub_date(get("issue_d"))
    earliest_cr = _parse_lendingclub_date(get("earliest_cr_line"))
    if issue_d and earliest_cr:
        delta = (issue_d - earliest_cr).days
        credit_history_months = delta / 30.4375  # approximate month length
        if credit_history_months < 0:
            credit_history_months = 0.0
    else:
        credit_history_months = np.nan

    purpose_risk_code = _purpose_risk_code(get("purpose"))
    home_ownership_risk_code = _home_ownership_risk_code(get("home_ownership"))
    purpose_x_home_ownership = purpose_risk_code * home_ownership_risk_code

    # New HF-compatible derived features (origination-only)
    annual_inc = getf("annual_inc")
    log_annual_inc = np.log1p(max(0.0, annual_inc)) if annual_inc is not None and not np.isnan(annual_inc) and annual_inc >= 0 else np.nan

    loan_to_income = feats.get("loan_to_income", np.nan)
    if loan_to_income is not None and not np.isnan(loan_to_income):
        if loan_to_income < 0.2:
            loan_to_income_bucket = 0.0
        elif loan_to_income <= 0.5:
            loan_to_income_bucket = 1.0
        elif loan_to_income <= 1.0:
            loan_to_income_bucket = 2.0
        else:
            loan_to_income_bucket = 3.0
    else:
        loan_to_income_bucket = np.nan

    fico_low = getf("fico_low")
    fico_high = getf("fico_high")
    fico_mid = feats.get("fico_mid", np.nan)
    if fico_low is not None and fico_high is not None and not np.isnan(fico_low) and not np.isnan(fico_high):
        fico_range = fico_high - fico_low
    else:
        fico_range = np.nan

    if fico_mid is not None and not np.isnan(fico_mid):
        fico_bucket = float(int(fico_mid // 20))
    else:
        fico_bucket = np.nan

    emp_months = feats.get("emp_months", np.nan)
    emp_length_years = emp_months / 12.0 if emp_months is not None and not np.isnan(emp_months) else np.nan

    if emp_length_years is not None and not np.isnan(emp_length_years) and home_ownership_risk_code is not None and not np.isnan(home_ownership_risk_code):
        emp_bucket = int(emp_length_years)
        emp_home_interaction = emp_bucket * home_ownership_risk_code
    else:
        emp_home_interaction = np.nan

    total_acc = getf("total_acc")
    revol_util_per_acc = revol_util_pct / max(total_acc, 1.0) if (revol_util_pct is not None and not np.isnan(revol_util_pct) and total_acc is not None and not np.isnan(total_acc)) else np.nan

    out = {k: feats[k] for k in FEATURE_NAMES_NO_LEAKAGE if k in feats}
    out["revol_util_bucket"] = revol_util_bucket
    out["credit_history_months"] = credit_history_months
    out["purpose_risk_code"] = purpose_risk_code
    out["home_ownership_risk_code"] = home_ownership_risk_code
    out["purpose_x_home_ownership"] = purpose_x_home_ownership
    out["loan_to_income_bucket"] = loan_to_income_bucket
    out["fico_range"] = fico_range
    out["fico_bucket"] = fico_bucket
    out["emp_length_years"] = emp_length_years
    out["emp_home_interaction"] = emp_home_interaction
    out["revol_util_per_acc"] = revol_util_per_acc
    out["log_annual_inc"] = log_annual_inc
    # Ensure order and only no-leakage keys (drop grade_num, int_rate_high, composite_risk)
    return {k: out[k] for k in FEATURE_NAMES_NO_LEAKAGE if k in out}


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


def build_features_from_dataframe_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Build origination-only (no-leakage) PD features from a DataFrame."""
    raw_list = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        raw_list.append(build_features_from_dict_no_leakage(raw))
    return pd.DataFrame(raw_list)


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names produced by build_*."""
    return list(build_features_from_dict({}).keys())


def get_feature_names_no_leakage() -> List[str]:
    """Return the ordered list of origination-only (no-leakage) feature names."""
    return list(FEATURE_NAMES_NO_LEAKAGE)


def parse_query_to_features(
    query: str, fill_missing: float = 0.0, use_no_leakage: bool = False
) -> Dict[str, float]:
    """
    Parse LendingClub benchmark query text and return a feature dict suitable for predict_pd.
    Replaces NaN with fill_missing so the model always receives numeric values.
    If use_no_leakage=True, returns origination-only features (same set as get_feature_names_no_leakage).
    """
    raw = _parse_lendingclub_query_text(query)
    feats = (
        build_features_from_dict_no_leakage(raw)
        if use_no_leakage
        else build_features_from_dict(raw)
    )
    for k in feats:
        v = feats[k]
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            feats[k] = fill_missing
    return feats
