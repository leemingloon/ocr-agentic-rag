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

V2 feature set (additive, still no-leakage):
- Adds one-hot encodings for key categoricals (purpose, home_ownership, verification_status,
  addr_state, application_type, initial_list_status). Excludes grade/sub_grade/int_rate.
- Adds a small set of raw origination numerics when present (e.g. term_months).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Post-origination fields to NEVER use (data leakage — only exist after loan is active)
POST_ORIGINATION_BLOCKLIST = {
    "lastpaymentamount",
    "last_pymnt_amnt",
    "lastpaymentdate",
    "last_pymnt_d",
    "total_pymnt",
    "totalpayment",
    "recoveries",
    "outstandingprincipal",
    "outstanding_principal",
    "outprincipal",
    "out_prcp",
    "next_pymnt_d",
    "collection_recovery_fee",
}

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
    # lastPaymentAmount intentionally EXCLUDED — post-origination, causes leakage
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


def _strip_blocklist_from_query_text(text: str) -> str:
    """
    Remove post-origination key: value pairs from the feature block string.
    Belt-and-suspenders: ensures blocklist fields never reach the parser output.
    """
    if not text or not isinstance(text, str):
        return text
    # Match "key: value" pairs; remove if key (lowercase) is in blocklist
    def _repl(m: re.Match) -> str:
        key_lower = m.group(1).strip().lower()
        if key_lower in POST_ORIGINATION_BLOCKLIST:
            return ""  # Remove this pair (leave trailing comma for next pair)
        return m.group(0)
    pattern = r"(\w+)\s*:\s*[^,\n]+"
    result = re.sub(pattern, _repl, text, flags=re.IGNORECASE)
    # Clean up double commas or ", ," left by removals
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r"^\s*,\s*", "", result)
    result = re.sub(r",\s*$", "", result)
    return result


def _parse_lendingclub_query_text(query: str) -> Dict[str, Any]:
    """
    Parse LendingClub benchmark query string into a flat dict of raw fields.
    Benchmark format often has "Text: ' addressState: fl, annualIncome: 97000, ... '." — we extract
    only that segment so key: value pairs are parsed correctly (no spurious matches from "Answer:", etc.).
    Post-origination fields (lastPaymentAmount, etc.) are stripped from the string before parsing.
    """
    out: Dict[str, Any] = {}
    if not query or not isinstance(query, str):
        return out
    # Extract the feature block between "Text: '" and "'." or "'\n" if present
    text_match = re.search(r"Text:\s*['\"]([^'\"]+)['\"]", query, re.IGNORECASE | re.DOTALL)
    parse_text = text_match.group(1).strip() if text_match else query
    # Strip blocklist fields from string before parsing (ensures no leakage)
    parse_text = _strip_blocklist_from_query_text(parse_text)
    # Extract "key: value" pairs; key may be camelCase, value numeric or string
    pattern = r"(\w+)\s*:\s*([^,\n]+)"
    for m in re.finditer(pattern, parse_text, re.IGNORECASE):
        key_raw = m.group(1).strip().lower()
        if key_raw in POST_ORIGINATION_BLOCKLIST:
            continue  # Skip post-origination fields (e.g. lastPaymentAmount) — leakage
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

# ----------------------------
# V2 categorical vocabularies
# ----------------------------
# NOTE: These are intentionally "small and stable" vocabularies so inference can be deterministic.
# Unknown / missing values map to UNK (i.e., all zeros except addr_state__UNK where applicable).

_PURPOSE_VOCAB = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
    "other",
]

_HOME_OWNERSHIP_VOCAB = ["rent", "mortgage", "own", "any", "none", "other"]
_VERIFICATION_STATUS_VOCAB = ["verified", "source_verified", "not_verified"]
_APPLICATION_TYPE_VOCAB = ["individual", "joint_app"]
_INITIAL_LIST_STATUS_VOCAB = ["f", "w"]

# 50 states + DC; (no territories by default). Keep stable for training/inference.
_ADDR_STATE_VOCAB = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY",
]


def _norm_verification_status(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    # LC uses "Source Verified" / "Not Verified" / "Verified"
    t = t.replace("-", " ").replace("/", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if t == "source verified":
        return "source_verified"
    if t == "not verified":
        return "not_verified"
    if t == "verified":
        return "verified"
    return None


def _norm_application_type(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    t = re.sub(r"\s+", " ", t)
    if t in ("individual",):
        return "individual"
    if t in ("joint app", "joint"):
        return "joint_app"
    return None


def _norm_initial_list_status(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if t in ("f", "w"):
        return t
    return None


def _norm_home_ownership(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    t = t.replace("-", "_").replace(" ", "_")
    # map common LC variants
    if t in ("rent", "mortgage", "own", "any", "none", "other"):
        return t
    return None


def _norm_purpose(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = _camel_to_snake(s)
    t = str(t).strip().lower()
    if not t:
        return None
    # map common variants
    if t in _PURPOSE_VOCAB:
        return t
    # Some datasets use 'debt_consolidation' vs 'debt_consolidation' already handled.
    return "other"


def _norm_addr_state(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper()
    if not t:
        return None
    if t in _ADDR_STATE_VOCAB:
        return t
    return "UNK"


def _parse_term_months(x: Any) -> float:
    """Parse LC term like '36 months' -> 36. NaN if missing/invalid."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        v = float(x)
        return v if v > 0 else np.nan
    t = str(x).strip().lower()
    if not t:
        return np.nan
    m = re.search(r"(\d+)", t)
    if not m:
        return np.nan
    try:
        v = float(m.group(1))
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def _camel_to_snake(s: str) -> str:
    """Convert camelCase or PascalCase to snake_case for HF benchmark compatibility."""
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", str(s)).lower().replace(" ", "_")


def _purpose_risk_code(s: Any) -> float:
    """Map purpose string to numeric risk code (0-2). Missing -> 1. HF uses camelCase (e.g. debtConsolidation)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 1.0
    key = _camel_to_snake(s)
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
    Post-origination fields (last_pymnt_amnt, etc.) are explicitly dropped to prevent leakage.
    """
    raw = {k: v for k, v in raw.items() if k.lower() not in POST_ORIGINATION_BLOCKLIST}
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

    # 9b. Derogatory composite: delinquency + inquiries (credit history depth)
    derog_composite = (delinq_2yrs if delinq_2yrs is not None and not np.isnan(delinq_2yrs) else 0.0) + inq_6m

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
        "derog_composite": derog_composite,
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
    "derog_composite",
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
    "is_missing_dti",
    "is_missing_credit_history",
]


def _v2_onehot_feature_names() -> list[str]:
    names: list[str] = []
    names += [f"purpose__{c}" for c in _PURPOSE_VOCAB]
    names += [f"home_ownership__{c}" for c in _HOME_OWNERSHIP_VOCAB]
    names += [f"verification_status__{c}" for c in _VERIFICATION_STATUS_VOCAB]
    names += [f"application_type__{c}" for c in _APPLICATION_TYPE_VOCAB]
    names += [f"initial_list_status__{c}" for c in _INITIAL_LIST_STATUS_VOCAB]
    # addr_state: fixed list + UNK bucket
    names += [f"addr_state__{c}" for c in _ADDR_STATE_VOCAB] + ["addr_state__UNK"]
    return names


FEATURE_NAMES_NO_LEAKAGE_V2 = (
    list(FEATURE_NAMES_NO_LEAKAGE)
    + [
        # Raw origination numerics (when present)
        "term_months",
        "open_acc_raw",
        "total_acc_raw",
        "pub_rec_raw",
        "pub_rec_bankruptcies_raw",
        "mort_acc_raw",
        "mo_sin_old_il_acct_raw",
        "mo_sin_old_rev_tl_op_raw",
    ]
    + _v2_onehot_feature_names()
)


def build_features_from_dict_no_leakage_v2(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    V2 additive no-leakage feature set:
    - Starts from build_features_from_dict_no_leakage (V1).
    - Adds stable one-hot encodings for key categoricals.
    - Adds a few raw origination numeric fields when present (kept as NaN if missing).

    Still excludes grade/sub_grade/int_rate and post-origination fields.
    """
    # Base v1 no-leakage features
    out = build_features_from_dict_no_leakage(raw)

    get = lambda k: raw.get(k, np.nan)
    getf = lambda k: _safe_float(get(k))

    # Raw numeric origination fields (only when present; NaN otherwise)
    out["term_months"] = _parse_term_months(get("term"))
    out["open_acc_raw"] = getf("open_acc")
    out["total_acc_raw"] = getf("total_acc")
    out["pub_rec_raw"] = getf("pub_rec")
    out["pub_rec_bankruptcies_raw"] = getf("pub_rec_bankruptcies")
    out["mort_acc_raw"] = getf("mort_acc")
    out["mo_sin_old_il_acct_raw"] = getf("mo_sin_old_il_acct")
    out["mo_sin_old_rev_tl_op_raw"] = getf("mo_sin_old_rev_tl_op")

    # One-hot categoricals (stable vocab)
    purpose = _norm_purpose(get("purpose"))
    home = _norm_home_ownership(get("home_ownership"))
    verif = _norm_verification_status(get("verification_status"))
    app = _norm_application_type(get("application_type"))
    ils = _norm_initial_list_status(get("initial_list_status"))
    st = _norm_addr_state(get("addr_state"))

    # Initialize all one-hot keys to 0 for deterministic output.
    for name in _v2_onehot_feature_names():
        out[name] = 0.0

    if purpose in _PURPOSE_VOCAB:
        out[f"purpose__{purpose}"] = 1.0
    if home in _HOME_OWNERSHIP_VOCAB:
        out[f"home_ownership__{home}"] = 1.0
    if verif in _VERIFICATION_STATUS_VOCAB:
        out[f"verification_status__{verif}"] = 1.0
    if app in _APPLICATION_TYPE_VOCAB:
        out[f"application_type__{app}"] = 1.0
    if ils in _INITIAL_LIST_STATUS_VOCAB:
        out[f"initial_list_status__{ils}"] = 1.0
    if st is None:
        out["addr_state__UNK"] = 1.0
    elif st == "UNK":
        out["addr_state__UNK"] = 1.0
    else:
        out[f"addr_state__{st}"] = 1.0

    # Ensure order & only v2 keys
    return {k: out.get(k, 0.0) for k in FEATURE_NAMES_NO_LEAKAGE_V2}


def build_features_from_dataframe_no_leakage_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Build V2 additive no-leakage features from a DataFrame."""
    raw_list = []
    for _, row in df.iterrows():
        raw_list.append(build_features_from_dict_no_leakage_v2(row.to_dict()))
    return pd.DataFrame(raw_list)


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
    # Phase 1.1 optional: is_missing flags for HF-missing fields (model learns different behavior)
    dti_ratio_val = feats.get("dti_ratio", np.nan)
    out["is_missing_dti"] = 1.0 if (dti_ratio_val is None or (isinstance(dti_ratio_val, float) and np.isnan(dti_ratio_val))) else 0.0
    out["is_missing_credit_history"] = 1.0 if (credit_history_months is None or (isinstance(credit_history_months, float) and np.isnan(credit_history_months))) else 0.0
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


def get_feature_names_no_leakage_v2() -> List[str]:
    """Return the ordered list of V2 additive origination-only (no-leakage) feature names."""
    return list(FEATURE_NAMES_NO_LEAKAGE_V2)


def sanitize_query_for_display(query: str) -> str:
    """
    Return query string with post-origination fields stripped from the Text block.
    Use when storing/displaying samples so proof files show a clean, no-leakage input.
    """
    if not query or not isinstance(query, str):
        return query
    text_match = re.search(r"(Text:\s*['\"])([^'\"]+)(['\"])", query, re.IGNORECASE | re.DOTALL)
    if not text_match:
        return query
    prefix, block, suffix = text_match.group(1), text_match.group(2), text_match.group(3)
    stripped_block = _strip_blocklist_from_query_text(block.strip()).strip()
    return query[: text_match.start()] + prefix + stripped_block + suffix + query[text_match.end() :]


def parse_query_to_features(
    query: str,
    fill_missing: Union[float, Dict[str, float]] = 0.0,
    use_no_leakage: bool = False,
    feature_version: str = "v1",
) -> Dict[str, float]:
    """
    Parse LendingClub benchmark query text and return a feature dict suitable for predict_pd.
    Replaces NaN with fill_missing so the model always receives numeric values.
    If use_no_leakage=True, returns origination-only features (same set as get_feature_names_no_leakage).
    fill_missing: float for uniform fill, or dict of {feature_name: value} for per-feature median imputation.
    """
    raw = _parse_lendingclub_query_text(query)
    if use_no_leakage:
        if str(feature_version).lower() in ("v2", "2", "no_leakage_v2"):
            feats = build_features_from_dict_no_leakage_v2(raw)
        else:
            feats = build_features_from_dict_no_leakage(raw)
    else:
        feats = build_features_from_dict(raw)
    for k in feats:
        v = feats[k]
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            if isinstance(fill_missing, dict):
                feats[k] = fill_missing.get(k, 0.0)
            else:
                feats[k] = fill_missing
    return feats
