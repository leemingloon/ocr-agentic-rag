"""
UCI Credit Card Default – feature names and feature building for PD (Probability of Default).

Dataset: UCI ML Repository "Default of Credit Card Clients" (Taiwan).
Target: default payment next month (1 = default, 0 = non-default).
Features: application/demographic and repayment history (PAY_0–PAY_6); no post-default
billing/payment amounts in the no-leakage set so the pipeline is suitable for training
and evaluation (e.g. 04_uci_feature_engineering.ipynb → XGBoost).
"""

from typing import List

# No-leakage feature set: limit balance, demographics, repayment status at application.
# PAY_* = repayment status (-1 to 8); BILL_AMT* / PAY_AMT* can be included for
# "at-application" snapshot if defined as of origination; here we use the standard
# 23 input columns from UCI (excluding ID and target).
UCI_FEATURE_NAMES: List[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def get_uci_feature_names() -> List[str]:
    """Return the ordered list of UCI feature names for PD training/eval."""
    return list(UCI_FEATURE_NAMES)
