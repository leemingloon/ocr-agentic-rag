#!/usr/bin/env python3
"""
Build LendingClub engineered dataset (no-leakage V2 features).

This is the script equivalent of `notebooks/01_pd_lendingclub_feature_engineering.ipynb`,
updated to generate the V2 feature set:
- strict no-leakage (excludes LC risk outputs like grade/sub_grade/int_rate)
- adds one-hot encodings for key categoricals (Lending Club PD best practice)

Output:
  data/credit_risk_pd/LendingClub/processed/lendingclub_engineered.parquet
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import os
import sys
import numpy as np
import pandas as pd


def _parse_issue_d(x) -> datetime | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%b-%Y")
    except Exception:
        return None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    data_dir = repo_root / "data" / "credit_risk_pd" / "LendingClub"
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "LoanStats3a.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Place LoanStats3a.csv under {data_dir}")

    # Match notebook behavior: some LoanStats exports have an extra header row.
    df_raw = pd.read_csv(csv_path, skiprows=1, low_memory=False)
    # Remove footer rows if present (e.g. "Total amount funded ...")
    df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains("Total", na=False)]

    if "loan_status" not in df_raw.columns:
        status_cols = [c for c in df_raw.columns if "status" in c.lower()]
        if not status_cols:
            raise ValueError("No loan_status-like column found")
        df_raw = df_raw.rename(columns={status_cols[0]: "loan_status"})

    df_raw["loan_status"] = df_raw["loan_status"].astype(str).str.strip()

    default_statuses = {
        "Charged Off",
        "Default",
        "charged off",
        "default",
        "Late (31-120 days)",
        "late (31-120 days)",
    }
    paid_statuses = {"Fully Paid", "fully paid"}

    mask_default = df_raw["loan_status"].isin(default_statuses)
    mask_paid = df_raw["loan_status"].isin(paid_statuses)
    df = df_raw[mask_default | mask_paid].copy()
    df["default"] = mask_default[mask_default | mask_paid].astype(int)

    # Normalize FICO column names expected by common_features
    rename = {"fico_range_low": "fico_low", "fico_range_high": "fico_high"}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Coerce numeric columns (handle % and commas)
    numeric_cols = [
        "loan_amnt",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "revol_util",
        "revol_bal",
        "open_acc",
        "total_acc",
        "inq_last_6mths",
        "fico_low",
        "fico_high",
        "pub_rec",
        "pub_rec_bankruptcies",
        "mort_acc",
        "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )

    # Leakage exclusion (strict): keep origination fields only; drop LC risk outputs
    leakage_cols = [
        "funded_amnt",
        "funded_amnt_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "recoveries",
        "collection_recovery_fee",
        "out_prncp",
        "out_prncp_inv",
        "last_pymnt_amnt",
        "last_pymnt_d",
        "grade",
        "sub_grade",
        "int_rate",
    ]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Outlier caps (same spirit as notebook)
    caps = {
        "annual_inc": (None, 1e7),
        "revol_util": (0, 100),
        "dti": (0, 50),
        "open_acc": (0, 50),
        "inq_last_6mths": (0, 20),
    }
    for col, (lo, hi) in caps.items():
        if col in df.columns and hi is not None:
            df[col] = df[col].clip(lower=df[col].min() if lo is None else lo, upper=hi)

    # Out-of-time split if issue_d parses (data-driven by min/max year)
    if "issue_d" in df.columns:
        issue_dt = df["issue_d"].apply(_parse_issue_d)
        issue_year = pd.to_datetime(issue_dt).dt.year
        years = issue_year.dropna().astype(int).unique()
        years = sorted(years)
        if len(years) >= 3:
            val_year, test_year = years[-2], years[-1]
            df["split"] = "train"
            df.loc[issue_year == val_year, "split"] = "val"
            df.loc[issue_year == test_year, "split"] = "test"
            df.loc[issue_year.isna(), "split"] = "train"
            print(f"Out-of-time split: train through {val_year - 1} val {val_year} test {test_year}")
        elif len(years) == 2:
            train_year, hold_year = years[0], years[1]
            df["split"] = "train"
            df.loc[issue_year.isna(), "split"] = "train"
            hold_mask = issue_year == hold_year
            hold_idx = df.index[hold_mask].tolist()
            rng = np.random.default_rng(42)
            rng.shuffle(hold_idx)
            mid = len(hold_idx) // 2
            df.loc[hold_idx[:mid], "split"] = "val"
            df.loc[hold_idx[mid:], "split"] = "test"
            print(f"Out-of-time split: train {train_year} val/test {hold_year}")
        else:
            df["split"] = "train"
            df.loc[issue_year.isna(), "split"] = "train"
            print("Only one year in data; out-of-time split not possible.")
    else:
        df["split"] = "train"

    # Build V2 features
    from credit_risk.feature_engineering.common_features import (
        build_features_from_dataframe_no_leakage_v2,
        get_feature_names_no_leakage_v2,
    )

    feature_names = get_feature_names_no_leakage_v2()
    X_feat = build_features_from_dataframe_no_leakage_v2(df)

    # Fill NaN: one-hot + flags -> 0, else median
    for c in X_feat.columns:
        if X_feat[c].isna().any():
            if "__" in c or c.startswith("is_missing"):
                X_feat[c] = X_feat[c].fillna(0.0)
            else:
                X_feat[c] = X_feat[c].fillna(X_feat[c].median())

    X_feat["default"] = df["default"].values
    X_feat["split"] = df["split"].values

    out_path = out_dir / "lendingclub_engineered.parquet"
    X_feat.to_parquet(out_path, index=False)

    cols = list(X_feat.columns)
    print(f"Saved {out_path}")
    print(f"Rows: {len(X_feat):,}  Cols: {len(cols):,}  Features: {len(feature_names):,}")
    print("Has purpose one-hots:", any(c.startswith("purpose__") for c in cols))
    print("Has addr_state one-hots:", any(c.startswith("addr_state__") for c in cols))
    print("Split counts:", X_feat["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()

