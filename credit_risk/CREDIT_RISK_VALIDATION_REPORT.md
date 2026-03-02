# PD Model – Validation Report (Template)

This document provides a **validation report** for the PD (Probability of Default) model used in this repository. It aligns with MRM expectations: purpose, data, methodology, performance, stability (PSI), back-testing, and monitoring. Reproducible numbers are in **`notebooks/02_pd_xgboost_training.ipynb`** (Section 4: credit metrics; Section 6: model card).

---

## 1. Purpose and scope

- **Purpose:** Estimate 12‑month PD for US consumer loans (Lending Club LoanStats3a) for use in ECL (IFRS 9), capital (Basel), and portfolio risk.
- **Scope:** Origination-time features only; no post-origination or outcome-related leakage (e.g. no grade, sub_grade, int_rate, total_pymnt, recoveries).
- **Output:** PD scores (0–1) and optional binary classification via a tuned threshold (e.g. F1-optimal).

---

## 2. Data and methodology

- **Data source:** Lending Club **LoanStats3a**; engineered features from `01_lendingclub_feature_engineering.ipynb`, then `lendingclub_engineered.parquet`.
- **Feature set:** No-leakage set from `credit_risk.feature_engineering.common_features.get_feature_names_no_leakage()`.
- **Splits:** Out-of-time when available (train: 2007–2015, val: 2016, test: 2017–2018); otherwise random 70/15/15 train/val/test, stratified.
- **Methodology:** XGBoost (and optional LightGBM stacking), tuned with **Optuna** (stratified 5‑fold CV, early stopping) to maximize AUC-ROC; refit on train+val for final artifact. Class imbalance handled via `scale_pos_weight` and threshold tuning (no SMOTE).
- **Default definition:** Charged Off / Default / Late 31–120 days vs Fully Paid.

---

## 3. Performance (discrimination)

Metrics below are from the **test set** (out-of-time). Re-run Section 4 of the notebook to refresh.

| Metric   | Value (example) | Notes |
|----------|------------------|--------|
| AUC-ROC  | *e.g. 0.605*    | Primary tuning target. |
| Gini     | *e.g. 0.21*     | Gini = 2×AUC−1. |
| KS       | *e.g. 0.17*     | Kolmogorov–Smirnov statistic. |
| AUC-PR   | *e.g. 0.19*     | Precision–recall area (imbalanced). |

*Replace with actual values from the notebook after each run; the notebook prints these in Section 4.*

---

## 4. Stability (PSI)

- **Population Stability Index (PSI):** Train vs test score distribution.
  - **&lt; 0.10:** Stable; no action.
  - **0.10–0.25:** Monitor; possible distribution shift.
  - **&gt; 0.25:** Unstable; investigate and consider retrain.
- **Example (notebook):** PSI (train vs test) ≈ 0.016 → stable. Update from notebook output when re-running.

---

## 5. Back-testing (predicted vs realised)

Back-test is **realised default rate by score decile** (low to high PD). The model is validated by checking that bad rate generally increases with decile (directional consistency).

- **Method:** Score test set, form deciles by predicted PD, compute realised bad rate per decile.
- **Notebook:** Section 4 reports “Bad rate by decile (low to high PD)” and a monotonicity check. Use that plot and the printed list as the back-test evidence.
- **Example (illustrative):**  
  Decile 1 (lowest PD): ~8.2%; …; Decile 10 (highest PD): ~23.3%.  
  *Replace with the exact list from the notebook.*

---

## 6. Limitations and monitoring

- **Limitations:** US consumer loans only; performance may differ in other geographies or products. Default definition as above; other definitions would require re-labelling.
- **Monitoring:**
  - **PSI** (train vs test or reference vs current): &lt; 0.1 stable; 0.1–0.25 monitor; &gt; 0.25 investigate/retrain.
  - **KS:** Drop of &gt; 5 points vs baseline triggers review.
  - **AUC:** Monitor on recent cohorts; document in model card (Section 6).
- **Retraining:** Trigger on PSI &gt; 0.25, material KS drop, or policy-based schedule.

---

## 7. References

- **Model development:** `notebooks/02_pd_xgboost_training.ipynb` (Sections 1–6).
- **Model card:** Section 6 of the same notebook (dataset, leakage exclusions, limitations, monitoring, top drivers, MAS FEAT).
- **Regulatory context:** `CREDIT_RISK_REGULATORY_CONTEXT.md` (IFRS 9, ECL, Basel, stress, MRM).
- **PD model code:** `credit_risk/models/pd_model.py`.
