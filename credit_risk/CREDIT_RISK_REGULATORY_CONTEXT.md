# Credit Risk Regulatory Context

This document describes how the models in this repository align with common regulatory and risk frameworks (IFRS 9, ECL, Basel, stress testing, MRM). It is for portfolio and interview clarity, not legal or compliance advice.

---

## 1. IFRS 9 and Expected Credit Loss (ECL)

**IFRS 9** requires institutions to recognise **Expected Credit Loss (ECL)** using forward-looking information, instead of only incurred losses.

- **ECL** for a loan or portfolio is typically: **ECL = PD × LGD × EAD** (and, for stage 2/3, lifetime measures).
- **PD (Probability of Default):** Estimated probability of default over the relevant horizon (e.g. 12 months). This repo’s main output is a **PD model** (see `credit_risk/models/pd_model.py` and notebook `02_pd_xgboost_training.ipynb`).
- **LGD (Loss Given Default):** Fraction of exposure lost once default occurs (1 − recovery rate). A placeholder LGD component is in `credit_risk/models/lgd_model.py`; full LGD requires recovery/default data.
- **EAD (Exposure at Default):** Credit exposure at default (e.g. outstanding balance). Often taken from systems or a simple formula; see `credit_risk/models/ead.py` for a conceptual placeholder.

**How this repo fits:** The **PD model** is designed to be used as the **PD input** to an ECL or pricing process. LGD and EAD are stubbed so the **credit risk suite** (PD + LGD + EAD → ECL) is present in code and docs; production ECL would plug in institution-specific LGD/EAD and staging logic.

---

## 2. Basel (IRB, capital)

Under **Basel** (e.g. IRB approach), **risk-weighted assets (RWA)** for credit risk depend on **PD**, **LGD**, and **EAD** (and maturity).

- **PD** is a key input to the regulatory capital formula. This repo’s PD model is of the type used for such inputs (point-in-time style, 12‑month horizon, no leakage).
- **LGD** and **EAD** are required for RWA; here they are placeholders for structure only.

**Stress testing** often applies stressed PD (and sometimes LGD) to compute capital under adverse scenarios. The PD model can be **stressed** by shifting inputs (e.g. higher default rate assumption) or by re‑estimating on stressed data; a simple sensitivity (e.g. “what if PD doubles”) can be run without full macro stress.

---

## 3. Stress testing

- **Use of PD:** Stressed ECL or capital can use **stressed PD** (e.g. from a satellite model or scenario) while keeping the same model structure.
- **Sensitivity:** A minimal stress is to scale or shift PD (e.g. multiply by 1.5 for a severe scenario) and recompute ECL. The notebook’s PSI and out-of-time test already support **distribution shift** awareness (train vs test).
- **Model risk:** Documenting limitations and monitoring (PSI, AUC, back-test) supports stress and MRM discussions.

---

## 4. Model Risk Management (MRM)

**MRM** typically requires:

- **Development documentation:** Purpose, data, methodology, variables, and assumptions. See **CREDIT_RISK_VALIDATION_REPORT.md** and the notebook’s Model card (Section 6).
- **Validation:** Discrimination (e.g. AUC, Gini, KS), calibration, and **stability (e.g. PSI)**. The notebook reports AUC, Gini, KS, PSI (train vs test), and bad rate by decile; the validation report template centralises these.
- **Back-testing:** Comparison of predicted default rates vs realised default rates (e.g. by score decile or cohort). The notebook’s “bad rate by decile” is a **back-test** of predicted vs realised by score band.
- **Ongoing monitoring:** PSI thresholds (e.g. &lt; 0.1 stable, 0.1–0.25 monitor, &gt; 0.25 investigate) and performance (AUC/KS) on recent cohorts are described in the model card and validation report.

**Where this repo stands:** The PD model is documented and validated (discrimination, stability, back-test by decile). Full MRM would add formal sign-off, independent validation, and governance per institution policy.

---

## 5. Summary table

| Framework   | Component | This repo |
|------------|-----------|-----------|
| IFRS 9 ECL | PD        | ✅ PD model (XGBoost + LightGBM, no-leakage features) |
| IFRS 9 ECL | LGD       | Placeholder / stub (`lgd_model.py`) |
| IFRS 9 ECL | EAD       | Placeholder / stub (`ead.py`) |
| Basel IRB  | PD/LGD/EAD| PD implemented; LGD/EAD stubbed for structure |
| Stress     | Stressed PD | Design supports sensitivity; no full macro stress in repo |
| MRM        | Validation, back-test, stability | ✅ Validation report template; PSI, KS, decile back-test in notebook |

---

## References

- IFRS 9 (ECL): forward-looking expected credit loss.
- Basel III/IV: IRB formulas using PD, LGD, EAD.
- SR 11-7 / model risk: development, validation, monitoring.
