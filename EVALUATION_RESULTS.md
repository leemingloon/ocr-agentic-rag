# Evaluation Results

**Document Intelligence + Credit Risk Platform**

Evaluation is run via **`eval_runner.py`**. All proof outputs are written under **`data/proof/`**. The canonical summary is **`data/proof/eval_summary.json`**, which is updated when evaluations are run. Demos and notebooks in **`notebooks/`** read from `data/proof/` for reported metrics.

---

## How to run evaluation

```bash
# Single category (e.g. RAG, one dataset)
python eval_runner.py --category rag --dataset FinQA --max_split 200 --max_category 200

# Single TAT-QA sample (for quick checks)
python eval_runner.py --category rag --dataset TATQA --max_split 1 --max_category 1 --debug --export_predictions_txt

# Credit risk PD (LendingClub)
python eval_runner.py --category credit_risk_PD --dataset LendingClub

# Credit risk memo (FinanceBench)
python eval_runner.py --category credit_risk_memo_generator --dataset FinanceBench

# Vision
python eval_runner.py --category vision --dataset DocVQA
```

Outputs are written under `data/proof/<category>/<dataset>/` (per-split samples and averages). Category-level averages and the global **`eval_summary.json`** are updated from these runs.

---

## Where results live

| Category | Proof path | Main datasets |
|----------|------------|----------------|
| **Credit risk memo** | `data/proof/credit_risk_memo_generator/` | FinanceBench |
| **Credit risk PD** | `data/proof/credit_risk_pd/` | LendingClub (tuned / untuned XGB, LR, etc.) |
| **OCR** | `data/proof/ocr/` | SROIE, FUNSD |
| **RAG** | `data/proof/rag/` | FinQA, TAT-QA |
| **Vision** | `data/proof/vision/` | ChartQA, DocVQA, InfographicsVQA, MMMU_* |

Methodology (adapters, metrics, aggregation) is in **`eval_runner.py`**, **`eval_dataset_adapters.py`**, and **`eval_postprocess_utils.py`**.

---

## Current results summary

Numbers below are taken from **`data/proof/eval_summary.json`** (as of the last eval run). Run `eval_runner.py` and re-open `eval_summary.json` to refresh.

### Credit risk memo generator (FinanceBench)

| Metric | Value | Notes |
|--------|--------|------|
| Exact match (mean) | 0.94 | 100 samples (train) |
| F1 (mean) | ~0.949 | |
| Relaxed match (mean) | 0.94 | |

Model: RiskMemoGenerator (Claude Sonnet). Proof: `data/proof/credit_risk_memo_generator/financebench/`.

### Credit risk PD (LendingClub)

| Metric | Value | Notes |
|--------|--------|------|
| AUC-ROC (mean) | ~0.605 | Aggregated across runs/splits; ~40k sample count in summary |
| F1 (mean) | ~0.157 | |
| Precision (mean) | ~0.328 | |
| Recall (mean) | ~0.171 | |

Per-split and per-model (e.g. tuned vs untuned XGB, LR) breakdowns are in `data/proof/credit_risk_pd/` and inside `eval_summary.json` under `categories.credit_risk_pd.datasets_breakdown`. Model: PDModel (XGBoost backbone).

### RAG (FinQA, TAT-QA)

| Dataset | Split | Exact match | Relaxed exact match | F1 (where applicable) |
|---------|--------|-------------|---------------------|------------------------|
| FinQA | test | 0.715 (143/200) | 0.775 (155/200) | — |
| FinQA | train | 0.905 (181/200) | 0.920 (184/200) | — |
| TAT-QA | dev | 0.770 (154/200) | 0.895 (179/200) | 0.624 |
| TAT-QA | test | 0.810 (162/200) | 0.985 (197/200) | 0.674 |

Weighted over 400 samples per dataset. Model: AgenticRAG (LangGraph, hybrid retriever, BGE reranker). Out-of-sample design: see **`data/proof/rag/OUT_OF_SAMPLE_AUDIT.md`**.

### Vision (DocVQA, ChartQA, InfographicsVQA, MMMU)

| Dataset | Metric | Value |
|---------|--------|--------|
| ChartQA | relaxed_accuracy_mean / strict_accuracy_mean | 1.0 (11 samples) |
| DocVQA | anls_mean / exact_match_mean | ~0.889 (9 samples, validation) |
| InfographicsVQA | anls_mean / exact_match_mean | 1.0 (7 samples, validation) |
| MMMU (Accounting, Economics, Finance, Math) | accuracy_mean | ~0.943 (29 samples across subjects) |

Model: VisionOCR (Claude Sonnet 4.6). Proof: `data/proof/vision/<dataset>/`.

### OCR (SROIE, FUNSD)

Proof under `data/proof/ocr/` (FUNSD, SROIE). Entity and word-level metrics are in `eval_summary.json`; current summary may show placeholder zeros until metrics are fully wired. Sample counts: FUNSD 150, SROIE 200.

---

## Categories and datasets (registry)

The list of categories and datasets is defined in **`eval_runner.py`** in **`AUTO_DATASETS`** and **`ADAPTER_REGISTRY`**, e.g.:

- **ocr:** SROIE, FUNSD  
- **vision:** DocVQA, ChartQA, InfographicsVQA, MMMU_Accounting, MMMU_Economics, MMMU_Finance, MMMU_Math  
- **rag:** FinQA, TATQA  
- **credit_risk_PD:** LendingClub  
- **credit_risk_sentiment / credit_risk_sentiment_finbert:** FinancialPhraseBank, FiQA  
- **credit_risk_memo_generator:** FinanceBench  

Adding a new dataset or category requires an adapter and, if needed, postprocessing in `eval_postprocess_utils.py`.

---

**Last updated:** March 2026 (from `data/proof/eval_summary.json` and repo state).
