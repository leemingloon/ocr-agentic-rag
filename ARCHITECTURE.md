# System Architecture

**Document Intelligence + Credit Risk Platform**  
**Pipeline: OCR → Agentic RAG → Multimodal Vision → Credit Risk**

This document describes the current system design. All evaluation metrics are produced by `eval_runner.py` and written under **`data/proof/`**. See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for reported numbers.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [OCR Pipeline](#ocr-pipeline)
3. [Agentic RAG System](#agentic-rag-system)
4. [Multimodal Vision](#multimodal-vision)
5. [Credit Risk Pipeline](#credit-risk-pipeline)
6. [Technology Stack](#technology-stack)
7. [Key Design Decisions](#key-design-decisions)
8. [References](#references)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Financial Documents                    │
│              (Reports, Filings, News, Invoices)                 │
└────────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: OCR PIPELINE                         │
│   3-tier detection: Cache → Classical → PaddleOCR               │
│   Recognition: Tesseract → PaddleOCR → Vision (Claude) fallback  │
└────────────────────────────┬────────────────────────────────────┘
                              │ Structured text + layout
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 2: AGENTIC RAG                            │
│   Chunking → BM25 + BGE-M3 → BGE-reranker → LangGraph → LLM     │
│   Benchmarks: FinQA, TAT-QA (see data/proof/rag/)               │
└────────────────────────────┬────────────────────────────────────┘
                              │ Enriched context
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 3: MULTIMODAL VISION                           │
│   Chart/document QA: Claude Sonnet 4 (vision)                   │
│   Benchmarks: DocVQA, ChartQA, InfographicsVQA, MMMU             │
└────────────────────────────┬────────────────────────────────────┘
                              │ Structured features
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            LAYER 4: CREDIT RISK PIPELINE                         │
│   Feature engineering → PD / Sentiment / Risk memo → Governance │
│   Benchmarks: LendingClub (PD), FinanceBench (memo), FiQA/PB     │
└─────────────────────────────────────────────────────────────────┘
```

---

## OCR Pipeline

- **Detection:** 3-tier routing — template cache → classical (morphology, contours) → PaddleOCR for unknown or low-quality pages.
- **Recognition:** Tesseract first; low-confidence or complex content can fall back to PaddleOCR or vision (Claude).
- **Evaluation:** FUNSD, SROIE; outputs under `data/proof/ocr/`.
- **Code:** `ocr_pipeline/` (detection, recognition, quality assessment).

---

## Agentic RAG System

- **Retrieval:** Hybrid sparse + dense: **BM25** (keyword) and **BGE-M3** (dense), then **BGE-reranker-v2-m3** (cross-encoder).
- **Chunking:** Structure-preserving, metadata-enriched; indices built per dataset (e.g. FinQA train/test, TAT-QA).
- **Orchestration:** **LangGraph** for multi-step planning, tool selection (calculator, retrieval, etc.), and synthesis.
- **LLM:** Claude Sonnet 4 for answer generation; prompts can be CFA L2–oriented for financial reasoning.
- **Benchmarks:** FinQA (train/test), TAT-QA (dev/test). Out-of-sample usage and index/QA separation are documented in **`data/proof/rag/OUT_OF_SAMPLE_AUDIT.md`**.
- **Code:** `rag_system/` (chunking, retrieval, reranking, agentic orchestrator).

---

## Multimodal Vision

- **Role:** Chart extraction, document QA, and complex layout when OCR alone is insufficient.
- **Model:** Claude Sonnet 4 (vision) with adaptive primers where needed.
- **Benchmarks:** DocVQA, ChartQA, InfographicsVQA, MMMU (Accounting, Economics, Finance, Math). Results under `data/proof/vision/`.
- **Code:** Integrated in vision/OCR and RAG paths (e.g. `rag_system/multimodal_rag.py`, vision eval in `eval_runner.py`).

---

## Credit Risk Pipeline

Implemented in **`credit_risk/`**: feature engineering, PD/sentiment/memo models, governance, and monitoring.

### Feature engineering

- **Ratio builder** (`feature_engineering/ratio_builder.py`): Debt/EBITDA, current ratio, interest coverage, quick ratio, debt/equity from financials.
- **Trend engine** (`feature_engineering/trend_engine.py`): Deterioration signals (e.g. rising debt/EBITDA, falling coverage).
- **NLP signals** (`feature_engineering/nlp_signals.py`): Sentiment and entity signals from news/filings; uses FinBERT in sentiment pipeline.

### Models

- **PD (Probability of Default):**
  - **LendingClub:** Origination-only features; train-time feature selection (missingness, KS, correlation); OOT split (e.g. train 2007–2009, test 2011). Models: Logistic Regression, XGBoost/LightGBM (Optuna-tuned), ANN. Brier score, isotonic calibration. Evaluated via TheFinAI/lendingclub-benchmark; proof under `data/proof/credit_risk_pd/`.
  - **Home Credit:** Triple-stream LSTM (installments, bureau, credit card); bureau vs no-bureau segments; OOT AUC, decile capture, PSI, origination-feature importance. Training and eval in notebooks (e.g. `notebooks/00_pd_homecredit_lstm_kaggle*.ipynb`, `02a_pd_xgboost_training.ipynb`, `02b_pd_ann_training.ipynb`). Data in `data/home_credit/`.
- **Sentiment:** FinBERT fine-tuned on Financial PhraseBank + FiQA; TF-IDF + LogReg baseline; post-inference fixes (negation, hedged language). Benchmarks: Financial PhraseBank, FiQA.
- **Risk memo:** LLM-generated memos from structured features; evaluated on **FinanceBench** (proof under `data/proof/credit_risk_memo_generator/`).

### Governance and monitoring

- **Prompt registry** (`governance/prompt_version.py`, `prompt_registry.py`): Versioned prompts for MAS FEAT–style governance.
- **Safety filter** (`governance/safety_filter.py`): Policy and safety checks on LLM outputs.
- **Drift:** Data and prediction drift detectors in `monitoring/` (e.g. KS-based).

### Orchestration

- **`credit_risk/pipeline.py`:** End-to-end pipeline (ratio builder, trend engine, NLP signals, PD model, counterfactual, risk memo generator, drift). Modes: `local`, `sagemaker`, `production`; can use S3 for SageMaker.

---

## Technology Stack

| Layer      | Components |
|-----------|------------|
| **OCR**   | Tesseract, PaddleOCR, OpenCV; optional ONNX; vision fallback (Claude) |
| **RAG**   | BM25, BGE-M3, BGE-reranker-v2-m3, LangGraph, Claude Sonnet 4 |
| **Vision**| Claude Sonnet 4 (vision) |
| **Credit risk** | Pandas/NumPy, FinBERT, XGBoost/LightGBM/scikit-learn, PyTorch (ANN/LSTM), SHAP; prompt registry (e.g. SQLite) |

---

## Key Design Decisions

- **3-tier OCR:** Reduces cost and latency by using cache and classical detection before PaddleOCR; completeness heuristics reduce false negatives before escalation.
- **LangGraph for RAG:** Explicit state graph and tool selection for deterministic, debuggable agentic QA.
- **Hybrid retrieval + reranking:** BM25 + BGE-M3 + BGE-reranker improves recall and precision over dense-only retrieval.
- **Prompt registry:** Versioned, approved prompts and audit trail for risk memo and other LLM use (MAS FEAT–aligned).
- **PD models:** Tree-based (XGBoost/LightGBM) and ANN for LendingClub; LSTM for Home Credit; calibration (e.g. isotonic) for probability outputs.

---

## References

- **Evaluation:** `eval_runner.py`, `data/proof/`, [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)
- **RAG out-of-sample:** `data/proof/rag/OUT_OF_SAMPLE_AUDIT.md`
- **Data layout:** [data/credit_risk/README.md](data/credit_risk/README.md)
- **Regulatory:** MAS FEAT (Fairness, Ethics, Accountability, Transparency)
