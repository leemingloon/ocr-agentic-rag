### Quick look at my work and results (for developers)

OCR → RAG → Credit Risk platform for financial documents<br>

**Status:** Deployed to Amazon Web Services (AWS) Sagemaker. Evaluated using Local / Google Colabs / Kaggle.

- **Credit risk:** , notebooks in [notebooks/](notebooks/) [Completed]
  - [demo_credit_risk_pd.ipynb](notebooks/demo_credit_risk_pd.ipynb)
  - [00_pd_homecredit_lstm_kaggle.ipynb](notebooks/00_pd_homecredit_lstm_kaggle.ipynb)
  - [01_pd_lendingclub_feature_engineering.ipynb](notebooks/01_pd_lendingclub_feature_engineering.ipynb)
  - [02a_pd_xgboost_training.ipynb](notebooks/02a_pd_xgboost_training.ipynb)
  - [02b_pd_ann_training.ipynb](notebooks/02b_pd_ann_training.ipynb)
  - [02c_pd_quantum_qvc_training.ipynb](notebooks/02c_pd_quantum_qvc_training.ipynb)
  - [02z_pd_model_comparison.ipynb](notebooks/02z_pd_model_comparison.ipynb)
  - [03_sentiment_FP_FiQA_feature_engineering.ipynb](notebooks/03_sentiment_FP_FiQA_feature_engineering.ipynb)
  - [04a_sentiment_finbert_training.ipynb](notebooks/04a_sentiment_finbert_training.ipynb)
  - [04b_sentiment_qnlp_training.ipynb](notebooks/04b_sentiment_qnlp_training.ipynb)
  - [04z_sentiment_model_comparison.ipynb](notebooks/04z_sentiment_model_comparison.ipynb)
  - PD predictions in [data/proof/credit_risk_pd/](data/proof/credit_risk_pd/) (e.g. [LendingClub](data/proof/credit_risk_pd/lendingclub/test/lendingclub_test_predictions.txt))
  - Credit risk memo generator on [FinanceBench](data/proof/credit_risk_memo_generator/financebench/train/financebench_train_predictions.txt).
- **RAG (model predictions):** Under [data/proof/rag/](data/proof/rag/). [Completed]
  - **FinQA:** [in-sample (train)](data/proof/rag/finqa/train/finqa_train_predictions.txt), [out-of-sample (test)](data/proof/rag/finqa/test/finqa_test_predictions.txt)
  - **TAT-QA:** [in-sample (test)](data/proof/rag/tatqa/test/tatqa_test_predictions.txt), [out-of-sample (dev)](data/proof/rag/tatqa/dev/tatqa_dev_predictions.txt)
- **Vision (model predictions):** [`*_predictions.txt`](data/proof/vision/docvqa/validation/docvqa_validation_predictions.txt) files under [data/proof/vision/](data/proof/vision/).
- **OCR (model predictions):** [`*_avg.json`](data/proof/ocr/funsd/funsd_avg.json) files under [data/proof/ocr/](data/proof/ocr/).

---

## ⚡ Overview & Key Results

End-to-end platform for **financial document intelligence and credit risk**, combining:

- **OCR:** Extract text and layout from PDFs, scans, and forms.
- **RAG:** Answer finance questions over reports, tables, and notes using LangGraph + hybrid retrieval.
- **Multimodal vision:** Use Claude Sonnet (vision) for charts, complex layouts, and visual QA.
- **Credit risk:** Build PD models, sentiment signals, and LLM-based risk memos from structured features.

### Key Results (benchmarks)

<small>

| Layer | Dataset | Metric | Value | Total population | OOT test sample size | Notes |
|-------|---------|--------|-------|-----------------|----------------------|------|
| **Credit risk PD (LSTM)** | Home Credit (full population) | OOT AUC-ROC | 0.756 | 307,511 | 61,502 | |
| **Credit risk PD (LSTM)** | Home Credit (has_repayment_bureau 88K) | OOT AUC-ROC | 0.749 | 88,816 | 17,763 | |
| **Credit risk PD (LSTM)** | Home Credit (has_bureau) | OOT AUC-ROC | 0.753 | 295,058 | 58,829 | |
| **Credit risk PD (LSTM)** | Home Credit (no_bureau) | OOT AUC-ROC | 0.744 | 12,453 | 109 | |
| **Credit risk PD (Logistic Regression)** | LendingClub | OOT AUC-ROC, Brier | 0.660, 0.236 | — | 21,721 | Preferred for rank-ordering |
| **Credit risk PD (Optuna-tuned XGBoost/LightGBM stack)** | LendingClub | OOT AUC-ROC, Brier | 0.636, 0.145 | — | 21,721 | Preferred for PD calibration/estimation |
| **Credit risk PD (ANN)** | LendingClub | OOT AUC-ROC | 0.616 | — | 21,721 | |
| **RAG** | FinQA (out-of-sample) | Relaxed / Exact | 77.5% / 71.5% | — | 200 | |
| **RAG** | FinQA (in-sample) | Relaxed / Exact | 92% / 90.5% | — | 200 | |
| **RAG** | TAT-QA (out-of-sample) | Relaxed / Exact | 89.5% / 77% | — | 200 | |
| **RAG** | TAT-QA (in-sample) | Relaxed / Exact | 98.5% / 81% | — | 200 | |
| **Vision-Language model (VLM)** | DocVQA, ChartQA, InfographicsVQA, MMMU | In-sample accuracy | ~90% | — | 56 | |
| **Credit risk memo generator** | FinanceBench | Exact match | ~94% | — | 100 | |
| **Credit risk PD (Quantum VQC)** | LendingClub | OOT AUC-ROC | 0.540 | — | 21,721 | |
| **Credit Risk Sentiment (QNLP)** | Financial PhraseBank & FiQA | Test F1 macro | ~0.40 | — | 112 | |

</small>

All evaluation metrics are produced via `eval_runner.py` and stored under `data/proof/`.  
See [ARCHITECTURE.md](ARCHITECTURE.md) and [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for full detail.

---

## 🎯 Use Cases & Benchmarks

### OCR Layer
- Invoice and receipt extraction (SROIE)
- Form and contract key-value extraction (FUNSD)
- General document text extraction feeding RAG and credit risk

### Multimodal Layer
- Chart understanding and numeric QA (ChartQA)
- Document visual QA (DocVQA, InfographicsVQA)
- Multichoice reasoning over financial diagrams and tables (MMMU Accounting/Economics/Finance/Math)

### RAG Layer
- Financial QA over reports and tables (FinQA)
- Table- and passage-heavy question answering (TAT-QA)
- Retrieval-augmented support for memo generation and analysis

### Credit Risk Layer
- Probability of default (PD) modelling (LendingClub; Home Credit in notebooks)
- Early warning and deterioration signals (ratios, trends, NLP sentiment)
- LLM-generated credit risk memos (FinanceBench-style Q&A)

---

## 📊 System Architecture & Technology Stack
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: OCR                                                                │
│ Detection: Cache → OpenCV (classical) → PaddleOCR (DL) | ONNX optional      │
│ Recognition: Tesseract → PaddleOCR → Claude Sonnet (vision fallback)        │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ Structured text + layout
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: RAG                                                                │
│ Chunking: structure-preserving | Embeddings: BGE-M3 | Retrieval: FAISS+BM25 │
│ Reranking: BGE-reranker-v2-m3 | Orchestration: LangGraph | LLM: Claude      │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ Enriched context
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: MULTIMODAL VISION                                                  │
│ Model: Claude 3.5 Sonnet Vision | DocVQA, ChartQA, InfographicsVQA, MMMU    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ Structured features
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: CREDIT RISK                                                        │
│ Features: Pandas/NumPy (ratios, trends) | NLP: FinBERT + rules              │
│ PD: XGBoost, LightGBM, LR, PyTorch (ANN/LSTM) | Explainability: SHAP        │
│ Memos: Claude Sonnet | Governance: prompt registry + safety filter          │
│ Monitoring: SciPy (KS), drift | Cloud: AWS SageMaker, S3                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Design Decisions

**Why this matters:** These choices shape the system’s reliability, cost, and regulatory fit.

| Decision | Rationale |
|----------|-----------|
| **Hybrid retrieval (BM25 + BGE-M3)** | Handles both numeric tables and text-heavy financial documents; sparse + dense covers different query patterns. |
| **Agentic orchestration (LangGraph)** | Dynamically selects tools instead of static pipelines; adapts to multi-step QA and reasoning. |
| **3-tier OCR fallback** | Balances cost, speed, and accuracy: cache → classical → DL -> Vision-Language. Cache for detected known document templates, Classical for text recognition, Deep Learning for table recognition, Vision (multi-modal) fallback for charts or complex diagrams understanding. |
| **LLM + structured features fusion for credit risk** | Not pure black-box models; interpretable PD models (LR, XGBoost, LSTM) plus LLM for memos and explanations. |
| **Governance layer (prompt registry + safety filters)** | Production readiness: to add versioning, audit trail, and safety filters for LLM outputs. |

---

## 🔒 Governance, Risk & Compliance (Work in progress)

### Fairness
- Target: bias gap &lt;10% across document types and benchmarks.
- Bias tests and layout cache/completeness heuristics are summarised in `data/proof/SUMMARY.md` (current gap and coverage).

### Ethics
- High-risk credit decisions must go to **human-in-the-loop**; low-risk segments may be auto-approved under configured thresholds.
- LLM usage (risk memos) is constrained by safety filters and prompt registry policies.

### Accountability
- Full audit trail from data → features → model → decision, including prompt/version for LLM calls.
- Prompt registry in `credit_risk/governance/` tracks versions, approvers, and status for all production prompts.

### Transparency
- PD models use SHAP for feature-attribution explanations where applicable.
- RAG answers and risk memos can surface retrieved evidence and key drivers; drift monitoring is logged via `credit_risk/monitoring/`.

---

## 📖 Documentation

### Core Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
- [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) - All 20 benchmark results
- [data/credit_risk/README.md](data/credit_risk/README.md) - Dataset download guide

---

## 📁 Repository Structure
```
ocr-agentic-rag/
├── ocr_pipeline/                 # Layer 1: OCR (3-tier detection + recognition)
│   ├── quality_assessment.py
│   ├── template_detector.py
│   ├── detection/
│   └── recognition/
├── rag_system/                   # Layer 2: RAG + multimodal fusion
│   ├── chunking.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── multimodal_rag.py
│   └── agentic/
│       ├── orchestrator.py
│       ├── tools.py
│       └── memory.py
├── credit_risk/                  # Layer 4: Credit Risk
│   ├── pipeline.py               # End-to-end PD + memo pipeline
│   ├── feature_engineering/
│   │   ├── ratio_builder.py
│   │   ├── trend_engine.py
│   │   └── nlp_signals.py
│   ├── models/
│   │   ├── pd_model.py
│   │   ├── pd_ann.py
│   │   ├── quantum_pd_model.py
│   │   ├── lgd_model.py
│   │   ├── ead.py
│   │   ├── ecl.py
│   │   └── counterfactual.py
│   ├── sentiment/                # FinBERT + rule-based sentiment
│   │   ├── pipeline.py
│   │   ├── config.py
│   │   ├── negation.py
│   │   ├── hedging.py
│   │   └── evaluation.py
│   ├── governance/
│   │   ├── risk_memo_generator.py
│   │   ├── prompt_registry.py
│   │   ├── prompt_version.py
│   │   └── safety_filter.py
│   └── monitoring/
│       ├── data_drift.py
│       └── prediction_drift.py
├── data/
│   ├── proof/                        # All evaluation outputs (see EVALUATION_RESULTS.md)
│   │   ├── rag/
│   │   ├── vision/
│   │   ├── ocr/
│   │   ├── credit_risk_pd/
│   │   └── credit_risk_memo_generator/
│   ├── credit_risk_pd/               # LendingClub benchmark (PD eval)
│   ├── credit_risk_sentiment/        # FinancialPhraseBank, FiQA
│   ├── credit_risk_memo_generator/   # FinanceBench
│   ├── home_credit/                  # Home Credit CSVs (notebooks)
│   └── rag/ / vision/ / ocr/         # Raw benchmark datasets (after download_datasets.py)
├── scripts/
│   ├── download_datasets.py          # Download all HF/other datasets into data/
│   └── build_*_embeddings_*.py       # Index-building helpers for RAG
├── notebooks/                        # 00–04z credit risk, RAG index build, demos
├── eval_runner.py                    # Unified evaluation entry point
├── eval_dataset_adapters.py
├── eval_postprocess_utils.py
├── ARCHITECTURE.md
├── EVALUATION_RESULTS.md
├── requirements.txt
└── README.md
```

---

## 💻 Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM minimum
- Tesseract OCR installed
- Anthropic API key (for Vision and RAG)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/leemingloon/ocr-agentic-rag.git
cd ocr-agentic-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Local PC

**Specs (suggested):** 16GB RAM, CPU-only is fine  
**Mode:** `local`  
**Typical usage:** quick demos and small eval slices  
**Runtime:** minutes on small batches  
**Cost:** $0

**Quick demos (run after installation):**
```bash
# 1) Small RAG demo (1 TATQA sample)
python eval_runner.py --category rag --dataset TATQA --max_split 1 --max_category 1 --debug --export_predictions_txt

# 2) Small credit risk memo demo (FinanceBench)
python eval_runner.py --category credit_risk_memo_generator --dataset FinanceBench --max_split 20 --max_category 20

# 3) Vision demo (DocVQA)
python eval_runner.py --category vision --dataset DocVQA --max_split 10 --max_category 10
```

---

### AWS SageMaker (Free Tier)

**Instance (example):** `ml.t3.medium` (2 vCPU, 4GB RAM)  
**Mode:** `sagemaker`  
**Typical usage:** batch evaluation on more samples with managed storage/compute  
**Cost:** $0 within free-tier hours (check your AWS account)

**Setup:**
```bash
# 1. Create S3 bucket
aws s3 mb s3://my-sagemaker-credit-risk

# 2. Download evaluation datasets into data/
python scripts/download_datasets.py

# 3. Upload data/ if you prefer remote storage
aws s3 sync data/ s3://my-sagemaker-credit-risk/data/

# 4. In SageMaker notebook, run evals as usual
python eval_runner.py --category rag --dataset FinQA --max_split 200 --max_category 200
```

**SageMaker Tips:**
- Use **`ml.t3.medium`** or similar free-tier instance
- Keep `max_split` / `max_category` modest for experiments
- Store `data/proof/` outputs on S3 if you need persistence

---

### Production (Full Datasets)

**Specs (example):** 16GB+ RAM; GPU recommended for heavy vision or OCR DL  
**Mode:** `production` (or multiple targeted `eval_runner.py` calls)  
**Typical usage:** full benchmark sweeps and renewal of `data/proof/eval_summary.json`  
**Cost:** depends on hardware/runtime (local vs cloud)
```bash
# 1. Download all required datasets
python scripts/download_datasets.py

# 2. Run category-level sweeps (examples)
python eval_runner.py --category ocr
python eval_runner.py --category vision
python eval_runner.py --category rag
python eval_runner.py --category credit_risk_PD
```

---

## 📝 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for full terms.

---

## 🙏 Acknowledgments

**Datasets and benchmarks:**
- Home Credit Default Risk (Kaggle)
- LendingClub (Kaggle, TheFinAI/lendingclub-benchmark)
- Financial PhraseBank, FiQA, FinanceBench (Hugging Face)
- SROIE, FUNSD, DocVQA, ChartQA, InfographicsVQA, MMMU

**Libraries and frameworks:**
- PaddleOCR, OpenCV, Tesseract
- LangGraph, FAISS, Hugging Face Transformers
- Anthropic Claude (Sonnet language-vision + text)
- XGBoost, LightGBM, scikit-learn, SHAP

---

**Contact:** Lee Ming Loon | Singapore
