OCR → RAG → Credit Risk Platform

**Document Intelligence + Credit Risk System**<br>
**Author:** Lee Ming Loon<br>
**Status:** Deployed to Amazon Web Services (AWS) Sagemaker. Evaluated using Local / Google Colabs / Kaggle.

**Credit risk (feature eng, train model):** 00-04z Jupyter notebooks (`.ipynb`) under `notebooks/` folder. [Completed]<br>
**RAG (model predictions):** dataset_split_samples.json files under `data/proof/rag/` folder. [Completed]<br>
**Vision (model predictions):** dataset_split_samples.json files under `data/proof/vision/` folder.<br>
**OCR (model predictions):** dataset_split_samples.json files under `data/proof/ocr/` folder.

---

## 🎯 Project Overview

End-to-end platform for **financial document intelligence and credit risk**, combining:

- **OCR:** Extract text and layout from PDFs, scans, and forms.
- **RAG:** Answer finance questions over reports, tables, and notes using LangGraph + hybrid retrieval.
- **Multimodal vision:** Use Claude Sonnet (vision) for charts, complex layouts, and visual QA.
- **Credit risk:** Build PD models, sentiment signals, and LLM-based risk memos from structured features.

All evaluation metrics and benchmarks are produced via `eval_runner.py` and stored under `data/proof/`.  
See `ARCHITECTURE.md` and `EVALUATION_RESULTS.md` for more detail.

---

## 📊 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: OCR PIPELINE                    │
│   3-Tier Detection: Cache → Classical → DL                  │
│   Recognition: Tesseract → PaddleOCR → Vision OCR           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Structured Text + Layout
                         │
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: RAG                               │
│   Retrieval: BM25 + BGE-M3 (hybrid dense+sparse)            │
│   Reranking: BGE-reranker-v2-m3 (cross-encoder)             │
│   Orchestration: LangGraph (autonomous tool selection)      │
│   Performance:                                              │
│        FinQA, out-of-sample:                                │
│               relaxed exact match 77.5%                     │
│               exact match 71.5%                             │
│        TATQA: out-of-sample:                                │
│               relaxed exact match 89.5%                     │
│               exact match: 77%                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Enriched Context
                         │
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: MULTIMODAL VISION                     │
│   Vision Model: Claude 3.5 Sonnet Vision                    │
│   Use Cases: Charts, handwriting, complex layouts           │
│   Performance:                                              │
│         DocVQA, ChartQA, InfographicsVQA, MMMU              |
|                In-sample: 90% accuracy, n=50                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Structured Features
                         │
┌───────────────────────────────────────────────────────────────────┐
│            LAYER 4: CREDIT RISK PIPELINE                          │
│   Feature Engineering: Ratios, Trends, NLP sentiment              │
│   PD Model: Logistic Regression — OOT AUC 0.66                    │
│   Risk Memos: LLM-generated — FinanceBench exact match 94% n=100  │
│   Governance: Prompt versioning, safety filters                   │
│   Monitoring: Drift detection — Logistic Regression 0.234         │
└───────────────────────────────────────────────────────────────────┘
```

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

## 🎯 Use Cases

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

## 💻 Running on Different Platforms

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

## 🔧 Technology Stack

### OCR
- **Detection:** OpenCV (classical), template cache, PaddleOCR (DL)
- **Recognition:** Tesseract, PaddleOCR; Claude Sonnet (vision) as fallback
- **Optimization:** Optional ONNX Runtime export for PaddleOCR

### RAG
- **Chunking:** Structure-preserving, metadata-enriched
- **Embeddings:** BGE-M3 (Hugging Face)
- **Retrieval:** FAISS + BM25 (hybrid sparse + dense)
- **Reranking:** BGE-reranker-v2-m3 (cross-encoder)
- **Orchestration:** LangGraph state graph
- **LLM:** Claude Sonnet (text, tool use, CoT)

### Multimodal
- **Vision model:** Claude Sonnet (vision)
- **Use cases:** Chart QA, document QA, complex layouts, OCR validation

### Credit Risk
- **Feature engineering:** Pandas, NumPy (ratios, trends, NLP signals)
- **Sentiment/NLP:** FinBERT (ProsusAI/finbert) + rule-based postprocessing
- **PD / risk models:** XGBoost, LightGBM, scikit-learn, PyTorch (ANN/LSTM)
- **Explainability:** SHAP
- **LLM:** Claude Sonnet (risk memos and explanations)
- **Monitoring:** SciPy (KS), custom drift utilities
- **Governance:** Prompt registry + safety filter (SQLite-backed)

### Infrastructure
- **Monitoring/observability:** (optional) OpenTelemetry, Prometheus, CloudWatch
- **Cloud:** AWS SageMaker, S3 (batch and notebook workflows)
- **Storage:** Local files, S3; optional PostgreSQL / SQLite for metadata

---

## 📖 Documentation

### Core Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
- [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) - All 20 benchmark results
- [data/credit_risk/README.md](data/credit_risk/README.md) - Dataset download guide

---

## 🔒 MAS FEAT Compliance

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
