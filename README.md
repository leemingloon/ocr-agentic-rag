OCR→Agentic RAG→Credit Risk Platform

**Complete Document Intelligence + Credit Risk System**  
**Author:** Lee Ming Loon  
**Status:** Deployed to AWS Sagemaker. Evaluating models on Local PC (in progress on evaluate_on_local branch). Finally, Production-ready prototype.

> **Evaluation data:** Demos and notebooks in the `notebooks/` folder load model evaluation results from the `data/proof/` folder. Any metrics or scores cited in this repo that are not backed by current artifacts under `data/proof/` are placeholders and will be filled in as evaluation progresses.

---

## 🎯 Project Overview

End-to-end pipeline: **OCR → Agentic RAG → Multimodal Vision → Credit Risk**

### Key Achievements

- E2E fidelity (image → answer) — *see `data/proof/`*
- STP (straight-through processing) — *TBD*
- Cost per document — *TBD*
- Benchmarks: OCR, Vision, RAG, Credit Risk — *proof under `data/proof/`*
- MAS FEAT: audit trails, prompt versioning — *TBD*

---

## 📊 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: OCR PIPELINE                     │
│   3-Tier Detection: Cache → Classical → DL                  │
│   Recognition: Tesseract → PaddleOCR → Vision OCR           │
│   Performance: TBD (see data/proof)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Structured Text + Layout
                         │
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: AGENTIC RAG                        │
│   Retrieval: BM25 + BGE-M3 (hybrid dense+sparse)            │
│   Reranking: BGE-reranker-v2-m3 (cross-encoder)             │
│   Orchestration: LangGraph (autonomous tool selection)       │
│   Performance: TBD (see data/proof)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Enriched Context
                         │
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: MULTIMODAL VISION                      │
│   Vision Model: Claude 3.5 Sonnet Vision                    │
│   Use Cases: Charts, handwriting, complex layouts            │
│   Performance: TBD (see data/proof)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ Structured Features
                         │
┌─────────────────────────────────────────────────────────────┐
│            LAYER 4: CREDIT RISK PIPELINE                     │
│   Feature Engineering: Ratios, Trends, NLP sentiment        │
│   PD Model: XGBoost — metrics TBD                           │
│   Risk Memos: LLM-generated — metrics TBD                   │
│   Governance: Prompt versioning, safety filters              │
│   Monitoring: Drift detection — TBD                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM minimum
- Tesseract OCR installed
- Anthropic API key (with vision support)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ocr-agentic-rag.git
cd ocr-agentic-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Create credit risk directories
mkdir -p credit_risk/feature_engineering \
         credit_risk/models \
         credit_risk/governance \
         credit_risk/monitoring \
         data/credit_risk/{lending_club,fiqa_sentiment,financebench,ectsum,credit_card_default,freddie_mac,home_credit,counterfactual_synthetic}
```

### Run Demo
```bash
# Quick demo (1 sample, ~3 seconds)
python run_e2e.py

# Local evaluation (80 samples, ~3 minutes)
python run_e2e.py --eval

# Full E2E demo
python examples/06_full_e2e_demo.py
```

---

## 📊 Evaluation Results

Evaluation is run via `eval_runner.py`; results are written under **`data/proof/`**.  
Demos in **`notebooks/`** read from `data/proof/` for any reported metrics.

**Categories with proof under `data/proof/`:**
- **Vision:** ChartQA, DocVQA, InfographicsVQA, MMMU (Accounting, Economics, Finance, Math) — *scores TBD*
- **RAG:** FinQA (and adversarial) — *scores TBD*

**Other categories** (OCR, full multimodal, other RAG datasets, Credit Risk, system tests) are not yet backed by artifacts in `data/proof/`. Scores will be added as evaluation progresses.

See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for methodology; current numbers are placeholders until filled from proof runs.

---

## 📁 Repository Structure
```
ocr-agentic-rag/
├── ocr_pipeline/              # Layer 1: OCR
│   ├── quality_assessment.py
│   ├── template_detector.py
│   ├── detection/            # 3-tier detection
│   └── recognition/          # Hybrid recognition + vision
├── rag_system/               # Layer 2: RAG
│   ├── chunking.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── multimodal_rag.py     # Layer 3: Multimodal
│   └── agentic/
│       ├── orchestrator.py
│       ├── tools.py
│       └── memory.py
├── credit_risk/              # Layer 4: Credit Risk (NEW)
│   ├── pipeline.py
│   ├── feature_engineering/
│   │   ├── ratio_builder.py
│   │   ├── trend_engine.py
│   │   └── nlp_signals.py
│   ├── models/
│   │   ├── pd_model.py
│   │   └── counterfactual.py
│   ├── governance/
│   │   ├── risk_memo_generator.py
│   │   ├── prompt_registry.py
│   │   ├── prompt_version.py
│   │   └── safety_filter.py
│   └── monitoring/
│       ├── data_drift.py
│       └── prediction_drift.py
├── evaluation/
│   ├── ocr_eval.py           # 6 OCR benchmarks
│   ├── multimodal_eval.py    # 8 multimodal benchmarks
│   ├── rag_eval.py           # 4 RAG benchmarks
│   ├── credit_risk_eval.py   # 6 credit risk benchmarks (NEW)
│   ├── e2e_functional_eval.py
│   ├── e2e_robustness_test.py
│   ├── e2e_bias_test.py
│   ├── e2e_adversarial_test.py
│   ├── e2e_load_test.py
│   └── e2e_full_suite.py
├── examples/
│   ├── 01_ocr_demo.py
│   ├── 02_rag_demo.py
│   ├── 03_e2e_demo.py
│   ├── 04_evaluation_demo.py
│   ├── 05_credit_risk_demo.py     # NEW
│   └── 06_full_e2e_demo.py        # NEW
├── scripts/
│   └── download_credit_datasets.sh # NEW
├── data/
│   ├── evaluation/
│   └── credit_risk/          # NEW: 8 datasets
├── run_e2e.py               # Main entry point (NEW)
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── EVALUATION_RESULTS.md
```

---

## 🎯 Use Cases

### OCR Layer
- Invoice processing (SROIE)
- Form extraction (FUNSD)

### Multimodal Layer
- Chart extraction, handwriting recognition
- Visual document QA (DocVQA, ChartQA, InfographicsVQA, MMMU — *see data/proof*)

### RAG Layer
- Financial QA (FinQA — *see data/proof*)
- Table reasoning (TAT-QA)

### Credit Risk Layer
- Default probability prediction (PD model)
- Credit deterioration
- Automated risk memo generation
- Covenant stress testing, what-if analysis
- Real-time drift monitoring

---

## 💻 Running on Different Platforms

### Local PC

**Specs:** 16GB RAM, i5-11500, no GPU  
**Mode:** `local` 
**Samples:** 80 total  
**Runtime:** ~3 minutes  
**Cost:** $0
```bash
python run_e2e.py --eval
```

---

### AWS SageMaker (Free Tier)

**Instance:** ml.t3.medium (2 vCPU, 4GB RAM)  
**Mode:** `sagemaker`  
**Samples:** 600 total  
**Runtime:** ~15-20 minutes  
**Cost:** $0 (within 250 hours/month free tier)

**Setup:**
```bash
# 1. Create S3 bucket
aws s3 mb s3://my-sagemaker-credit-risk

# 2. Generate sample datasets
python scripts/create_sample_datasets.py --mode sagemaker

# 3. Upload to S3
aws s3 sync data/credit_risk/ s3://my-sagemaker-credit-risk/data/

# 4. Launch SageMaker notebook
# Use ml.t3.medium instance type

# 5. Run in notebook
python run_e2e.py --mode sagemaker --s3-bucket my-sagemaker-credit-risk --eval
```

**SageMaker Tips:**
- Use **ml.t3.medium** (free tier eligible)
- Process in batches of 10 to avoid memory issues
- Results auto-saved to S3
- Monitor with CloudWatch (free)

---

### Production (Full Datasets)

**Specs:** 16GB+ RAM, GPU recommended  
**Mode:** `production`  
**Samples:** 3.7M total  
**Runtime:** ~2-4 hours  
**Cost:** $0 (local) or ~$5-10 (AWS)
```bash
# Download full datasets
bash scripts/download_all_datasets.sh

# Run full evaluation
python run_e2e.py --mode production --eval
```

---

## 🔧 Technology Stack

### OCR
- **Detection:** OpenCV (classical), PaddleOCR (DL), Template Cache
- **Recognition:** Tesseract, PaddleOCR, Claude Vision
- **Optimization:** ONNX Runtime (12x speedup)

### RAG
- **Chunking:** Structure-preserving
- **Embeddings:** BGE-M3 (HuggingFace)
- **Retrieval:** FAISS + BM25 (hybrid)
- **Reranking:** BGE-reranker-v2-m3
- **Orchestration:** LangGraph
- **LLM:** Claude Sonnet 4

### Multimodal
- **Vision Model:** Claude 4.6 Sonnet Vision
- **Chart Understanding:** Vision-first (95% accuracy)

### Credit Risk (NEW)
- **Feature Engineering:** Pandas, NumPy
- **NLP:** FinBERT (ProsusAI/finbert)
- **ML Models:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **LLM:** Claude Sonnet 4 (risk memos)
- **Monitoring:** Scipy (KS test), Evidently AI
- **Governance:** SQLite (prompt registry)

### Infrastructure
- **Monitoring:** OpenTelemetry, Prometheus
- **Cloud:** AWS SageMaker, S3
- **Storage:** PostgreSQL, SQLite

---

## 📖 Documentation

### Core Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
- [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) - All 20 benchmark results
- [data/credit_risk/README.md](data/credit_risk/README.md) - Dataset download guide

### Quick References
- **OCR Demo:** `examples/01_ocr_demo.py`
- **RAG Demo:** `examples/02_rag_demo.py`
- **Credit Risk Demo:** `examples/05_credit_risk_demo.py`
- **Full E2E Demo:** `examples/06_full_e2e_demo.py`
- **Evaluation Suite:** `python run_e2e.py --eval`

---

## 🔒 MAS FEAT Compliance

### Fairness
- Bias gap target: &lt;10% threshold — *TBD*

### Ethics
- Human-in-the-loop for high-risk decisions
- Autonomous approval for low-risk cases only

### Accountability
- Audit trail and lineage tracking (data → features → model → decision)
- Prompt versioning (LLM calls logged)

### Transparency
- SHAP explainability for PD model
- LLM explanations with citation tracking
- Drift monitoring — *TBD*

---

## ⚡ Performance Benchmarks

Latency, throughput, and cost figures are TBD and will be updated from runs that write to `data/proof/`.  
See `eval_runner.py` and `data/proof/` for current evaluation outputs.

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

**Datasets:**
- Lending Club (Kaggle)
- FiQA, FinanceBench, (HuggingFace)
- SROIE, FUNSD, DocVQA (Academia)

**Frameworks:**
- PaddleOCR, LlamaIndex, LangGraph
- HuggingFace Transformers
- Anthropic Claude
- XGBoost, scikit-learn, SHAP

---

**Contact:** Lee Ming Loon | Singapore  