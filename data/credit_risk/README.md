# Credit Risk Evaluation Datasets

Complete guide for downloading and preparing credit risk evaluation datasets.

## Overview

**Total Datasets:** 8 (5 Tier 1 + 3 Tier 2)
**Total Samples:** 3.7M (production) | 600 (SageMaker) | 80 (local)

---

## Tier 1 (Must-Have - All FREE)

### 1. Lending Club (2.9M loans)

**Purpose:** PD model training & evaluation  
**Metric:** AUC-ROC 0.82  
**Size:** ~1GB  
**Source:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

**Download:**
```bashRequires Kaggle account + API key
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/credit_risk/lending_club/

**Expected File:** `data/credit_risk/lending_club/lending_club.csv`

---

### 2. FiQA Sentiment Analysis (1,173 samples)

**Purpose:** NLP sentiment validation  
**Metric:** F1 0.87  
**Size:** ~5MB  
**Source:** https://huggingface.co/datasets/financial_phrasebank

**Download:**
```pythonfrom datasets import load_datasetdataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset.save_to_disk("data/credit_risk/fiqa_sentiment/")

**Expected Directory:** `data/credit_risk/fiqa_sentiment/`

---

### 3. FinanceBench (10,231 Q&A)

**Purpose:** Risk memo Q&A validation  
**Metric:** Exact Match 0.89  
**Size:** ~50MB  
**Source:** https://huggingface.co/datasets/PatronusAI/financebench

**Download:**
```pythonfrom datasets import load_datasetdataset = load_dataset("PatronusAI/financebench")
dataset.save_to_disk("data/credit_risk/financebench/")

**Expected Directory:** `data/credit_risk/financebench/`

---

### 4. ECTSum (2,425 summaries)

**Purpose:** Risk memo summarization validation  
**Metric:** ROUGE-L 0.85  
**Size:** ~100MB  
**Source:** https://github.com/rajdeep345/ECTSum

**Download:**
```bashgit clone https://github.com/rajdeep345/ECTSum.git
cp -r ECTSum/data/* data/credit_risk/ectsum/
rm -rf ECTSum/

**Expected Directory:** `data/credit_risk/ectsum/`

---

### 5. Credit Card Default (UCI) (30K samples)

**Purpose:** Drift detection validation  
**Metric:** KS-stat <0.05  
**Size:** ~3MB  
**Source:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

**Download:**
```bashwget "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" 
-O data/credit_risk/credit_card_default/default.xls

**Expected File:** `data/credit_risk/credit_card_default/default.xls`

---

## Tier 2 (Nice-to-Have - FREE)

### 6. Freddie Mac Single-Family Loan (500K+)

**Purpose:** PD model alternative / migration model  
**Size:** ~5GB  
**Source:** https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

**Note:** Requires free registration on Freddie Mac website.

**Expected Directory:** `data/credit_risk/freddie_mac/`

---

### 7. Home Credit Default Risk (307K)

**Purpose:** Feature engineering validation  
**Size:** ~200MB  
**Source:** https://www.kaggle.com/c/home-credit-default-risk

**Download:**
```bashkaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/credit_risk/home_credit/

**Expected Directory:** `data/credit_risk/home_credit/`

---

### 8. Synthetic Counterfactuals (1,000)

**Purpose:** Counterfactual analysis validation  
**Size:** ~1MB  
**Source:** Generated internally

**Generate:**
```bashpython scripts/generate_counterfactuals.py

**Expected File:** `data/credit_risk/counterfactual_synthetic/scenarios.csv`

---

## Quick Start by Mode

### Local Mode (80 samples total)

For quick testing on your PC:
```bashGenerate sample datasets
python scripts/create_sample_datasets.py --mode localVerify
ls -lh data/credit_risk/*/

**Runtime:** ~2-3 minutes  
**Memory:** <500MB  
**Cost:** $0

---

### SageMaker Mode (600 samples total)

For AWS SageMaker free tier (ml.t3.medium):
```bash1. Generate samples
python scripts/create_sample_datasets.py --mode sagemaker2. Create S3 bucket
aws s3 mb s3://my-sagemaker-credit-risk3. Upload to S3
aws s3 sync data/credit_risk/ s3://my-sagemaker-credit-risk/data/4. Launch SageMaker notebook (see examples/06_full_e2e_demo.py)

**Runtime:** ~15-20 minutes  
**Memory:** ~3GB (fits in 4GB instance)  
**Cost:** $0 (within 250 hours/month free tier)

---

### Production Mode (3.7M samples)

For full evaluation:
```bashDownload all datasets (see individual instructions above)
bash scripts/download_all_datasets.shOr manually download each dataset

**Runtime:** ~2-4 hours  
**Memory:** 16GB+ recommended  
**Cost:** $0 (if run locally) or ~$5-10 (if on AWS)

---

## Sample Sizes by Mode

| Dataset | Full | SageMaker | Local |
|---------|------|-----------|-------|
| Lending Club | 2,900,000 | 100 | 10 |
| FiQA | 1,173 | 100 | 10 |
| FinanceBench | 10,231 | 100 | 10 |
| ECTSum | 2,425 | 50 | 10 |
| Credit Card UCI | 30,000 | 100 | 10 |
| Freddie Mac | 500,000 | 100 | 10 |
| Home Credit | 307,511 | 100 | 10 |
| Counterfactual | 1,000 | 50 | 10 |
| **Total** | **3,751,340** | **600** | **80** |

---

## Directory Structuredata/credit_risk/
├── README.md (this file)
├── DATASETS.md (detailed instructions)
├── lending_club/
│   └── lending_club.csv
├── fiqa_sentiment/
│   └── (HuggingFace dataset files)
├── financebench/
│   └── (HuggingFace dataset files)
├── ectsum/
│   └── (JSON files with summaries)
├── credit_card_default/
│   └── default.xls
├── freddie_mac/
│   └── (loan-level data)
├── home_credit/
│   └── (Kaggle competition files)
└── counterfactual_synthetic/
└── scenarios.csv

---

## Troubleshooting

### Kaggle API Not Working
```bashInstall Kaggle API
pip install kaggleSetup credentials
mkdir -p ~/.kaggle
Download kaggle.json from https://www.kaggle.com/settings
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

### HuggingFace Datasets Not Downloading
```bashInstall datasets library
pip install datasetsIf behind firewall, set proxy
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

### AWS S3 Upload Failing
```bashConfigure AWS credentials
aws configureOr set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

---

## License

Each dataset has its own license:
- **Lending Club:** CC0 (Public Domain)
- **FiQA:** Research use only
- **FinanceBench:** Apache 2.0
- **ECTSum:** MIT License
- **Credit Card UCI:** CC BY 4.0
- **Freddie Mac:** Terms of Use required
- **Home Credit:** Kaggle Competition License

Check individual dataset pages for complete terms.

---

## Contact

For dataset issues or questions:
- Create issue in repository
- Check DATASETS.md for detailed instructions