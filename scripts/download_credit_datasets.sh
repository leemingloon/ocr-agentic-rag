#!/bin/bash

# Download Credit Risk Evaluation Datasets
# Compatible with: Local, SageMaker, Production

set -e

echo "=========================================="
echo "Credit Risk Dataset Downloader"
echo "=========================================="
echo ""
echo "Mode: $1"  # local, sagemaker, or production
MODE=${1:-local}

# Create directories
mkdir -p data/credit_risk/{lending_club,fiqa_sentiment,financebench,ectsum,credit_card_default,freddie_mac,home_credit,counterfactual_synthetic}

echo ""
echo "✓ Created directory structure"
echo ""

# Tier 1 Datasets (Free)
echo "Tier 1 Datasets (Must-Have - All FREE)"
echo "=========================================="

echo ""
echo "1. Lending Club (2.9M loans)"
echo "   Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club"
echo "   Purpose: PD model training"
echo "   Download: Requires Kaggle API"
echo "   Command: kaggle datasets download -d wordsforthewise/lending-club"

echo ""
echo "2. FiQA Sentiment (1,173 samples)"
echo "   Source: https://huggingface.co/datasets/financial_phrasebank"
echo "   Purpose: NLP sentiment validation"
echo "   Download: Via HuggingFace datasets library"

echo ""
echo "3. FinanceBench (10,231 Q&A)"
echo "   Source: https://huggingface.co/datasets/PatronusAI/financebench"
echo "   Purpose: Risk memo Q&A validation"
echo "   Download: Via HuggingFace datasets library"

echo ""
echo "4. ECTSum (2,425 summaries)"
echo "   Source: https://github.com/rajdeep345/ECTSum"
echo "   Purpose: Risk memo summarization"
echo "   Download: git clone https://github.com/rajdeep345/ECTSum.git"

echo ""
echo "5. Credit Card Default UCI (30K)"
echo "   Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
echo "   Purpose: Drift detection"
echo "   Download: Direct download from UCI"

echo ""
echo "Tier 2 Datasets (Nice-to-Have - FREE)"
echo "=========================================="

echo ""
echo "6. Freddie Mac (500K+)"
echo "   Source: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset"
echo "   Note: Requires registration"

echo ""
echo "7. Home Credit (307K)"
echo "   Source: https://www.kaggle.com/c/home-credit-default-risk"
echo "   Download: kaggle competitions download -c home-credit-default-risk"

echo ""
echo "8. Synthetic Counterfactuals (1K)"
echo "   Generated internally"
echo "   Command: python scripts/generate_counterfactuals.py"

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""

if [ "$MODE" = "local" ]; then
    echo "For LOCAL mode (80 samples):"
    echo "  python scripts/create_sample_datasets.py --mode local"
elif [ "$MODE" = "sagemaker" ]; then
    echo "For SAGEMAKER mode (600 samples):"
    echo "  1. Create S3 bucket: aws s3 mb s3://your-sagemaker-bucket"
    echo "  2. Generate samples: python scripts/create_sample_datasets.py --mode sagemaker"
    echo "  3. Upload to S3: aws s3 sync data/credit_risk/ s3://your-sagemaker-bucket/data/"
else
    echo "For PRODUCTION mode (3.7M samples):"
    echo "  Download all datasets using instructions in data/credit_risk/DATASETS.md"
fi

echo ""
echo "See data/credit_risk/DATASETS.md for complete instructions"
echo ""