#!/bin/bash

# Download Evaluation Datasets Script
# Downloads all 10 benchmark datasets

set -e

echo "=========================================="
echo "Downloading Evaluation Datasets (10 benchmarks)"
echo "=========================================="

# Create evaluation directory
mkdir -p data/evaluation

echo ""
echo "Note: Most datasets require manual download due to licensing."
echo "This script will provide download links and instructions."
echo ""

# Dataset information
cat > data/evaluation/DATASETS.md << 'EOF'
# Evaluation Datasets

## OCR Benchmarks (6 datasets)

### 1. OmniDocBench v1.5
- **Source:** https://github.com/opendatalab/OmniDocBench
- **Download:** Follow instructions on GitHub
- **Size:** ~50GB (sample: 1GB)
- **Location:** `data/evaluation/omnidocbench_sample/`

### 2. SROIE (ICDAR 2019)
- **Source:** https://rrc.cvc.uab.es/?ch=13
- **Download:** Register and download from competition page
- **Size:** ~1GB
- **Location:** `data/evaluation/sroie_sample/`

### 3. FUNSD (ICDAR 2019)
- **Source:** https://guillaumejaume.github.io/FUNSD/
- **Download:** Direct download from website
- **Size:** ~300MB
- **Location:** `data/evaluation/funsd_sample/`

### 4. DocVQA (CVPR 2021)
- **Source:** https://rrc.cvc.uab.es/?ch=17
- **Download:** Register and download
- **Size:** ~12GB (sample: 500MB)
- **Location:** `data/evaluation/docvqa_sample/`

### 5. InfographicsVQA (WACV 2022)
- **Source:** https://www.docvqa.org/datasets/infographicvqa
- **Download:** Download from DocVQA website
- **Size:** ~5GB (sample: 200MB)
- **Location:** `data/evaluation/infographicsvqa_sample/`

### 6. DUDE (NeurIPS 2023)
- **Source:** https://dude-dataset.github.io/
- **Download:** Follow GitHub instructions
- **Size:** ~2GB (sample: 100MB)
- **Location:** `data/evaluation/dude_sample/`

## RAG Benchmarks (4 datasets)

### 7. HotpotQA
- **Source:** https://hotpotqa.github.io/
- **Download:** Direct download
- **Size:** ~500MB
- **Location:** `data/evaluation/hotpotqa_sample/`

### 8. FinQA (NeurIPS 2021)
- **Source:** https://github.com/czyssrs/FinQA
- **Download:** Clone repository and download data
- **Size:** ~100MB
- **Location:** `data/evaluation/finqa_sample.json`

### 9. TAT-QA (ACL 2021)
- **Source:** https://github.com/NExTplusplus/TAT-QA
- **Download:** Follow GitHub instructions
- **Size:** ~50MB
- **Location:** `data/evaluation/tatqa_sample.json`

### 10. BIRD-SQL
- **Source:** https://bird-bench.github.io/
- **Download:** Follow website instructions
- **Size:** ~200MB
- **Location:** `data/evaluation/bird_sql_sample.json`

## Quick Start

For evaluation without full datasets:
1. Create sample directories as listed above
2. Add 10-20 samples per dataset for quick testing
3. Run: `python examples/04_evaluation_demo.py`

For production evaluation:
1. Download complete datasets from sources above
2. Extract to specified locations
3. Update sample_size in evaluation scripts to full dataset size
EOF

echo "✓ Created DATASETS.md with download instructions"
echo ""
echo "Download instructions written to: data/evaluation/DATASETS.md"
echo ""
echo "Next steps:"
echo "1. Read data/evaluation/DATASETS.md for download links"
echo "2. Download datasets you need (start with OmniDocBench, SROIE, FinQA)"
echo "3. Place in data/evaluation/ directories"
echo "4. Run: python examples/04_evaluation_demo.py"