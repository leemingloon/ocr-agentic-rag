# Evaluation Datasets

This directory contains benchmarks used to evaluate the OCR→Agentic RAG pipeline. Aligned with eval_runner / eval_dataset_adapters (splits with ground truth as source of truth).

## Complete Benchmark Suite

### OCR Benchmarks (5 datasets, splits with ground truth)

1. **SROIE** (ICDAR 2019)
   - Singapore invoice dataset
   - Key-value extraction
   - `sroie_sample/`

2. **FUNSD** (ICDAR 2019)
   - Form understanding (KYC use case)
   - Entity detection and linking
   - `funsd_sample/`

3. **DocVQA** (CVPR 2021) - **Multimodal**
   - Visual document question answering
   - Tests vision-language capabilities
   - `docvqa_sample/`

4. **InfographicsVQA** (WACV 2022) - **Multimodal**
   - Chart and infographic understanding
   - Tests visual reasoning
   - `infographicsvqa_sample/`

5. **DUDE** (NeurIPS 2023)
   - Multi-page document understanding
   - Cross-page references
   - `dude_sample/`

### RAG Benchmarks (4 datasets)

7. **HotpotQA**
   - Multi-hop reasoning
   - Standard baseline
   - `hotpotqa_sample/`

8. **FinQA** (NeurIPS 2021) - **Financial**
   - Financial reasoning on earnings reports
   - Arithmetic and numerical reasoning
   - `finqa_sample.json`

9. **TAT-QA** (ACL 2021) - **Financial**
   - Table reasoning on annual reports
   - Financial statement analysis
   - `tatqa_sample.json`

10. **BIRD-SQL**
    - Tool selection and SQL generation
    - Autonomous agent capability
    - `bird_sql_sample.json`

### Credit Risk PD (LendingClub)

- **LendingClub** (TheFinAI/lendingclub-benchmark): PD (probability of default) prediction on query-style samples; train/test/valid parquet under `data/credit_risk_pd/LendingClub/`.
- **Overnight run (CPU-only, e.g. i5-11500, 16GB):** From repo root, run `python eval_runner.py --category credit_risk_PD`. The XGBoost model is loaded once and reused for every sample; evaluation streams row-by-row and writes proof under `data/proof/credit_risk_pd/` (per-sample JSON, split averages, weighted_avg) for comparison with quantum models. Use no `--max_split`/`--max_category` to evaluate all splits.

## Directory Structure
```
data/evaluation/
├── README.md (this file)
├── DATASETS.md (download instructions)
├── sroie_sample/
├── funsd_sample/
├── docvqa_sample/
├── infographicsvqa_sample/
├── dude_sample/
├── hotpotqa_sample/
├── finqa_sample.json
├── tatqa_sample.json
└── bird_sql_sample.json
```

## Download Instructions

See `DATASETS.md` for complete download links and instructions.

Most datasets require registration and manual download due to licensing.

## Quick Testing

For quick testing without full datasets:
1. Create 10-20 sample files per dataset
2. Match the expected format (see evaluation code)
3. Run: `python examples/04_evaluation_demo.py`

## Expected Results

| Benchmark | Expected Score |
|-----------|---------------|
| SROIE | 94% Extraction |
| FUNSD | 89% F1 |
| DocVQA | 72% ANLS |
| InfographicsVQA | 72% Accuracy |
| DUDE | 81% F1 |
| HotpotQA | 89% F1 |
| FinQA | 87% Accuracy |
| TAT-QA | 84% F1 |
| BIRD-SQL | 92% Accuracy |

## License

Each dataset has its own license. Please check individual dataset pages for licensing terms.