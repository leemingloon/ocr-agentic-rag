# Evaluation Datasets

This directory contains the 10 industry-standard benchmarks used to evaluate the OCR→Agentic RAG pipeline.

## Complete Benchmark Suite

### OCR Benchmarks (6 datasets)

1. **OmniDocBench v1.5** (CVPR 2025)
   - Industry standard OCR benchmark
   - Text detection, table structure, layout
   - `omnidocbench_sample/`

2. **SROIE** (ICDAR 2019)
   - Singapore invoice dataset
   - Key-value extraction
   - `sroie_sample/`

3. **FUNSD** (ICDAR 2019)
   - Form understanding (KYC use case)
   - Entity detection and linking
   - `funsd_sample/`

4. **DocVQA** (CVPR 2021) - **Multimodal**
   - Visual document question answering
   - Tests vision-language capabilities
   - `docvqa_sample/`

5. **InfographicsVQA** (WACV 2022) - **Multimodal**
   - Chart and infographic understanding
   - Tests visual reasoning
   - `infographicsvqa_sample/`

6. **DUDE** (NeurIPS 2023)
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

## Directory Structure
```
data/evaluation/
├── README.md (this file)
├── DATASETS.md (download instructions)
├── omnidocbench_sample/
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
| OmniDocBench | 85% Text F1 |
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