# Root cause: finqa_avg.json vs finqa_train_avg.json

## Observed difference

| Source | sample_count | relaxed_exact_match | exact_match | f1 |
|--------|--------------|----------------------|-------------|-----|
| **finqa_avg.json** | 200 | 0.9200 (184/200) | 0.9050 (181/200) | 0.0225 |
| **finqa_train_avg.json** | 199 | 0.9196 (183/199) | 0.9045 (180/199) | 0.0226 |

## Root cause (by design)

- **finqa_train_avg.json** = **split-level** summary for the **train** split only.
  - Path: `data/proof/rag/finqa/train/finqa_train_avg.json`
  - Built from `finqa_train_samples.json` (199 rows with metrics).
  - Schema: flat `sample_count`, `relaxed_exact_match`, `exact_match`, `f1`.

- **finqa_avg.json** = **dataset-level** summary over **all splits** (train + test).
  - Path: `data/proof/rag/finqa/finqa_avg.json`
  - Built by **aggregating split-level avg files**: it reads every `finqa_{split}_avg.json` under `data/proof/rag/finqa/<split>/` and merges them.
  - Current splits present: **test** (1 sample), **train** (199 samples) → total **200**.
  - Schema: `dataset`, `sample_count`, `weighted_metrics`, `timestamp`, `splits`, `splits_breakdown`.

So the numeric difference comes from the **extra test split**:

- **test** has 1 sample and `finqa_test_avg.json`: REM 1/1, EM 1/1, f1 0.0.
- **train** has 199 samples and `finqa_train_avg.json`: REM 183/199, EM 180/199, f1 0.0226.

Aggregation (in `eval_postprocess_utils.build_rag_dataset_avg_payload` and `eval_runner.py` refresh path):

- `sample_count` = 199 + 1 = **200**.
- `relaxed_exact_match` = (183 + 1) / 200 = **184/200**.
- `exact_match` = (180 + 1) / 200 = **181/200**.
- `f1` = (0.0226 * 199 + 0.0 * 1) / 200 = **0.0225** (rounded).

## Code path

1. **Split-level** (`finqa_train_avg.json`, `finqa_test_avg.json`):
   - Written in `eval_runner.py` after aggregating that split’s `*_samples.json` (e.g. `aggregate_rag_split_metrics(rows_for_agg)` for RAG).
   - Path: `split_dir / f"{dataset_name.lower()}_{split_name}_avg.json"` (e.g. `finqa/train/finqa_train_avg.json`).

2. **Dataset-level** (`finqa_avg.json`):
   - Built in `refresh_dataset_weighted_avg_from_files()`:
     - Iterates over `dataset_proof_dir.iterdir()` (subdirs = splits: `train`, `test`).
     - For each split, reads `{dataset}_{split}_avg.json` into `split_avgs_from_files`.
   - Then calls `build_rag_dataset_avg_payload(dataset_name, split_avgs_from_files, timestamp)` and writes `dataset_proof_dir / f"{dataset_name.lower()}_avg.json"`.

So **any** split that has a `finqa_{split}_avg.json` under `data/proof/rag/finqa/<split>/` is included in **finqa_avg.json**. Right now that’s **test** (1 sample) and **train** (199 samples), hence 200 and the small metric delta.

## Summary

- **finqa_train_avg.json** = train split only (199 samples).
- **finqa_avg.json** = train + test (199 + 1 = 200 samples).
- The difference is intentional: one is per-split, the other is dataset-wide. To make them match you would either (a) remove or ignore the test split when building the dataset avg, or (b) treat finqa_avg as “all splits” and finqa_train_avg as “train only” and keep both.
