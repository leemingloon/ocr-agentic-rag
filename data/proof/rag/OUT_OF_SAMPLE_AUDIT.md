# Out-of-sample RAG evaluation — full audit

This document audits that the **out-of-sample** test (FinQA test split, TAT-QA dev split) does not use QA or corpus from the **reference** set (train_qa.json, tatqa_dataset_test_gold.json, finqa_train_samples.json, tatqa_test_samples.json), and that proof sample files are not contaminated by the wrong source.

---

## 1. Reference vs eval splits

| Role | FinQA | TAT-QA |
|------|--------|--------|
| **Reference (in-sample)** | train_qa.json → produces finqa_train_samples.json | tatqa_dataset_test_gold.json → produces tatqa_test_samples.json |
| **Out-of-sample eval** | test.json (FinQA test) | tatqa_dataset_dev.json (TAT-QA dev) |

No QA from the reference set may appear in the out-of-sample eval. Overlapping samples are excluded at eval time via hardcoded exclude sets in the adapters.

---

## 2. Index build (notebook: `notebooks/rag_build_retriever_index.ipynb`)

### 2.1 FinQA indices

- **FinQA train index** (section 4/5): Built from **`data/rag/FinQA/train/train_qa.json`** only.  
  Output: `data/rag/FinQA/train/finqa_retriever_index/`.  
  Used when running eval with **`--split train`** (in-sample).

- **FinQA test index** (section 7): Built from **`data/rag/FinQA/test/test.json`** only.  
  Script: `scripts/build_finqa_embeddings_colab.py --train_qa data/rag/FinQA/test/test.json --output data/rag/FinQA/test/finqa_retriever_index`.  
  Output: `data/rag/FinQA/test/finqa_retriever_index/`.  
  Used when running eval with **`--split test`** (out-of-sample).

**Conclusion:** The FinQA **test** index does **not** include any corpus from train_qa.json, finqa_train_samples.json, tatqa_test_samples.json, or tatqa_dataset_test_gold.json. It is built solely from test.json.

### 2.2 TAT-QA index

- **TAT-QA index** (section 9): Built from **all three splits**:  
  `tatqa_dataset_train.json`, `tatqa_dataset_dev.json`, `tatqa_dataset_test_gold.json`  
  (see `scripts/build_tatqa_embeddings_colab.py`: `splits_and_files`).  
  Output: `data/rag/TAT-QA/tatqa_retriever_index/`.  
  This **single** index is used for both in-sample (test split) and out-of-sample (dev split) eval.

So for **TAT-QA dev** (out-of-sample), the index **does** contain documents from train, dev, and test_gold. The “out-of-sample” guarantee is enforced by **which QA are evaluated**, not by removing test_gold from the index: dev QA are streamed from tatqa_dataset_dev.json, and any dev sample whose (Q,A) appears in the combined reference (train_qa + tatqa_test_gold) is excluded via `TATQA_DEV_EXCLUDE_SAMPLE_IDS`. So no **QA** from the reference set is ever run; the index can still contain test_gold **documents** for retrieval. If you need a strict “dev index with zero test_gold corpus” for a paper, you would need a separate index built from train+dev only (not implemented in the current notebook).

---

## 3. Out-of-sample run: `scripts/run_rag_out_of_sample.sh` → `eval_runner.py`

### 3.1 What the script runs

```bash
# FinQA out-of-sample
python eval_runner.py --category rag --dataset FinQA --max_split 200 --max_category 200 --debug --export_predictions_txt --split test

# TAT-QA out-of-sample
python eval_runner.py --category rag --dataset TATQA --max_split 200 --max_category 200 --debug --export_predictions_txt --split dev
```

So: **FinQA uses `--split test`**, **TAT-QA uses `--split dev`**. No reference split (train for FinQA, test for TAT-QA) is requested.

### 3.2 Code trace: where do samples come from?

1. **`eval_runner.py`** (e.g. `run_evaluation()` / main) parses `--split` into `run_split` and passes it to `evaluate_dataset(..., dataset_split=run_split)`.

2. **`evaluate_dataset()`** (eval_runner.py) builds the adapter (e.g. `FinQAAdapter` or `TATQAAdapter`) and calls:
   ```python
   dataset_iter = adapter.load_split(
       dataset_split=dataset_split,  # "test" or "dev"
       max_samples_per_split=...,
       max_samples_per_category=...,
       only_splits_with_gt=only_gt,
   )
   ```
   So the **only** source of samples is `adapter.load_split(dataset_split=run_split)`.

3. **FinQAAdapter.load_split()** (eval_dataset_adapters.py):
   - For `split == "test"`: reads **`FILE_MAPPING["test"]["dataset_path"]`** = **`data/rag/FinQA/test/test.json`**.
   - For each entry, computes `sample_id` (e.g. `entry.get("id")` or `filename-{idx}`).
   - **Skips** any `sample_id` in **`FINQA_TEST_EXCLUDE_SAMPLE_IDS`** (overlap with reference).
   - **Does not** read `finqa_train_samples.json` or `train_qa.json` as the QA source; those are only used when `--split train` is run.

4. **TATQAAdapter.load_split()** (eval_dataset_adapters.py):
   - For `split == "dev"`: uses **`split_files["dev"]`** = **`tatqa_dataset_dev.json`** (path: `data/rag/TAT-QA/tatqa_dataset_dev.json`).
   - For each question, **skips** if `sample_id` is in **`TATQA_DEV_EXCLUDE_SAMPLE_IDS`** (overlap with reference).
   - **Does not** read `tatqa_test_samples.json` or `tatqa_dataset_test_gold.json` as the QA source for this run; those are only used when `--split test` is run.

5. **Proof output:** Results are written to:
   - FinQA test: `data/proof/rag/finqa/test/finqa_test_samples.json`
   - TAT-QA dev: `data/proof/rag/tatqa/dev/tatqa_dev_samples.json`  
   So **finqa_train_samples.json** and **tatqa_test_samples.json** are **never read** during the out-of-sample run; they are produced by separate in-sample runs (`--split train` and `--split test`).

6. **Retriever used:** For FinQA, `_get_rag_retriever_for_dataset(dataset_name, dataset_split=dataset_split)` (eval_runner.py) uses `dataset_split` to choose the index: when `dataset_split == "test"` it loads the **test** index from `data/rag/FinQA/test/finqa_retriever_index` (built from test.json only). So retrieval for out-of-sample FinQA test uses only the test-index corpus.

**Conclusion:** The out-of-sample run **never** uses QA from finqa_train_samples.json, tatqa_test_samples.json, train_qa.json, or tatqa_dataset_test_gold.json. It streams QA only from test.json (FinQA) or tatqa_dataset_dev.json (TAT-QA), after excluding overlap IDs. Proof files for the reference splits are not read in this flow.

---

## 4. Contamination check: proof samples vs wrong source

**Question:** Could any QA in **finqa_train_samples.json** have come from **test.json**? Could any QA in **tatqa_test_samples.json** have come from **tatqa_dataset_dev.json**? If yes, that would be contamination and weaken the out-of-sample claim.

**Check:** Run:

```bash
python scripts/audit_out_of_sample_contamination.py
```

**Result (audit run):**

- **FinQA:** All 200 `sample_id`s in finqa_train_samples.json appear in **train_qa.json**; **0** appear in test.json. No contamination.
- **TAT-QA:** All 200 `sample_id`s in tatqa_test_samples.json appear in **tatqa_dataset_test_gold.json**; **0** appear in tatqa_dataset_dev.json. No contamination.

So the proof sample files used as “reference” are **not** contaminated by the out-of-sample sources (test.json, tatqa_dataset_dev.json). Safe for academic use.

---

## 5. Summary table

| Check | Status |
|-------|--------|
| FinQA test index built only from test.json (no train_qa / reference) | Yes (notebook section 7) |
| Out-of-sample run loads QA only from test.json (FinQA) / tatqa_dev (TAT-QA) | Yes (adapter load_split) |
| Out-of-sample run never reads finqa_train_samples.json or tatqa_test_samples.json | Yes (different split → different file written) |
| Overlap samples excluded at eval (FINQA_TEST_EXCLUDE_SAMPLE_IDS, TATQA_DEV_EXCLUDE_SAMPLE_IDS) | Yes (adapter skips those IDs) |
| finqa_train_samples.json contains 0 QA from test.json | Verified (audit script) |
| tatqa_test_samples.json contains 0 QA from tatqa_dataset_dev.json | Verified (audit script) |
| TAT-QA index for dev eval contains test_gold corpus | Yes (single index from train+dev+test); exclusion is QA-level only |

---

## 6. Reproducing the audit

1. **Index build:** Follow `notebooks/rag_build_retriever_index.ipynb` (sections 4, 7, 9). Ensure FinQA test index is built from test.json only.
2. **Out-of-sample run:** `bash scripts/run_rag_out_of_sample.sh` (calls eval_runner with `--split test` and `--split dev`).
3. **Contamination:** `python scripts/audit_out_of_sample_contamination.py` (run from repo root).
