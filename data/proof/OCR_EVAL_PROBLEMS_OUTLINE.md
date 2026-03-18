# OCR evaluation: problems outline for diagnosis and fixes

**Purpose:** This document is a concise, factual outline of the current state and known problems with OCR evaluation under `data/proof`. It is intended for another agent (e.g. a large language model) to diagnose root causes and recommend concrete fixes.

---

## 1. Intended design (what should happen)

- **Proof layout:** Per category `ocr`, proof lives under `data/proof/ocr/<dataset>/<split>/`:
  - `{dataset}_{split}_samples.json`: list of per-sample rows with `sample_id`, `split`, `ground_truth`, `input_text`, `prediction`, `prediction_error`, `metrics`.
  - `{dataset}_{split}_avg.json`: split-level aggregate with `sample_count` and `*_mean` metrics (e.g. `entity_match_mean`, `word_recall_mean`).
  - `{dataset}_avg.json` at dataset level; category-level and `eval_summary.json` roll up from these.
- **Metrics:**
  - **SROIE:** `entity_match`, `entity_matched`, `entity_total` (per sample); split/dataset use `entity_match_mean`.
  - **FUNSD:** `word_recall`, `words_matched`, `words_gt` (per sample); split/dataset use `word_recall_mean`.
- **Single entry point:** All OCR metrics are produced by `eval_postprocessing_utils.compute_ocr_metrics(prediction, ground_truth, dataset_name)` (SROIE: entity extraction + normalize + soft CER; FUNSD: substring + normalize + fuzzy word match).
- **Engines:** By default `eval_runner.py --category ocr` uses **PaddleOCR** (full det+rec). Optionally `OCR_EVAL_USE_TESSERACT=1` uses Tesseract path (with preprocessing). If PaddleOCR fails to load, pipeline falls back to Tesseract.

---

## 2. Current state of proof data

- **Per-sample files:** `data/proof/ocr/sroie/train/sroie_train_samples.json` and `data/proof/ocr/funsd/train/funsd_train_samples.json` exist and contain:
  - `ground_truth`: for SROIE a dict (company, date, address, total); for FUNSD a list of words.
  - `prediction`: raw OCR text string.
  - `metrics`: e.g. `entity_match`, `entity_matched`, `entity_total` (SROIE) or `word_recall`, `words_matched`, `words_gt` (FUNSD).
- **Split avg files:** `*_train_avg.json` (and any `*_test_avg.json`) currently contain **only** `{"sample_count": N}`. They **do not** contain `entity_match_mean`, `word_recall_mean`, or other `*_mean` keys. So:
  - Category/dataset weighted averages that read from these files see no OCR metrics to aggregate.
  - `eval_summary.json` and category-level summaries therefore do not reflect OCR accuracy meaningfully.
- **Cause of missing _mean:** When proof is written by the **notebook** (`demo_hybrid_ocr_eval.ipynb`), `write_ocr_proof` sets `split_avg = {"sample_count": sample_count}` only and does not compute or write `entity_match_mean` / `word_recall_mean`. When proof is written by **eval_runner**, it does call `aggregate_metrics(split_metric_rows)` and should write `*_mean` into the split avg file—so if the last OCR run was from the notebook, the avg files will lack metrics.

---

## 3. Known problems (list for diagnosis)

### 3.1 Proof and aggregation

| # | Problem | Where it shows | Likely cause |
|---|--------|----------------|---------------|
| P1 | Split avg files (`*_avg.json`) contain only `sample_count`, no `entity_match_mean` or `word_recall_mean`. | `data/proof/ocr/<dataset>/<split>/*_avg.json` | Notebook `write_ocr_proof` does not aggregate per-sample metrics into `split_avg`; or eval_runner was not the last writer. |
| P2 | Category-level and eval_summary may show no or zero OCR metrics. | `data/proof/ocr/*/eval_summary.json`, category avg | Aggregation reads `*_mean` from split avg files; if those keys are missing (P1), weighted averages are empty or zero. |
| P3 | Inconsistent proof between notebook and eval_runner: notebook writes different structure (e.g. no _mean in avg). | Same paths written by two code paths | Two sources of truth for writing proof; notebook does not use same aggregation as eval_runner. |

### 3.2 Scoring and postprocessing

| # | Problem | Where it shows | Likely cause |
|---|--------|----------------|---------------|
| P4 | SROIE entity scores very low (e.g. 0/4 or 1/4) even when OCR text clearly contains total or date. | Per-sample `metrics.entity_match` | Normalization (comma vs dot for total, date format) or entity extraction heuristics failing; or address never extracted so 1/3 or 1/4 at best. |
| P5 | FUNSD word recall very low (e.g. &lt;0.1) despite prediction containing many GT words. | Per-sample `metrics.word_recall` | Tokenization mismatch (pred is one string, GT is word list); substring or fuzzy matching not applied, or applied too strictly. |
| P6 | Address (SROIE) almost never matches. | Per-sample details / entity_matched | No robust address extraction from raw OCR; heuristics in `extract_sroie_entities_from_text` (longest segment 25–200 chars) may not capture real address lines. |
| P7 | Very short predictions (e.g. 25–50 chars) for some samples; scores stay low. | `prediction` length vs long `ground_truth` | OCR or detection failed or returned minimal text; scoring is correct but reflects missing content (upstream OCR quality). |

### 3.3 Engine and pipeline

| # | Problem | Where it shows | Likely cause |
|---|--------|----------------|---------------|
| P8 | Default OCR eval uses only PaddleOCR; Tesseract preprocessing (Otsu, deskew, 300 DPI) and table reconstruction never run in eval. | eval_runner always calls `force_paddleocr=True` unless `OCR_EVAL_USE_TESSERACT=1` | By design for comparable PaddleOCR scores; but if PaddleOCR fails or is unavailable, fallback uses Tesseract—then preprocessing is used. |
| P9 | PaddleOCR fails to load (e.g. on Windows or restricted env); eval may then use Tesseract fallback or fail. | Logs / "PaddleOCR not available" / empty or low scores | Version pin (2.6.1.3 / 2.5.2), first-run model download timeout, or import/init errors. Pre-download script exists but may not have been run. |
| P10 | Table structure is not reconstructed in the eval pipeline; tables come out as flat text. | N/A (no table output in proof) | `recognize_with_table()` exists but is not invoked by eval_runner or HybridOCR; table reconstruction is opt-in elsewhere. |
| P11 | `low_confidence_words` (Tesseract) are not persisted in proof. | Proof rows have no `low_confidence_words` | Eval_runner can pass them in metadata from HybridOCR, but proof row schema may not write them to `*_samples.json` for review. |

### 3.4 Data and adapters

| # | Problem | Where it shows | Likely cause |
|---|--------|----------------|---------------|
| P12 | OCR datasets (SROIE, FUNSD) may be loaded from HuggingFace when local parquet is missing; run can be slow or differ from local. | Adapters in `eval_dataset_adapters.py` | `load_split` uses HF when local path/split has no parquet; eval_runner can run `data/generate_pq_first_5_rows.py --category ocr` to fill missing parquet. |
| P13 | Ground truth shape differs by dataset (SROIE: dict, FUNSD: list); any code that assumes one shape can break. | Adapters / compute_ocr_metrics | Handled in `compute_ocr_metrics` by dataset_name; but callers must pass correct ground_truth type. |

---

## 4. Data flow (where to look)

- **eval_runner.py:** For `category == "ocr"`, loads samples via adapter, calls `run_model(..., category="ocr")` → HybridOCR `process_document(image, force_paddleocr=...)` → gets `answer` (text) and `metadata`. Then calls `compute_ocr_metrics(pred_text, ground_truth, dataset_name)` to get `metric_row`. Builds per-sample row with `ground_truth`, `prediction`, `metrics`; appends to list; writes `*_samples.json`. Then for each split, reads back `*_samples.json`, builds `split_metric_rows` from `row["metrics"]`, calls `aggregate_metrics(split_metric_rows)`, sets `split_avg["sample_count"]`, writes `*_avg.json`. So **eval_runner does write _mean** if it is the one that wrote the proof. If the notebook wrote the proof last, **avg files will only have sample_count**.
- **Notebook:** Builds results with `sroie_entity_score` / `funsd_word_recall` (which call `eval_postprocessing_utils`), then `write_ocr_proof(dataset_name, split, results)` which writes `split_avg = {"sample_count": sample_count}` only—**no aggregation of metrics**.
- **refresh_category_weighted_avg_from_files:** Reads split `*_avg.json` files and expects keys ending in `_mean` to compute weighted dataset/category metrics. If those keys are missing, OCR contributes nothing to category/eval_summary.

---

## 5. Files and code locations (for fixes)

| Concern | File(s) / location |
|--------|---------------------|
| Per-sample OCR metrics | `eval_postprocessing_utils.compute_ocr_metrics`, `eval_runner.py` (category ocr, metric_row) |
| Split avg content | `eval_runner.py` (aggregate_metrics, then write to `*_avg.json`); notebook `write_ocr_proof` (only sample_count) |
| OCR engine choice | `eval_runner.run_model` (force_paddleocr, OCR_EVAL_USE_TESSERACT) |
| SROIE entity extraction / matching | `eval_postprocessing_utils.extract_sroie_entities_from_text`, `sroie_entity_match_improved`, `normalize_sroie_value` |
| FUNSD word recall | `eval_postprocessing_utils.funsd_word_recall_improved` (substring, normalize, fuzzy) |
| Proof layout | `data/proof/ocr/<dataset>/<split>/*.json` |
| Lessons and design | `data/proof/OCR_LESSONS.md` |

---

## 6. Suggested diagnostic steps (for the diagnosing agent)

1. **Inspect existing proof:** Read `data/proof/ocr/sroie/train/sroie_train_avg.json` and `data/proof/ocr/funsd/train/funsd_train_avg.json`. Confirm whether they contain any `*_mean` keys or only `sample_count`.
2. **Trace who wrote proof last:** If avg files have only `sample_count`, the notebook’s `write_ocr_proof` was likely the last writer. Fix: have the notebook compute and write the same split_avg as eval_runner (aggregate per-sample metrics and write `entity_match_mean`, `word_recall_mean`, etc.), or re-run OCR eval with eval_runner and overwrite proof.
3. **Check per-sample metrics:** Open a few `*_samples.json` rows. Confirm `metrics` has `entity_match`/`entity_matched`/`entity_total` (SROIE) or `word_recall`/`words_matched`/`words_gt` (FUNSD). If present, aggregation logic in eval_runner should produce _mean; if avg still lacks _mean, the writer was the notebook.
4. **SROIE low scores:** For a sample with low entity_match but prediction containing e.g. the total, run `extract_sroie_entities_from_text(prediction)` and `normalize_sroie_value("total", gt_total)` and compare. Check if address is in GT and whether extraction/heuristic ever fills address.
5. **FUNSD low recall:** For a sample with low word_recall, check if GT words appear as substrings in prediction (normalized). If yes, substring matching may be broken or not applied; if no (typos), fuzzy threshold may be too strict.
6. **Engine and env:** Confirm whether PaddleOCR loads (e.g. run `scripts/pre_download_paddleocr_models.py`). If not, eval falls back to Tesseract; then check that Tesseract path uses preprocessing and that results are still written to the same proof paths.

---

## 7. Recommended fix directions (high level)

- **P1/P2/P3:** Unify proof writing so that (a) split avg always includes `*_mean` from per-sample metrics (entity_match_mean, word_recall_mean, etc.), and (b) either the notebook calls the same aggregation as eval_runner or only eval_runner writes proof for OCR. Ensure `refresh_category_weighted_avg_from_files` and eval_summary see OCR metrics.
- **P4–P7:** Revisit postprocessing (normalization, entity extraction, address heuristic, substring/fuzzy) using worst-offending samples from proof; optionally add minimum prediction-length flag for “low confidence” samples and document in OCR_LESSONS.
- **P8–P11:** Document when Tesseract vs PaddleOCR is used; optionally persist `low_confidence_words` in proof for human review; keep table reconstruction as opt-in.
- **P12–P13:** Ensure parquet generation runs when needed and that adapters and compute_ocr_metrics receive the correct ground_truth type per dataset.

---

*End of outline. Use this together with `data/proof/OCR_LESSONS.md` and the code paths above to diagnose and implement fixes.*
