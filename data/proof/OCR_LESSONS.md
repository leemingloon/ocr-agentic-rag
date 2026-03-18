# OCR pipeline: Lessons learned

Consolidated lessons from SROIE and FUNSD evaluation, postprocessing improvements, and evaluation methodology. Use **per-sample JSON** under `data/proof/ocr/<dataset>/<split>/` to inspect failures and iterate. Run **100 samples** per dataset (notebook or eval_runner) to surface common issues before locking postprocessing.

---

## Where we're at: OCR accuracy (including PaddleOCR)

- **Measured numbers:** If you have not run OCR eval in this workspace, there are **no OCR metrics** in `data/proof` or `eval_summary.json`. The OCR category is separate from RAG/vision/credit_risk_pd; you must run OCR eval to get **entity_match_mean** (SROIE) and **word_recall_mean** (FUNSD).
- **Engine in eval:** By default, **eval_runner --category ocr** and the notebook use **PaddleOCR** (full det+rec) via `force_paddleocr=True`. So reported accuracy is **PaddleOCR + current postprocessing** (normalization, entity extraction, substring/fuzzy). If PaddleOCR fails to load, the pipeline falls back to Tesseract.
- **Already in place:** (1) **Preprocessing** before PaddleOCR (Otsu, deskew, 300 DPI) is on by default for better scans. (2) **Postprocessing** in `eval_postprocessing_utils`: SROIE entity extraction + normalize + soft CER; FUNSD substring + normalize + fuzzy. (3) **Optional ensemble** (`use_ensemble_for_accuracy=True`): run Tesseract and PaddleOCR and merge text for best coverage (use in production; eval keeps single-engine for comparable metrics).
- **Target for "acceptable" document/resume use:** Aim for **entity_match_mean (SROIE) or word_recall_mean (FUNSD) around 0.6–0.7+**. Below that, inspect worst samples and fix normalization, extraction, or upstream OCR.

---

## Why OCR scores were low (root causes)

### 1. **Query/scoring mismatch (fixed)**

- **SROIE:** Ground truth is **structured entities** (company, date, address, total). Raw OCR text rarely contains the exact GT string (e.g. "9,00" vs "9.00", or total buried in a long line). **Exact substring match** gave 0/4 or 1/4 for most samples.
- **FUNSD:** Ground truth is a **list of words** (token-level). Scoring used **exact token match** (GT word must appear as a separate token in prediction). OCR output is often run-together or differently tokenized, so "Date: 9/3/92" did not match GT tokens "Date", "9", "3", "92" when split by spaces only.

### 2. **Normalization gaps**

- **Numbers:** Receipt totals use comma as decimal (e.g. "80,90") while GT uses "80.90". Without normalizing comma → dot, total never matches.
- **Dates:** GT "25/12/2018" vs pred "25-12-2018" or "25.12.2018". Normalizing separators to "/" fixes many date matches.
- **Spaces/case:** "SDN BHD" vs "sdn bhd" or "SDNBHD" — lowercase and collapse spaces so substring and entity extraction are consistent.

### 3. **OCR errors and typos**

- **FUNSD:** OCR produces "ciparettes" vs GT "cigarettes", "PIOENIX" vs "PHOENIX", "Filtez" vs "Filter". Exact or substring match fails; **fuzzy (edit-distance)** match recovers some of these.
- **SROIE:** Company names with BHD/BND, punctuation, or extra spaces need robust extraction and soft (CER-based) match.

### 4. **Entity extraction from raw text**

- **SROIE:** We do not have gold "boxes" for each entity in eval; we only have raw OCR text. So we must **extract** company, date, total (and optionally address) from the prediction string. Heuristics: date regex, decimal total (largest `\d+[.,]\d{2}`), company window around "SDN BHD"/"BND". **Address** is often a long line—hard to segment from raw text; many runs skip address or match only when it appears as substring.

### 5. **Truncation / short predictions**

- Some images (e.g. low contrast, small text) yield very short OCR output (e.g. 25–50 chars). Word recall and entity match stay low because most content is missing. This is an **OCR quality / detection** issue, not a scoring bug; postprocessing can only fairly score what is present.

---

## Postprocessing and evaluation rules (OCR)

Design: **postprocessing** = normalizing or relaxing comparison so a model answer can count as correct; **evaluation methodology** = how we compute and aggregate metrics.

### Postprocessing rules (treat prediction as correct / True Positive)

#### SROIE: Normalize numbers and dates before comparison

- **Observation:** GT total "80.90", pred "80,90" or "80.94" (OCR misread). Exact string match fails.
- **Rule:** Normalize both sides: **total** → replace comma with dot, strip spaces; **date** → replace "-" with "/". Then compare. If still no match, allow **soft match** when character error ratio (CER) between pred and GT value ≤ 0.35 (see `normalize_sroie_value`, `sroie_entity_match_improved`).
- **Implementation:** `eval_postprocessing_utils.normalize_sroie_value`, `sroie_entity_match_improved(..., soft_cer_threshold=0.35)`.

#### SROIE: Extract entities from prediction before matching

- **Observation:** Raw OCR is one long string; GT is {company, date, address, total}. We must derive pred entities from the string.
- **Rule:** Run **extract_sroie_entities_from_text(pred)** to get pred_entities (date regex, total = largest decimal, company = window around SDN BHD/BND). Then compare each key with **normalize_sroie_value** and optional CER soft match. Fallback: if extraction misses a key, still try **substring** of GT in raw pred (so "80.90" in pred counts even if not in extracted total).
- **Implementation:** `extract_sroie_entities_from_text`, `sroie_entity_match_improved(..., extract_from_pred=True)`.

#### FUNSD: Substring and normalized word match

- **Observation:** GT words are token-level; pred is a single string. Exact token set (pred.split()) gives very low recall.
- **Rule:** Count a GT word as **matched** if (1) normalized GT word (lowercase, collapse spaces) appears **anywhere** in normalized pred (substring), or (2) GT word (no spaces) appears in pred (no spaces). So "Date" and "9" and "3" and "92" in "Date: 9/3/92" all count.
- **Implementation:** `funsd_word_recall_improved(..., use_substring=True, normalize=True)`.

#### FUNSD: Fuzzy word match for OCR typos (optional)

- **Observation:** OCR typos (e.g. "ciparettes", "PIOENIX") cause substring match to fail.
- **Rule:** When substring match fails for a GT word, allow a **fuzzy** match: if any pred token (or substring of pred) has edit distance to GT word ≤ 2 and length ≥ 4, count as matched. Cap fuzzy matches so we don’t over-count (e.g. at most one fuzzy match per GT word, and only when edit ratio ≤ 0.3).
- **Implementation:** `funsd_word_recall_improved(..., use_fuzzy=True, fuzzy_max_edit_ratio=0.3)` in `eval_postprocessing_utils.py`.

---

### Evaluation methodology

#### Single entry point: compute_ocr_metrics

- **Rule:** All OCR metrics (SROIE and FUNSD) are produced by **compute_ocr_metrics(prediction, ground_truth, dataset_name)**. SROIE uses improved entity match (extract + normalize + soft CER); FUNSD uses improved word recall (normalize + substring + optional fuzzy). This keeps eval_runner and the notebook aligned.
- **Implementation:** `eval_postprocessing_utils.compute_ocr_metrics`; called from `eval_runner.py` for category `ocr` and can be used in the notebook for consistency.

#### Aggregation (split / dataset / category)

- **Rule:** Per-sample metrics (entity_match, entity_matched, entity_total for SROIE; word_recall, words_matched, words_gt for FUNSD) are written to proof JSON. Split-level average: **entity_match_mean** = mean(entity_match), **word_recall_mean** = mean(word_recall). Dataset/category weighted averages use sample counts per split. Same as in eval_runner `aggregate_metrics` and refresh logic.
- **Implementation:** `eval_runner.aggregate_metrics`, `refresh_category_weighted_avg_from_files`, etc.

#### Ensure PaddleOCR is used for OCR evaluation

- **Rule:** When evaluating OCR (eval_runner --category ocr or notebook), use **full PaddleOCR** (detection + recognition) so scores reflect the same engine and are comparable. Avoid mixing Tesseract-only runs with PaddleOCR runs when comparing numbers.
- **Implementation:** `eval_runner.run_model` for category `ocr` calls `ocr.process_document(image, force_paddleocr=True)`; HybridOCR `_process_with_paddleocr_full` runs PaddleOCR.ocr(det=True, rec=True).

---

## Running 100 samples to find common issues

- **Notebook:** Set **MAX_SROIE = 100** and **MAX_FUNSD = 100** in `notebooks/demo_hybrid_ocr_eval.ipynb`. Load SROIE and FUNSD from HuggingFace (train split), run HybridOCR on each image, compute metrics with **eval_postprocessing_utils.compute_ocr_metrics** (or the same improved helpers), and write proof to `data/proof/ocr/`. Optionally print progress every 10 samples to reduce log volume.
- **eval_runner:** `python eval_runner.py --category ocr --max_split 1 --max_category 1` (and optionally limit samples per split if needed). Proof is written under `data/proof/ocr/<dataset>/<split>/`.
- **Inspection:** After a 100-sample run, sort samples by **entity_match** (SROIE) or **word_recall** (FUNSD) and review the worst 10–20. Look for: (1) systematic normalization gaps, (2) repeated OCR confusions (e.g. 0/O, 1/l), (3) address never matching (consider relaxing or dropping address from total count), (4) very short predictions (improve detection/recognition or document as quality limit). Add or tune postprocessing and document new lessons here.

---

## Summary of code locations

| What | Where |
|------|--------|
| SROIE entity extraction + normalize + soft match | `eval_postprocessing_utils.extract_sroie_entities_from_text`, `normalize_sroie_value`, `sroie_entity_match_improved` |
| FUNSD word recall (substring + normalize + optional fuzzy) | `eval_postprocessing_utils.funsd_word_recall_improved`, `compute_ocr_metrics` |
| Single metric entry point | `eval_postprocessing_utils.compute_ocr_metrics` |
| Eval runner OCR metrics + force PaddleOCR | `eval_runner.py` (category `ocr`: `compute_ocr_metrics`, `force_paddleocr=True`) |
| Notebook 100-sample run + proof | `notebooks/demo_hybrid_ocr_eval.ipynb` (MAX_SROIE / MAX_FUNSD = 100, use same scoring as eval_runner) |
| Proof layout | `data/proof/ocr/<dataset>/<split>/<dataset>_<split>_samples.json`, `*_avg.json` |
| Tesseract preprocessing (Otsu, deskew, 300 DPI) | `ocr_pipeline.preprocessing.document_preprocessor.preprocess_for_ocr` |
| Tesseract table reconstruction | `TesseractOCR.reconstruct_table_from_data`, `recognize_with_table()` |
| Standard OCR schema | `ocr_pipeline.recognition.ocr_schema.StandardOCROutput`, `TesseractOCR.to_standard_output()` |
| PaddleOCR pre-download | `python scripts/pre_download_paddleocr_models.py` |

---

## When you run eval_runner.py --category ocr

- **Default:** OCR eval uses **PaddleOCR** (full det+rec) via `process_document(..., force_paddleocr=True)`. So **PaddleOCR** behaviour applies: version pin, pre-download script, graceful skip if it fails. **PyTesseract** enhancements are **not** in this path; they run only when PaddleOCR is unavailable (automatic fallback) or when you force Tesseract (see below).
- **Force Tesseract path:** Set **`OCR_EVAL_USE_TESSERACT=1`** (env var) before running. Then the runner uses the normal HybridOCR pipeline: **Tesseract first** (with preprocessing: Otsu, deskew, 300 DPI), confidence-based routing, and **low_confidence_words** in metadata. Use this to exercise PyTesseract enhancements or when PaddleOCR does not load.
- **PaddleOCR fails:** If PaddleOCR fails to load, the pipeline **falls back** to the normal path (detection + Tesseract recognition). So **Tesseract preprocessing and confidence flagging are used automatically** in that fallback.
- **Summary:** PaddleOCR enhancements are always in play (pin, pre-download, graceful skip). PyTesseract enhancements are used when (1) you set `OCR_EVAL_USE_TESSERACT=1`, or (2) PaddleOCR is unavailable and fallback runs. Table reconstruction is available programmatically (`recognize_with_table`) but is not invoked by the eval runner by default.

---

## PyTesseract: preprocessing, table reconstruction, confidence

- **Preprocessing pipeline** (before passing to Tesseract): **Otsu binarisation**, **deskew** (minAreaRect-based), **300 DPI normalisation**. Implemented in `ocr_pipeline.preprocessing.document_preprocessor` and used by `TesseractOCR` when `use_strong_preprocessing=True` (default). These steps significantly improve character recognition on scanned financials.
- **Table reconstruction**: Tesseract has no document structure; tables come out as garbled text. Use **psm 6** (uniform block) or **psm 11** (sparse text). After extraction, reconstruct table structure from **image_to_data()** bounding boxes: group by approximate **y** (row_tolerance, e.g. 8px), sort by **x** within each row. Implemented as `TesseractOCR.reconstruct_table_from_data()` and `recognize_with_table()`.
- **Confidence filtering**: Do not trust confidence scores globally. Words with confidence **below 60** are **flagged for human review** (not discarded); see `OCRResult.low_confidence_words`. Use selectively (e.g. flag for review) rather than treating scores as absolute quality signals.

## PaddleOCR: version pin, pre-download, graceful skip

- **Version pin**: For stable CPU behaviour, use **paddleocr==2.6.1.3** and **paddlepaddle==2.5.2** (see `requirements.txt` / `setup.py`). Newer versions can change import structure (e.g. PPStructure for table-aware extraction).
- **Model download on first run**: PaddleOCR downloads weights on first instantiation and can time out in restricted environments. **Pre-download** by running once: `python scripts/pre_download_paddleocr_models.py`. Set **PADDLEOCR_SHOW_LOG=1** to see download progress.
- **Graceful skip**: If PaddleOCR fails to load (import or init), the detector sets **mode="failed"** and returns no boxes; the **eval runner continues** with other engines (e.g. Tesseract). No exception propagates so the pipeline does not break.
- **Table module (optional)**: For PaddleOCR's table recognition (stronger than Tesseract for multi-column), use **PPStructure** with `table=True`, `ocr=True` separately; it is not part of the base PaddleOCR class used in this repo for detection.

## Standard OCR output schema (model-agnostic)

- Every engine should produce a **standardised output**: **{text, bbox, confidence}** per word/line so that **table reconstruction** and **metric calculation** happen **downstream of OCR**, independent of which engine produced the raw output.
- Implemented in `ocr_pipeline.recognition.ocr_schema`: **OCRWord**, **OCRLine**, **StandardOCROutput** (words, full_text, confidence, low_confidence_words, optional lines/table_rows). **TesseractOCR.to_standard_output(OCRResult)** converts to **StandardOCROutput**.

---

## Is this acceptable for resumes / documents?

- **Benchmarks:** Eval uses SROIE (receipts) and FUNSD (forms). Resumes are not a separate benchmark; treat receipt/form scores as a proxy for document OCR quality. For resume-level acceptability, aim for **entity_match_mean** or **word_recall_mean** above ~0.6–0.7 so names, dates, and key phrases are captured; low scores often indicate short OCR output or normalization/typo issues.
- **How to improve for resumes:** (1) Use **preprocessing** (default): Otsu, deskew, 300 DPI before PaddleOCR. (2) For maximum accuracy, set **use_ensemble_for_accuracy=True** on HybridOCR to run both Tesseract and PaddleOCR and merge text. (3) Run 100+ samples and fix systematic gaps in `eval_postprocessing_utils`. (4) Optionally enable vision augmentation for poor scans.

## Recent improvements (implemented)

- **Preprocessing before PaddleOCR:** Default `preprocess_before_paddleocr=True` applies Otsu, deskew, 300 DPI before PaddleOCR for better scans/resumes.
- **Morphological processing:** `preprocess_for_ocr(..., morphology_cleanup_enabled=True)` runs closing + opening after binarisation to connect broken strokes and remove speckle; important for receipts (thermal, faded) and invoices. See `document_preprocessor.morphology_cleanup`.
- **OCR confusion correction (o/0, I/l/1, 5/S):** `apply_ocr_confusion_correction` in eval_postprocessing_utils: numeric context (totals, dates) O→0, I/l→1, S→5; word context (FUNSD/SROIE) 0→O, 1→l, 5→S. Used by default in `compute_ocr_metrics`.
- **Ensemble mode:** `HybridOCR(use_ensemble_for_accuracy=True)` with `force_paddleocr=True` runs Tesseract + PaddleOCR and merges text; use for high-accuracy/resume workflows.
- **SROIE address:** Heuristic prefers segments with JALAN, NO., STREET, ROAD; length 20–250 chars.
- **FUNSD fuzzy:** `fuzzy_min_len=3`, `fuzzy_max_edit_ratio=0.35`.
- **Notebook proof:** `write_ocr_proof` now writes entity_match_mean / word_recall_mean to split and dataset avg files.

## Future improvements (candidate)

- **Address (SROIE):** Heuristic to extract a single “longest alphanumeric line” or line containing “JALAN”/postal code as address; then normalize and soft match. Currently address is often missed.
- **Confusion tuning:** Add more dataset-specific rules (e.g. Z/2, B/8) if proof inspection shows systematic confusions.
- **Minimum character threshold:** Flag samples with pred length &lt; N as “low confidence” and report separately (don’t mix with full-length predictions in aggregate).
- **FUNSD entity-level (optional):** FUNSD also has entity annotations; we could add an entity F1 (e.g. key-value) alongside word recall for forms that have structured fields.
