# OCR evaluation (SROIE, FUNSD)

## Quick runs

To avoid long runs, limit samples:

```bash
python eval_runner.py --category ocr --dataset SROIE --max_split 5 --max_category 20
python eval_runner.py --category ocr --dataset FUNSD --max_split 5 --max_category 20
```

## SROIE: document entities

SROIE is receipt OCR with **document_entities**: `company`, `date`, `address`, `total`. Metrics are `entity_match` (fraction of these four fields matched).

### Improvements in this repo

1. **Ensemble OCR for SROIE**  
   When `dataset_name == "SROIE"`, the eval uses HybridOCR with `use_ensemble_for_accuracy=True`: both Tesseract and PaddleOCR run and their text is merged. This improves coverage of company names and totals.

2. **Tesseract whitelist**  
   `&` is allowed so company names like "INDAH GIFT & HOME DECO" are not stripped.

3. **Entity extraction** (`eval_postprocessing_utils.extract_sroie_entities_from_text`)  
   - **Date**: Multiple patterns (dd/mm/yyyy, dd-mm-yy, d/m/yyyy). Normalized to dd/mm/yyyy for comparison.  
   - **Total**: Prefer value after "total"/"RM"/"TOTAL"; else largest decimal in the text.  
   - **Company**: SDN BHD / BND; fallback to segments containing ENTERPRISE, TRADING, DECO, GIFT, etc.  
   - **Address**: Longest line with JALAN/NO./STREET and digits (20–250 chars).

4. **Confusion correction**  
   Applied before entity extraction: O→0, I/l→1, S→5 in numeric/date contexts; 0→O, 1→l, 5→S in word contexts.

### Layout-aware and entity-specific logic (implemented)

- **Layout-aware regions** (`ocr_pipeline.layout_regions`): SROIE uses header (top 40% of lines) for company/address/date and footer (bottom 35%) for total. Entity extraction runs per region then merges; fallback to full text. FUNSD has header (top 25%) and body for future use.
- **Entity-specific regex (SROIE)**: Total is matched with patterns for `RM 9.00`, `R M 9.00`, `Total : 9.00`, etc., then largest decimal. Date supports d/m/yyyy and dd-mm-yy. Company: SDN BHD / BND plus ENTERPRISE, TRADING, DECO, GIFT, etc.
- **Entity-specific normalization (FUNSD)**: GT words are normalized with `_normalize_word_funsd`: strip trailing `:` and outer `( )` so form labels like "Date:" match OCR "Date". Word recall uses normalized pred tokens (lowercased) for matching.

### Further ideas

- **Vision OCR fallback**: Enable `use_vision_augmentation=True` for low-confidence receipts.  
- **PaddleOCR-only for receipts**: Set `OCR_EVAL_USE_TESSERACT=0` (default) so full PaddleOCR (or ensemble) is used.

---

## FUNSD: ground_truth (token_labels) explained

In `funsd_test_samples.json` (and train), each sample has **`ground_truth.token_labels`**: a list of **integers** with no other text. Those integers are **NER (Named Entity Recognition) tag IDs**: one number per **word** in the document, in **reading order**.

### What each integer means

| ID | Label       | Meaning                          |
|----|-------------|----------------------------------|
| 0  | O           | Outside (not a field)            |
| 1  | B-HEADER    | Begin of HEADER                  |
| 2  | I-HEADER    | Inside HEADER                    |
| 3  | B-QUESTION  | Begin of QUESTION (form label)   |
| 4  | I-QUESTION  | Inside QUESTION                  |
| 5  | B-ANSWER    | Begin of ANSWER (form value)     |
| 6  | I-ANSWER    | Inside ANSWER                    |

So the list is **aligned 1:1 with the document words**: `token_labels[i]` is the label for the i-th word. For example, if the document has words `["INVOICE", "Date", ":", "01/01/2020", ...]`, then typically `0→O`, `1→B-HEADER` (INVOICE), `3→B-QUESTION` (Date), etc.

### How to get “ground truth” intuitively

- **At eval time**: The dataset row has both `words` (list of strings) and `ner_tags` (same list of integers). The eval builds the **GT words we care about** as: every word whose label is not `O` (i.e. HEADER, QUESTION, or ANSWER). So **GT words** = `[words[i] for i in range(len(words)) if token_labels[i] != 0]`. The metric **word_recall** is: how many of those GT words appear in the OCR prediction (with normalization and fuzzy matching).
- **From the proof JSON only**: The proof files do **not** store the `words` list—only `token_labels`. So from the JSON alone you cannot reconstruct the exact GT words; you only see which positions are HEADER (1,2), QUESTION (3,4), or ANSWER (5,6). To see the actual words, you need the original dataset (e.g. HuggingFace `nielsr/funsd`) where each row has `words` + `ner_tags` (same as `token_labels`).

### Short summary

- **token_labels** = one integer per word in reading order.  
- **0** = ignore for recall; **1,2,3,4,5,6** = HEADER / QUESTION / ANSWER.  
- **GT words** for the metric = words at positions where `token_labels[i] != 0`.

---

## FUNSD: techniques from crcresearch/FUNSD

We align with [crcresearch/FUNSD](https://github.com/crcresearch/FUNSD) and the original FUNSD paper (Jaume et al., 2019) as follows.

### 1. Semantic entities (header / question / answer)

FUNSD is **entity-centric**: each form field is a **semantic entity** (a group of words that belong together). We implement:

- **Entity extraction from GT**: `get_funsd_entities_from_sample(sample)` groups consecutive B-I tokens into entities with `label` in `{"header", "question", "answer"}` and `text` = concatenated words.
- **Entity-level recall**: `entity_recall` = fraction of GT entities whose text appears in the OCR prediction (substring or fuzzy). Reported alongside `word_recall` in FUNSD metrics (`entity_recall`, `entity_matched`, `entity_total`).

### 2. Spatial layout (bbox-based regions)

FUNSD uses **bounding boxes** `[left, top, right, bottom]` per word. We provide:

- **`assign_bbox_to_region(bbox, image_height, dataset_name)`**: assigns a bbox to header/body (FUNSD) or header/body/footer (SROIE) by vertical position (y centre / image height).
- **`split_words_by_region(words, bboxes, image_height, dataset_name)`**: groups words by region for downstream use (e.g. when OCR returns word-level bboxes).

So when you have word-level or line-level bboxes (e.g. from Tesseract/PaddleOCR), you can assign text to header vs body for form understanding.

### 3. Preprocessing for noisy scanned documents

FUNSD comes from **noisy scanned forms** (e.g. 100 dpi, low quality). In this repo:

- **Strong preprocessing** is used for Tesseract and PaddleOCR: Otsu binarisation, deskew, 300 DPI normalisation (`ocr_pipeline.preprocessing.document_preprocessor.preprocess_for_ocr`). This matches the need for form understanding in low-quality scans.

### 4. References and variants

- **Original**: Jaume et al. (2019), “FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents.”
- **FUNSD Revised** (Vu et al., 2020): corrected labels; you can use it as an alternative GT source if you load that dataset.
- **FUNSD+**: larger, revised dataset; can be added as a second benchmark.
