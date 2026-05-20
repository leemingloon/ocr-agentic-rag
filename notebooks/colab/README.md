# Colab — OCR FUNSD + SROIE eval

Open **`ocr_funsd_sroie_eval_colab.ipynb`** in [Google Colab](https://colab.research.google.com/):

- Upload from GitHub: `notebooks/colab/ocr_funsd_sroie_eval_colab.ipynb` in your repo
- Or: File → Upload notebook

## Steps

1. **Runtime → Change runtime type → T4 GPU**
2. Edit `rec_batch_num` / `ocr_force_reeval` in cell 1
3. **Runtime → Run all**
4. Download `ocr_proof_bundle.zip` from the last cells

## Merge results locally

Unzip the downloaded zip, then from repo root:

```bash
OCR_PROOF_SRC=/path/to/ocr_colab_outputs python scripts/sync_kaggle_ocr_proof.py
```

(`ocr_colab_outputs` must contain `ocr_proof_bundle/` and optionally `ocr_eval_summary.json`.)

## vs Kaggle

Colab clones GitHub directly — no `dataset_sources` / invalid dataset slug. Push your branch to GitHub before running if you need the latest local changes.
