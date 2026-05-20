# OCR FUNSD + SROIE — Kaggle kernel

Runs full OCR proof evaluation (`eval_runner.py`) on Kaggle GPU and writes pullable artefacts under `kaggle_outputs/`.

## Layout

```
ocr_funsd_sroie_eval_kaggle/
  ocr_funsd_sroie_eval_kaggle.ipynb   # notebook
  kernel-metadata.json                # kaggle kernels push metadata
  kaggle_outputs/                     # kaggle kernels output -p ... (gitignored contents)
  README.md
```

## Prerequisites

1. [Kaggle API](https://github.com/Kaggle/kaggle-api) installed and `~/.kaggle/kaggle.json` configured.
2. **Internet on** (default): notebook clones `https://github.com/leemingloon/ocr-agentic-rag.git` into `/kaggle/working/ocr-agentic-rag` if no dataset is attached. Optional: attach a Kaggle dataset with parquet under `data/ocr/` to avoid HF download.
3. Kernel settings: **GPU on**, **Internet on** (for `pip install paddlepaddle-gpu`).

## Push and run

From repo root:

```bash
kaggle kernels push -p notebooks/kaggle-kernels/ocr_funsd_sroie_eval_kaggle
```

Start the kernel on Kaggle (or `kaggle kernels run leemingloon/ocr-funsd-sroie-eval-kaggle` if configured).

## Pull outputs (agent loop)

```bash
kaggle kernels output leemingloon/ocr-funsd-sroie-eval-kaggle \
  -p notebooks/kaggle-kernels/ocr_funsd_sroie_eval_kaggle/kaggle_outputs
```

Then merge into local proof:

```bash
python scripts/sync_kaggle_ocr_proof.py
```

## Outputs

| File / folder | Purpose |
|---------------|---------|
| `ocr_eval_summary.json` | Split-level `*_avg.json` metrics (quick agent read) |
| `ocr_proof_bundle/` | Full `data/proof/ocr` tree (samples + avg JSON) |
| `ocr_eval_run.log` | eval_runner stdout |

## Environment variables (notebook)

| Variable | Default | Meaning |
|----------|---------|---------|
| `OCR_USE_GPU` | `1` on Kaggle | Paddle GPU det+rec |
| `OCR_SKIP_TESSERACT_ENSEMBLE` | `1` on Kaggle | Skip CPU Tesseract pass (~2x faster) |
| `OCR_FAST` | `1` on Kaggle | Smaller upscale, optional no angle cls |
| `OCR_PREFETCH` | `1` | Threaded sample prefetch during OCR |
| `OCR_REC_BATCH_NUM` | `16` | Paddle `rec_batch_num` (GPU VRAM; try 24–32 on T4) |
| `OCR_PADDLE_MIN_SIDE` | `720` | Min image side before det (higher = more VRAM) |
| `OCR_FORCE_REEVAL` | `0` | Set `1` to re-OCR every sample |
| `KAGGLE_REPO_DATASET` | `leemingloon/ocr-agentic-rag` | Dataset slug (mounted as `leemingloon-ocr-agentic-rag`) |

## Local dry-run

Open the notebook from repo root context or run cells locally: `OUT_DIR` becomes `kaggle_outputs/` under this folder (same as `paper_lstm_stream_loo`).
