# Evaluation Results

**Document Intelligence + Credit Risk Platform**  
**Evaluator:** Lee Ming Loon

---

## Where results live

Model evaluation is run via **`eval_runner.py`**. All proof outputs are written under **`data/proof/`**.

Demos and notebooks in **`notebooks/`** load evaluation data from **`data/proof/`**. Any metrics they show come from those artifacts only.

---

## What is currently under `data/proof/`

- **Vision:** ChartQA, DocVQA, InfographicsVQA, MMMU (Accounting, Economics, Finance, Math) — per-sample and weighted-avg JSON plus predictions.
- **RAG:** FinQA (per-sample and weighted-avg), plus adversarial RAG samples.

Scores for these categories are **TBD** and will be filled in as evaluation is completed and reviewed. Do not treat numbers in this repo as final until they are explicitly documented as such.

---

## Categories not yet backed by proof

The following do **not** currently have evaluation artifacts under `data/proof/`; figures for them are placeholders until proof runs are added:

- **OCR:** OmniDocBench, SROIE, FUNSD, DUDE, DocVQA (OCR track), InfographicsVQA (OCR)
- **Other RAG:** HotpotQA, TAT-QA, BIRD-SQL (only FinQA and adversarial have proof so far)
- **Credit Risk:** PD (Lending Club), NLP sentiment (FiQA, Financial Phrase Bank), Risk Memo (FinanceBench, ECTSum), drift, counterfactual
- **System tests:** Robustness, bias & fairness, adversarial, load

Methodology (metrics, datasets, aggregation) is implemented in `eval_runner.py`, `eval_dataset_adapters.py`, and `eval_postprocess_utils.py`. When new proof is generated, this document will be updated and scores added.

---

**Last updated:** February 2026
