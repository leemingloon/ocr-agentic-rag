#!/usr/bin/env python3
"""
Compute per-dataset (Financial PhraseBank vs FiQA) val/test F1 macro for sentiment models.
Requires: data/credit_risk_sentiment/processed/ (run 03_sentiment_FP_FiQA_feature_engineering.ipynb first).
Optional: models/sentiment/finbert_tuned_v1 for FinBERT fine-tuned predictions.
Usage: from repo root, python scripts/sentiment_per_dataset_metrics.py
"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    processed = ROOT / "data" / "credit_risk_sentiment" / "processed"
    if not (processed / "val.parquet").exists():
        print("Run 03_sentiment_FP_FiQA_feature_engineering.ipynb first.")
        return
    import pandas as pd
    from sklearn.metrics import f1_score

    val_df = pd.read_parquet(processed / "val.parquet")
    test_df = pd.read_parquet(processed / "test.parquet")
    if "source" not in val_df.columns:
        print("No 'source' column in processed data. Re-run notebook 03 (it now adds source).")
        return

    out = {
        "n_val": len(val_df),
        "n_test": len(test_df),
        "val_per_source": {},
        "test_per_source": {},
    }
    for name, df in [("val", val_df), ("test", test_df)]:
        for src in ["FinancialPhraseBank", "FiQA"]:
            mask = df["source"] == src
            n = int(mask.sum())
            out[f"{name}_per_source"][src] = {"n": n}
    # If you have predictions (e.g. from FinBERT), add them and compute F1 per source:
    # pred_val = ...  # shape (len(val_df),) or (len(val_df), 3) for logits
    # for src in ["FinancialPhraseBank", "FiQA"]:
    #     m = val_df["source"] == src
    #     out["val_per_source"][src]["f1_macro"] = float(f1_score(val_df.loc[m, "label"], pred_val[m], average="macro"))
    print(json.dumps(out, indent=2))
    proof_dir = ROOT / "data" / "proof" / "credit_risk_sentiment"
    proof_dir.mkdir(parents=True, exist_ok=True)
    with open(proof_dir / "per_source_counts.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", proof_dir / "per_source_counts.json")

if __name__ == "__main__":
    main()
