#!/usr/bin/env python3
"""Print credit-risk eval metrics from data/proof/eval_summary.json for resume fill-in.
Run from repo root. Invoked by run_credit_risk_eval_overnight.sh / .bat after evals."""
import json
from pathlib import Path

def main():
    p = Path(__file__).resolve().parent.parent / "data" / "proof" / "eval_summary.json"
    if not p.exists():
        print("Run evals first; then eval_summary.json will exist.")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    overview = d.get("overview", {})
    labels = {
        "credit_risk_pd": "PD (XGBoost, LendingClub)",
        "credit_risk_pd_quantum": "PD (Quantum VQC, LendingClub)",
        "credit_risk_sentiment": "Sentiment (classical pkl, FinancialPhraseBank + FiQA)",
        "credit_risk_sentiment_finbert": "Sentiment (FinBERT pretrained, FinancialPhraseBank + FiQA)",
    }
    for cat, label in labels.items():
        if cat not in overview:
            continue
        o = overview[cat]
        n = o.get("sample_count_total", 0)
        mb = o.get("metrics_breakdown") or {}
        vals = {k: v.get("value") for k, v in mb.items() if v.get("value") is not None}
        if not vals:
            print(f"  {label}: n={n} (no metrics yet)")
        else:
            parts = [
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in sorted(vals.items())
            ]
            print(f"  {label}: n={n} -> " + ", ".join(parts))
    print("(Source: data/proof/eval_summary.json)")


if __name__ == "__main__":
    main()
