#!/usr/bin/env bash
# Run from repo root. Evaluates all credit_risk PD and sentiment models on their datasets (ground truth only).
# Uses CPU only; no GPU or API keys required. Suitable for overnight on e.g. i5-11500 16GB.

set -e
cd "$(dirname "$0")/.."

echo "=== 1/4 credit_risk_PD (LendingClub, classical XGBoost) ==="
python eval_runner.py --category credit_risk_PD

echo ""
echo "=== 2/4 credit_risk_PD_quantum (LendingClub, VQC) ==="
python eval_runner.py --category credit_risk_PD_quantum

echo ""
echo "=== 3/4 credit_risk_sentiment (FinancialPhraseBank + FiQA, classical pkl) ==="
python eval_runner.py --category credit_risk_sentiment

echo ""
echo "=== 4/4 credit_risk_sentiment_finbert (FinancialPhraseBank + FiQA, FinBERT) ==="
python eval_runner.py --category credit_risk_sentiment_finbert

echo ""
echo "Done. Check data/proof/credit_risk_pd/ data/proof/credit_risk_pd_quantum/ data/proof/credit_risk_sentiment/ data/proof/credit_risk_sentiment_finbert/"
