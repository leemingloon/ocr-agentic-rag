@echo off
REM Evaluates all credit_risk PD and sentiment models on their datasets (ground truth only).
REM Uses CPU only; no GPU or API keys required. Suitable for overnight on e.g. i5-11500 16GB.
cd /d "%~dp0.."

echo === 1/4 credit_risk_PD (LendingClub, classical XGBoost) ===
python eval_runner.py --category credit_risk_PD
if errorlevel 1 echo PD eval had errors.

echo.
echo === 2/4 credit_risk_PD_quantum (LendingClub, VQC) ===
python eval_runner.py --category credit_risk_PD_quantum
if errorlevel 1 echo PD quantum eval had errors.

echo.
echo === 3/4 credit_risk_sentiment (FinancialPhraseBank + FiQA, classical pkl) ===
python eval_runner.py --category credit_risk_sentiment
if errorlevel 1 echo Sentiment classical eval had errors.

echo.
echo === 4/4 credit_risk_sentiment_finbert (FinancialPhraseBank + FiQA, FinBERT) ===
python eval_runner.py --category credit_risk_sentiment_finbert
if errorlevel 1 echo Sentiment FinBERT eval had errors.

echo.
echo Done. Check data/proof/credit_risk_pd/ data/proof/credit_risk_pd_quantum/ data/proof/credit_risk_sentiment/ data/proof/credit_risk_sentiment_finbert/
