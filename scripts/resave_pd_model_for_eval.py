#!/usr/bin/env python3
"""
Optional one-off script: load pd_model_local_v1.pkl (saved from notebook with __main__._StackedPDWrapper)
with the sys.modules hack, then re-save as pd_model_local_v2.pkl so the class is serialized under
credit_risk.models.pd_model._StackedPDWrapper.

The notebook (02a_pd_xgboost_training) now imports _StackedPDWrapper from credit_risk.models.pd_model
before saving, so newly saved v1.pkl files load in eval_runner without this script. Use this script
only if you have an existing v1.pkl that was saved before that notebook change.

Run from repo root:
    python scripts/resave_pd_model_for_eval.py

Requires: models/pd/pd_model_local_v1.pkl (from notebook Save cell).
Creates:  models/pd/pd_model_local_v2.pkl
"""
from pathlib import Path
import sys

# Ensure repo root is on path so "credit_risk" can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Inject so joblib can resolve __main__._StackedPDWrapper when loading v1
from credit_risk.models.pd_model import _StackedPDWrapper
main_module = sys.modules["__main__"]
if not hasattr(main_module, "_StackedPDWrapper"):
    setattr(main_module, "_StackedPDWrapper", _StackedPDWrapper)

import joblib

V1_PATH = REPO_ROOT / "models" / "pd" / "pd_model_local_v1.pkl"
V2_PATH = REPO_ROOT / "models" / "pd" / "pd_model_local_v2.pkl"


def main():
    if not V1_PATH.exists():
        print(f"Missing {V1_PATH}. Run the notebook Save cell first to create v1.")
        sys.exit(1)
    V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {V1_PATH} ...")
    model_data = joblib.load(V1_PATH)
    print(f"Re-saving to {V2_PATH} (clean module path for _StackedPDWrapper) ...")
    joblib.dump(model_data, V2_PATH)
    print("Done. Use pd_model_local_v2.pkl for eval_runner (it is preferred automatically).")


if __name__ == "__main__":
    main()
