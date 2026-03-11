#!/usr/bin/env python3
"""Load pd_model_local_v2.pkl and report whether it is the stacked model (XGB+LGB) or XGBoost only."""

import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # So joblib can deserialize _StackedPDWrapper
    import credit_risk.models.pd_model  # noqa: F401
    pkl_path = repo_root / "models" / "pd" / "pd_model_local_v2.pkl"
    if not pkl_path.exists():
        print(f"Not found: {pkl_path}")
        sys.exit(1)
    import joblib
    data = joblib.load(pkl_path)
    model = data.get("model")
    if model is None:
        print("No 'model' key in pkl")
        sys.exit(1)
    # _StackedPDWrapper has xgb_model and lgb_model
    is_stacked = hasattr(model, "xgb_model") and hasattr(model, "lgb_model")
    cls_name = type(model).__name__
    print(f"Model class: {cls_name}")
    print(f"Saved model is STACKED (XGB+LGB): {is_stacked}")
    if "metadata" in data:
        meta = data["metadata"]
        if "val_auc_roc" in meta:
            print(f"Stored val_auc_roc: {meta.get('val_auc_roc')}")
        if "test_auc_roc" in meta:
            print(f"Stored test_auc_roc: {meta.get('test_auc_roc')}")
    return 0 if is_stacked else 1

if __name__ == "__main__":
    sys.exit(main())
