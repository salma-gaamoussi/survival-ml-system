"""
End-to-end training pipeline.
Run from the project root:
    python src/train.py --data-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib

from data.data import load_data, split_data
from src.features import add_features
from src.preprocess import build_preprocessor
from src.evaluate import score_model, compute_ibs
from models.km_cox import fit_kaplan_meier, fit_cox, c_index
from models.rsf import fit_rsf
from models.deepsurv import train_deepsurv, predict_risk


def main(data_path: str):
    print("=" * 50)
    print("Loading data...")
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    print("\nEngineering features...")
    X_train_feat = add_features(X_train)
    X_test_feat  = add_features(X_test)

    print("\nFitting preprocessor...")
    pre = build_preprocessor()
    X_train_pre = pre.fit_transform(X_train_feat)
    X_test_pre  = pre.transform(X_test_feat)
    print(f"  Feature matrix shape: {X_train_pre.shape}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pre, "models/preprocessor.pkl")

    os.makedirs("notebooks", exist_ok=True)
    print("\n--- Kaplan-Meier (population baseline) ---")
    fit_kaplan_meier(y_train, save_path="notebooks/km_curve.png")

    print("\n--- Cox Proportional Hazards ---")
    cox = fit_cox(X_train_pre, y_train)
    cox_ci = c_index(cox, X_test_pre, y_test)
    print(f"  C-index: {cox_ci:.3f}")

    print("\n--- Random Survival Forest ---")
    rsf = fit_rsf(X_train_pre, y_train)
    rsf_ci = c_index(rsf, X_test_pre, y_test)
    ibs    = compute_ibs(rsf, X_test_pre, y_train, y_test)
    print(f"  C-index: {rsf_ci:.3f}  |  IBS: {ibs:.4f}")
    joblib.dump(rsf, "models/rsf.pkl")

    print("\n--- DeepSurv (Cox neural network) ---")
    ds_model  = train_deepsurv(X_train_pre, y_train, epochs=50)
    ds_scores = predict_risk(ds_model, X_test_pre)
    ds_ci     = score_model("DeepSurv", ds_scores, y_test)

    print("\n" + "=" * 50)
    print("Results summary")
    print(f"  Cox PH    C-index : {cox_ci:.3f}")
    print(f"  RSF       C-index : {rsf_ci:.3f}  |  IBS : {ibs:.4f}")
    print(f"  DeepSurv  C-index : {ds_ci:.3f}")

    results = {
        "cox_cindex":      round(float(cox_ci), 4),
        "rsf_cindex":      round(float(rsf_ci), 4),
        "rsf_ibs":         round(float(ibs),    4),
        "deepsurv_cindex": round(float(ds_ci),  4),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nArtifacts saved:")
    print("  models/preprocessor.pkl")
    print("  models/rsf.pkl")
    print("  models/deepsurv.pt")
    print("  notebooks/km_curve.png")
    print("  results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all survival models")
    parser.add_argument("--data-path", required=True, help="Path to the Telco CSV")
    args = parser.parse_args()
    main(args.data_path)
