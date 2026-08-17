#!/usr/bin/env python3
"""
Final model training: Ridge regression, the model selected after comparing
Baseline / Ridge / Random Forest / Gradient Boosting (see
reports/modeling/model_selection.md for the full comparison and rationale).

Saves a reproducible artifact (reports/modeling/ridge_final.pkl +
ridge_final.json) containing: the fitted Ridge model, the StandardScaler,
the exact feature list/order, and the train-mean imputation values needed
to build features for any new player the same way this script does.

Run from repo root: python3 scripts/modeling/06_train_final_ridge.py
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
OUT_DIR = f"{ROOT}/reports/modeling"
TARGET = "kbo_first_year_WAR"

LEVEL_FEATURES = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career"],
    "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES = ["age_at_kbo_entry", "n_pitch_types_recorded"]
FEATURE_COLS = LEVEL_FEATURES["mlb"] + LEVEL_FEATURES["aaa"] + OTHER_FEATURES + ["has_mlb_record", "has_aaa_record"]


def build_features(d, train_means, train_other_means):
    d = d.copy()
    d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
    d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)
    for level, cols in LEVEL_FEATURES.items():
        has_col = f"has_{level}_record"
        for col in cols:
            d[col] = d[col].where(d[has_col] == 1, train_means[col])
            d[col] = d[col].fillna(train_means[col])
    for col in OTHER_FEATURES:
        d[col] = d[col].fillna(train_other_means[col])
    return d


def main():
    df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
    modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
    train = modeling_pop[modeling_pop["연도"] <= 2023].copy()  # test(2025) intentionally not touched here

    train["has_mlb_record_tmp"] = (train["mlb_career_ip"].fillna(0) > 0).astype(int)
    train["has_aaa_record_tmp"] = (train["aaa_career_ip"].fillna(0) > 0).astype(int)
    train_means = {}
    for level, cols in LEVEL_FEATURES.items():
        has_col = f"has_{level}_record_tmp"
        for col in cols:
            train_means[col] = float(train.loc[train[has_col] == 1, col].mean())
    train_other_means = {col: float(train[col].mean()) for col in OTHER_FEATURES}

    train_f = build_features(train, train_means, train_other_means)
    train_fit = train_f.dropna(subset=[TARGET])

    X_train_raw = train_fit[FEATURE_COLS].values
    y_train = train_fit[TARGET].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    alphas = np.logspace(-2, 3, 60)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)

    artifact = {
        "model": ridge,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "level_features": LEVEL_FEATURES,
        "other_features": OTHER_FEATURES,
        "train_means": train_means,
        "train_other_means": train_other_means,
        "target": TARGET,
        "train_years": "2014-2023",
        "n_train": len(train_fit),
    }
    with open(f"{OUT_DIR}/ridge_final.pkl", "wb") as f:
        pickle.dump(artifact, f)

    summary = {
        "alpha": float(ridge.alpha_),
        "intercept": float(ridge.intercept_),
        "feature_cols": FEATURE_COLS,
        "coefficients_standardized": {c: float(v) for c, v in zip(FEATURE_COLS, ridge.coef_)},
        "scaler_mean": {c: float(v) for c, v in zip(FEATURE_COLS, scaler.mean_)},
        "scaler_scale": {c: float(v) for c, v in zip(FEATURE_COLS, scaler.scale_)},
        "train_means_for_imputation": train_means,
        "train_other_means_for_imputation": train_other_means,
        "target": TARGET,
        "train_years": "2014-2023",
        "n_train": len(train_fit),
        "note": "prediction = intercept + sum(coef_i * (raw_feature_i - scaler_mean_i) / scaler_scale_i)",
    }
    with open(f"{OUT_DIR}/ridge_final.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"alpha={ridge.alpha_:.4f}, n_train={len(train_fit)}")
    print(f"Saved -> {OUT_DIR}/ridge_final.pkl, ridge_final.json")


if __name__ == "__main__":
    main()
