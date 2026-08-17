#!/usr/bin/env python3
"""
2025 holdout test evaluation. Loads the saved Ridge artifact as-is (no
retraining, no refitting scaler/imputation -- everything comes from the
train-only values baked into ridge_final.pkl) and scores it on the 17
players never touched during split/baseline/Ridge/RF/GB comparison.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

with open(f"{ROOT}/reports/modeling/ridge_final.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
scaler = artifact["scaler"]
FEATURE_COLS = artifact["feature_cols"]
LEVEL_FEATURES = artifact["level_features"]
OTHER_FEATURES = artifact["other_features"]
train_means = artifact["train_means"]
train_other_means = artifact["train_other_means"]
TARGET = artifact["target"]

print(f"불러온 아티팩트: alpha={model.alpha_:.4f}, n_train={artifact['n_train']}, "
      f"train_years={artifact['train_years']}")
print(f"features: {FEATURE_COLS}")


def build_features(d):
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


df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()
test = modeling_pop[modeling_pop["연도"] == 2025].copy()

print(f"\n2025 holdout 대상: {len(test)}명 (kbo_no_appearance=1 제외 대상 여부 확인: "
      f"{(df[df['연도']==2025]['kbo_no_appearance']==1).sum()}명 발견)")

train_f = build_features(train).dropna(subset=[TARGET])
val_f = build_features(val).dropna(subset=[TARGET])
test_f = build_features(test).dropna(subset=[TARGET])
print(f"타깃 결측으로 제외된 2025 선수: {len(test) - len(test_f)}명")


def predict(d):
    X = scaler.transform(d[FEATURE_COLS].values)
    return model.predict(X)


pred_train = predict(train_f)
pred_val = predict(val_f)
pred_test = predict(test_f)


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


print("\n" + "=" * 70)
print("1. Train / Validation / Test(2025 holdout) 성능")
print("=" * 70)
rows = [
    {"split": "Train (2014-2023)", **metrics(train_f[TARGET].values, pred_train)},
    {"split": "Validation (2024)", **metrics(val_f[TARGET].values, pred_val)},
    {"split": "Test (2025, holdout)", **metrics(test_f[TARGET].values, pred_test)},
]
print(pd.DataFrame(rows).to_string(index=False))

val_r2 = rows[1]["R2"]
test_r2 = rows[2]["R2"]
print(f"\n2. Val R2={val_r2} vs Test R2={test_r2}  (차이={round(test_r2-val_r2,3)})")

# ---- 3. per-player prediction vs actual ----
test_f = test_f.copy()
test_f["pred_WAR"] = pred_test
test_f["error"] = test_f["pred_WAR"] - test_f[TARGET]
test_f["abs_error"] = test_f["error"].abs()

print("\n" + "=" * 70)
print("3. 2025 홀드아웃 17명 전원: 예측 vs 실제")
print("=" * 70)
show = test_f[["선수명", TARGET, "pred_WAR", "error"]].rename(
    columns={TARGET: "actual_WAR"}).round(3).sort_values("actual_WAR", ascending=False)
print(show.to_string(index=False))

print("\n--- 오차 상위 5명 (|예측-실제| 기준) ---")
top_err = test_f[["선수명", TARGET, "pred_WAR", "error", "abs_error"]].rename(
    columns={TARGET: "actual_WAR"}).round(3).sort_values("abs_error", ascending=False).head(5)
print(top_err.to_string(index=False))
