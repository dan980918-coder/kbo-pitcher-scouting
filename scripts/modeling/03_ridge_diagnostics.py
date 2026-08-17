#!/usr/bin/env python3
"""Modeling diagnostics: alpha-stability curve + aaa_fip_last sign check."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
TARGET = "kbo_first_year_WAR"

modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()

LEVEL_FEATURES = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career"],
    "aaa": ["aaa_fip_last", "aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES = ["age_at_kbo_entry", "n_pitch_types_recorded"]
ALL_RAW_FEATURES = LEVEL_FEATURES["mlb"] + LEVEL_FEATURES["aaa"] + OTHER_FEATURES


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


train["has_mlb_record_tmp"] = (train["mlb_career_ip"].fillna(0) > 0).astype(int)
train["has_aaa_record_tmp"] = (train["aaa_career_ip"].fillna(0) > 0).astype(int)
train_means = {}
for level, cols in LEVEL_FEATURES.items():
    has_col = f"has_{level}_record_tmp"
    for col in cols:
        train_means[col] = train.loc[train[has_col] == 1, col].mean()
train_other_means = {col: train[col].mean() for col in OTHER_FEATURES}

train_f = build_features(train, train_means, train_other_means)
val_f = build_features(val, train_means, train_other_means)

FEATURE_COLS = ALL_RAW_FEATURES + ["has_mlb_record", "has_aaa_record"]
train_fit = train_f.dropna(subset=[TARGET])
val_fit = val_f.dropna(subset=[TARGET])

X_train_raw = train_fit[FEATURE_COLS].values
y_train = train_fit[TARGET].values
X_val_raw = val_fit[FEATURE_COLS].values
y_val = val_fit[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# ---- 1. alpha stability curve ----
print("=" * 74)
print("1. Alpha 안정성 -- CV 성능 곡선 (5-fold, 동일 fold 재사용)")
print("=" * 74)
alphas_report = np.logspace(-2, 3, 16)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for a in alphas_report:
    model = Ridge(alpha=a)
    mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    rows.append({"alpha": round(a, 3), "cv_MAE_mean": round(mae_scores.mean(), 4),
                 "cv_MAE_std": round(mae_scores.std(), 4),
                 "cv_R2_mean": round(r2_scores.mean(), 4), "cv_R2_std": round(r2_scores.std(), 4)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
best_idx = t["cv_MAE_mean"].idxmin()
print(f"\nCV MAE 기준 최적 alpha (이 그리드 내): {t.loc[best_idx, 'alpha']} (RidgeCV가 고른 값: 13.66과 비교)")

# ---- 2. aaa_fip_last sign check ----
print("\n" + "=" * 74)
print("2. aaa_fip_last 부호 확인")
print("=" * 74)
corr_cols = ["aaa_fip_last", "aaa_hr9_last", "aaa_bb9_3yr"]
sub = train[corr_cols].dropna()
print(f"(train, pairwise-complete n={len(sub)})")
print(sub.corr().round(3).to_string())

print("\n--- aaa_fip_last 제외하고 Ridge 재학습 (동일 alpha 그리드로 재선택) ---")
FEATURE_COLS_NO_FIP = [c for c in FEATURE_COLS if c != "aaa_fip_last"]
X_train2_raw = train_fit[FEATURE_COLS_NO_FIP].values
X_val2_raw = val_fit[FEATURE_COLS_NO_FIP].values
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2_raw)
X_val2 = scaler2.transform(X_val2_raw)

from sklearn.linear_model import RidgeCV
alphas = np.logspace(-2, 3, 60)
ridge2 = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
ridge2.fit(X_train2, y_train)
pred_val2 = ridge2.predict(X_val2)

coef2 = pd.DataFrame({"feature": FEATURE_COLS_NO_FIP, "coef": ridge2.coef_}).sort_values(
    "coef", key=lambda s: s.abs(), ascending=False)
print(f"선택된 alpha (aaa_fip_last 제외): {ridge2.alpha_:.4f}")
print(coef2.to_string(index=False))


def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"model": label, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}


# original model (with aaa_fip_last) for direct comparison
ridge_full = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
ridge_full.fit(X_train, y_train)
pred_val_full = ridge_full.predict(X_val)

print("\n--- Validation(2024) 비교: aaa_fip_last 포함 vs 제외 ---")
res = [
    metrics(y_val, pred_val_full, "포함 (9개 변수)"),
    metrics(y_val, pred_val2, "제외 (8개 변수)"),
]
print(pd.DataFrame(res).to_string(index=False))
