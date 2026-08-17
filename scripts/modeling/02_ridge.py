#!/usr/bin/env python3
"""
Modeling step 3: Ridge regression with missing-data dummies, standardization,
CV-selected alpha, and a fair head-to-head vs the baseline on an identical
validation population.

Missing-data strategy: has_mlb_record / has_aaa_record dummies (1 if the
player has ANY pre-KBO record at that level, via career_ip > 0) + train-mean
imputation for the level-specific numeric features when the record is
missing. This lets Ridge use the 3 players with neither MLB nor AAA record
(both dummies 0, features fall back to the train mean, i.e. contribute close
to nothing beyond the intercept-like effect) -- something the single-feature
baseline structurally could not do, since it had no fallback when its one
feature was undefined.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
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
            d[col] = d[col].fillna(train_means[col])  # safety net for any residual NaN
    for col in OTHER_FEATURES:
        d[col] = d[col].fillna(train_other_means[col])
    return d


# means computed on TRAIN only (no leakage), among players who actually have that level's record
train["has_mlb_record_tmp"] = (train["mlb_career_ip"].fillna(0) > 0).astype(int)
train["has_aaa_record_tmp"] = (train["aaa_career_ip"].fillna(0) > 0).astype(int)
train_means = {}
for level, cols in LEVEL_FEATURES.items():
    has_col = f"has_{level}_record_tmp"
    for col in cols:
        train_means[col] = train.loc[train[has_col] == 1, col].mean()
train_other_means = {col: train[col].mean() for col in OTHER_FEATURES}

print("=" * 70)
print("결측 대치에 쓴 train 평균값")
print("=" * 70)
for k, v in {**train_means, **train_other_means}.items():
    print(f"  {k}: {v:.3f}")

train_f = build_features(train, train_means, train_other_means)
val_f = build_features(val, train_means, train_other_means)

FEATURE_COLS = ALL_RAW_FEATURES + ["has_mlb_record", "has_aaa_record"]

print(f"\nTrain 결측 대치 후 잔여 NaN 체크: {train_f[FEATURE_COLS].isna().sum().sum()}건")
print(f"Val 결측 대치 후 잔여 NaN 체크: {val_f[FEATURE_COLS].isna().sum().sum()}건")

train_fit = train_f.dropna(subset=[TARGET])
val_fit = val_f.dropna(subset=[TARGET])
print(f"\nRidge 학습 표본: {len(train_fit)}/{len(train)} (전체 train 포함, 결측으로 인한 제외 없음)")
print(f"Ridge validation 평가 표본: {len(val_fit)}/{len(val)}")

# ---- explicit check: the 3 no-record players are now included ----
no_record = train_f[(train_f["has_mlb_record"] == 0) & (train_f["has_aaa_record"] == 0)]
print(f"\nMLB/AAA 둘 다 기록 없는 선수 (baseline에서는 제외됐던 3명): {no_record['선수명'].tolist()}")
print("-> has_mlb_record=0, has_aaa_record=0 상태로 Ridge 학습/평가에 포함됨 (더미 방식으로 커버됨)")

X_train_raw = train_fit[FEATURE_COLS].values
y_train = train_fit[TARGET].values
X_val_raw = val_fit[FEATURE_COLS].values
y_val = val_fit[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

alphas = np.logspace(-2, 3, 60)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
ridge.fit(X_train, y_train)

print("\n" + "=" * 70)
print("3. Ridge 회귀 (CV로 alpha 선택)")
print("=" * 70)
print(f"선택된 alpha: {ridge.alpha_:.4f}")

coef_table = pd.DataFrame({"feature": FEATURE_COLS, "coef": ridge.coef_}).sort_values(
    "coef", key=lambda s: s.abs(), ascending=False)
print("\nRidge 계수 (표준화된 변수 기준, 절대값 큰 순):")
print(coef_table.to_string(index=False))
print(f"\nintercept: {ridge.intercept_:.4f}")

pred_train_ridge = ridge.predict(X_train)
pred_val_ridge = ridge.predict(X_val)

# ---- baseline recomputed on the SAME val population (fair comparison) ----
def baseline_feature(d):
    return d["mlb_fip_last"].where(d["mlb_fip_last"].notna(), d["aaa_fip_last"])


train["baseline_fip"] = baseline_feature(train)
val_fit["baseline_fip"] = baseline_feature(val_fit)

train_bl_fit = train.dropna(subset=["baseline_fip", TARGET])
Xb_train = train_bl_fit[["baseline_fip"]].values
yb_train = train_bl_fit[TARGET].values
baseline_model = LinearRegression().fit(Xb_train, yb_train)

# predict for every val_fit row: linear pred if baseline_fip available, else train mean fallback
pred_val_baseline_fair = np.where(
    val_fit["baseline_fip"].notna(),
    baseline_model.predict(val_fit[["baseline_fip"]].fillna(0).values),
    yb_train.mean(),
)
n_val_fallback = val_fit["baseline_fip"].isna().sum()

print("\n" + "=" * 70)
print("4. 공정 비교 -- 동일한 validation 표본(n={})으로 재평가".format(len(val_fit)))
print("=" * 70)
print(f"(Baseline 쪽에서 baseline_fip가 없어 train 평균으로 대체 예측한 인원: {n_val_fallback}명)")


def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"model": label, "n": len(y_true), "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}


results = [
    metrics(y_val, pred_val_baseline_fair, "Baseline (fair, n={})".format(len(val_fit))),
    metrics(y_val, pred_val_ridge, "Ridge (n={})".format(len(val_fit))),
]
print("\n" + pd.DataFrame(results).to_string(index=False))

print("\n--- 참고: Train 성능 ---")
print(f"Ridge train: MAE={mean_absolute_error(y_train, pred_train_ridge):.3f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_train, pred_train_ridge)):.3f}, "
      f"R2={r2_score(y_train, pred_train_ridge):.3f}")
