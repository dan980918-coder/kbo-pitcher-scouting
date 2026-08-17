#!/usr/bin/env python3
"""
Modeling step 1-2: temporal train/val/test split + simplest baseline model.

Split (per PROJECT_GUIDELINES.md temporal validation principle):
  Train: 2014-2023 entry, Validation: 2024 entry, Test: 2025 entry (holdout,
  not inspected in this step or any step until final model comparison).

kbo_no_appearance==1 players (파커 마켈, 에니 로메로) have a missing target
(no KBO WAR at all -- STATIZ never created a page) and are excluded from
train/val/test; reported separately.

Baseline model: single-feature linear regression on "most recent overseas
FIP" -- mlb_fip_last if the player has an MLB record, else aaa_fip_last.
This is the simplest possible "just look at recent overseas performance"
reference point every more complex model must beat.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")

TARGET = "kbo_first_year_WAR"

# ---- excluded (missing target) ----
excluded = df[df["kbo_no_appearance"] == 1]
print("=" * 70)
print("제외 대상 (kbo_no_appearance=1, 타깃 결측)")
print("=" * 70)
print(excluded[["선수명", "연도", "kbo_no_appearance"]].to_string(index=False))

modeling_pop = df[df["kbo_no_appearance"] != 1].copy()

# ---- 1. temporal split ----
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()
test = modeling_pop[modeling_pop["연도"] == 2025].copy()

print("\n" + "=" * 70)
print("1. 데이터 분할 (temporal validation)")
print("=" * 70)
print(f"Train (2014-2023): {len(train)}명")
print(f"Validation (2024):  {len(val)}명")
print(f"Test (2025, holdout, 이번 단계 미열람): {len(test)}명")
print(f"제외(target 결측): {len(excluded)}명")
print(f"합계: {len(train) + len(val) + len(test) + len(excluded)}명 (167명과 일치해야 함)")

print("\nTrain 연도별 표본수:")
print(train["연도"].value_counts().sort_index().to_string())

# ---- 2. baseline model ----
print("\n" + "=" * 70)
print("2. Baseline 모델: 최근 해외 FIP 단일변수 선형회귀")
print("=" * 70)


def baseline_feature(d):
    return d["mlb_fip_last"].where(d["mlb_fip_last"].notna(), d["aaa_fip_last"])


for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
    split_df["baseline_fip"] = baseline_feature(split_df)

n_missing_both = train["baseline_fip"].isna().sum()
print(f"Train 중 mlb_fip_last, aaa_fip_last 둘 다 없는 선수: {n_missing_both}명 "
      f"({train[train['baseline_fip'].isna()]['선수명'].tolist()})")
print("-> 이 선수들은 baseline 모델 학습/평가에서 제외 (예측 근거 자체가 없음)")

train_fit = train.dropna(subset=["baseline_fip", TARGET])
val_fit = val.dropna(subset=["baseline_fip", TARGET])

X_train = train_fit[["baseline_fip"]].values
y_train = train_fit[TARGET].values
X_val = val_fit[["baseline_fip"]].values
y_val = val_fit[TARGET].values

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n학습 표본: {len(train_fit)}명 (전체 train {len(train)}명 중 결측 {len(train) - len(train_fit)}명 제외)")
print(f"계수: WAR = {model.intercept_:.4f} + ({model.coef_[0]:.4f}) * baseline_fip")
print("(FIP는 낮을수록 좋은 지표이므로 계수가 음수면 방향이 상식과 일치)")

pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

print(f"\n--- Train 성능 (n={len(train_fit)}) ---")
print(f"MAE={mean_absolute_error(y_train, pred_train):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_train, pred_train)):.3f}  "
      f"R2={r2_score(y_train, pred_train):.3f}")

print(f"\n--- Validation(2024) 성능 (n={len(val_fit)}) ---")
n_val_missing = val["baseline_fip"].isna().sum()
if n_val_missing:
    print(f"(주: val {len(val)}명 중 baseline_fip 결측 {n_val_missing}명은 이 평가에서 제외됨)")
print(f"MAE={mean_absolute_error(y_val, pred_val):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_val, pred_val)):.3f}  "
      f"R2={r2_score(y_val, pred_val):.3f}")

# always-predict-mean comparison, for context
mean_pred_val = np.full_like(y_val, y_train.mean())
print(f"\n--- 참고: 항상 train 평균({y_train.mean():.3f})만 예측하는 무정보 모델의 val 성능 ---")
print(f"MAE={mean_absolute_error(y_val, mean_pred_val):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_val, mean_pred_val)):.3f}  "
      f"R2={r2_score(y_val, mean_pred_val):.3f}")
