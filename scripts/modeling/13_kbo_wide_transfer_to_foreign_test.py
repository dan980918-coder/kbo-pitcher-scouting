#!/usr/bin/env python3
"""
Direct empirical test: does the KBO-wide "this season -> next season WAR"
Ridge model (trained on domestic-continuity pairs) actually transfer to
foreign debutants when their MLB/AAA stats are mapped into its feature slots?

No refitting on foreign data -- the domestic-trained Ridge (coefficients +
scaler) is applied as-is to mapped foreign inputs, exactly the "does this
generalize out-of-population" test the prior claim should have been based on.

WHIP_in is dropped (no MLB/AAA WHIP exists in analysis_dataset_v1.csv for
the 167 players) -- retrained on the 4 remaining features (IP_in, FIP_in,
K9_in, BB9_in) for a clean, complete substitution.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
FEATURES = ["IP_in", "FIP_in", "K9_in", "BB9_in"]
TARGET = "WAR_out"


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


# ---------------------------------------------------------------------
# 1. Retrain KBO-wide Ridge on 4 features (WHIP dropped), same split as before
# ---------------------------------------------------------------------
pairs = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_season_pairs_2015_2025.csv")
pairs = pairs[~((pairs["season_in"] == 2018) & (pairs["season_out"] == 2019))].copy()
pairs = pairs.dropna(subset=FEATURES + [TARGET])
pairs = pairs[pairs["IP_in"] > 0]
train = pairs[pairs["season_in"] <= 2022]
val = pairs[pairs["season_in"] == 2023]

X_train_raw = train[FEATURES].values
y_train = train[TARGET].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

alphas = np.logspace(-2, 3, 60)
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = RidgeCV(alphas=alphas, cv=cv5, scoring="neg_mean_absolute_error")
ridge.fit(X_train, y_train)

X_val = scaler.transform(val[FEATURES].values)
pred_val = ridge.predict(X_val)
print("=" * 90)
print("1. KBO 전체 모델 재학습 (4변수: IP_in, FIP_in, K9_in, BB9_in -- WHIP 제외)")
print("=" * 90)
print(f"alpha={ridge.alpha_:.4f}")
print(f"KBO 내부 Val(2023) 기준: {metrics(val[TARGET].values, pred_val)}")
coefs = dict(zip(FEATURES, ridge.coef_))
print("표준화 계수:", {k: round(v, 3) for k, v in coefs.items()})
print(f"scaler mean: {dict(zip(FEATURES, scaler.mean_.round(3)))}")
print(f"scaler scale: {dict(zip(FEATURES, scaler.scale_.round(3)))}")

# ---------------------------------------------------------------------
# 2. Map foreign-pitcher inputs (MLB priority, then AAA, per has_record convention)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. 167명 외국인 투수 입력값 매핑")
print("=" * 90)

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)

n_neither = ((d["has_mlb_record"] == 0) & (d["has_aaa_record"] == 0)).sum()
print(f"MLB/AAA 기록 둘 다 없는 선수: {n_neither}명 -- 매핑 자체가 불가능하므로 이번 평가에서 제외")

d["IP_in_mapped"] = np.where(d["has_mlb_record"] == 1, d["mlb_ip_last"], d["aaa_ip_last"])
d["FIP_in_mapped"] = np.where(d["has_mlb_record"] == 1, d["mlb_fip_minus_career"], d["aaa_fip_minus_career"])
d["K9_in_mapped"] = np.where(d["has_mlb_record"] == 1, d["mlb_k9_last"], d["aaa_k9_last"])
d["BB9_in_mapped"] = np.where(d["has_mlb_record"] == 1, d["mlb_bb9_last"], d["aaa_bb9_last"])

mapped_cols = ["IP_in_mapped", "FIP_in_mapped", "K9_in_mapped", "BB9_in_mapped"]
evaluable = d[(d["has_mlb_record"] == 1) | (d["has_aaa_record"] == 1)].copy()
evaluable = evaluable.dropna(subset=mapped_cols + ["kbo_first_year_WAR"])
print(f"매핑 가능 + 타깃 있는 선수: {len(evaluable)}명 (167명 모집단 중)")

X_foreign = scaler.transform(evaluable[mapped_cols].values)
pred_foreign = ridge.predict(X_foreign)
y_foreign = evaluable["kbo_first_year_WAR"].values

evaluable["predicted_WAR_kbo_wide_mapped"] = pred_foreign

# ---------------------------------------------------------------------
# 3. Compare vs actual, and vs the 8-var foreign-only Ridge (on same subset + on its own 2024 Val)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. 실제 kbo_first_year_WAR과 비교")
print("=" * 90)
print(f"전체 매핑 가능 인원({len(evaluable)}명) 기준: {metrics(y_foreign, pred_foreign)}")

val_2024 = evaluable[evaluable["연도"] == 2024]
print(f"\n(참고) 2024 Val 서브셋만({len(val_2024)}명, 기존 8변수 Ridge의 Val R2=0.068과 같은 대상): "
      f"{metrics(val_2024['kbo_first_year_WAR'].values, val_2024['predicted_WAR_kbo_wide_mapped'].values)}")

print(f"\n예측범위: {pred_foreign.min():.2f} ~ {pred_foreign.max():.2f} (실제 범위: {y_foreign.min():.2f} ~ {y_foreign.max():.2f})")

out_cols = ["선수명", "연도", "kbo_first_year_WAR", "predicted_WAR_kbo_wide_mapped"] + mapped_cols
evaluable[out_cols].to_csv(f"{ROOT}/reports/modeling/kbo_wide_mapped_to_foreign_predictions_lastIP.csv", index=False)
print(f"\n저장: {ROOT}/reports/modeling/kbo_wide_mapped_to_foreign_predictions_lastIP.csv")

# evaluable already carries the original 8-var Ridge's predicted_WAR column (inherited from df)
both = evaluable.copy()
both["predicted_WAR"] = pd.to_numeric(both["predicted_WAR"], errors="coerce")
both = both.dropna(subset=["predicted_WAR"])
print(f"\n같은 {len(both)}명 대상, 기존 8변수 Ridge 예측치로 비교: "
      f"{metrics(both['kbo_first_year_WAR'].values, both['predicted_WAR'].values)}")

val17_both = both[both["연도"] == 2024]
print(f"\n2024 Val 서브셋({len(val17_both)}명)만 같은 대상으로 재비교:")
print(f"  KBO 전체모델(4변수, 매핑): {metrics(val17_both['kbo_first_year_WAR'].values, val17_both['predicted_WAR_kbo_wide_mapped'].values)}")
print(f"  기존 8변수 외국인 Ridge  : {metrics(val17_both['kbo_first_year_WAR'].values, val17_both['predicted_WAR'].values)}")
