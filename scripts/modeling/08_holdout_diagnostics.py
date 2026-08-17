#!/usr/bin/env python3
"""Two diagnostics on the 2025 holdout result: bootstrap CI for Test R^2,
and sensitivity to the 3 largest-error breakout seasons."""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

with open(f"{ROOT}/reports/modeling/ridge_final.pkl", "rb") as f:
    artifact = pickle.load(f)

model, scaler = artifact["model"], artifact["scaler"]
FEATURE_COLS = artifact["feature_cols"]
LEVEL_FEATURES = artifact["level_features"]
OTHER_FEATURES = artifact["other_features"]
train_means, train_other_means = artifact["train_means"], artifact["train_other_means"]
TARGET = artifact["target"]


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
test = modeling_pop[modeling_pop["연도"] == 2025].copy()
test_f = build_features(test).dropna(subset=[TARGET])

X_test = scaler.transform(test_f[FEATURE_COLS].values)
y_test = test_f[TARGET].values
pred_test = model.predict(X_test)
names = test_f["선수명"].values

# ---- 1. bootstrap CI for Test R^2 ----
print("=" * 70)
print("1. Test R2 부트스트랩 (1000회, 복원추출, n=17 고정)")
print("=" * 70)
rng = np.random.default_rng(42)
n = len(y_test)
boot_r2 = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    yt, yp = y_test[idx], pred_test[idx]
    if np.var(yt) == 0:  # degenerate resample, skip (r2 undefined)
        continue
    boot_r2.append(r2_score(yt, yp))
boot_r2 = np.array(boot_r2)

point_r2 = r2_score(y_test, pred_test)
ci_low, ci_high = np.percentile(boot_r2, [2.5, 97.5])
print(f"유효 부트스트랩 반복 수: {len(boot_r2)}/1000 (분산 0인 재추출 제외)")
print(f"점추정 Test R2: {point_r2:.4f}")
print(f"부트스트랩 평균: {boot_r2.mean():.4f}, 중앙값: {np.median(boot_r2):.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"부트스트랩 표준편차: {boot_r2.std():.4f}")

val_r2 = 0.068
in_ci = ci_low <= val_r2 <= ci_high
print(f"\nVal R2({val_r2})가 이 95% CI 안에 있는가: {in_ci}")

pct_below_0 = (boot_r2 < 0).mean() * 100
print(f"참고: 부트스트랩 R2가 0 미만인 비율: {pct_below_0:.1f}%")

# ---- 2. sensitivity: drop 3 largest-error breakout players ----
print("\n" + "=" * 70)
print("2. 극단치 3명(코디 폰세, 라일리 톰프슨, 잭 로그) 제외 민감도")
print("=" * 70)
drop_names = {"코디 폰세", "라일리 톰프슨", "잭 로그"}
mask = ~pd.Series(names).isin(drop_names).values

y_full, p_full = y_test, pred_test
y_sub, p_sub = y_test[mask], pred_test[mask]

def metrics(y_true, y_pred, label):
    return {"case": label, "n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


res = [metrics(y_full, p_full, "17명 전체"), metrics(y_sub, p_sub, "14명 (3명 제외)")]
print(pd.DataFrame(res).to_string(index=False))

# also bootstrap the 14-person subset for a quick stability check
n_sub = len(y_sub)
boot_r2_sub = []
for _ in range(1000):
    idx = rng.integers(0, n_sub, size=n_sub)
    yt, yp = y_sub[idx], p_sub[idx]
    if np.var(yt) == 0:
        continue
    boot_r2_sub.append(r2_score(yt, yp))
boot_r2_sub = np.array(boot_r2_sub)
ci_low_sub, ci_high_sub = np.percentile(boot_r2_sub, [2.5, 97.5])
print(f"\n(참고) 14명 기준 R2 부트스트랩 95% CI: [{ci_low_sub:.4f}, {ci_high_sub:.4f}]")
