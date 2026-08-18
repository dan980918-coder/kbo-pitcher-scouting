#!/usr/bin/env python3
"""
KBO-wide "this season -> next season WAR" model (new track, separate from the
foreign-pitcher 8-variable Ridge). 1,694 season-pairs (2018->2019 transition
excluded, see model_selection.md SS11), temporal split by season_in:
Train=2015-2022, Val=2023, Test=2024 (holdout).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
TARGET = "WAR_out"
FEATURES = ["FIP_in", "WHIP_in", "K9_in", "BB9_in", "IP_in", "WAR_in", "ball_era"]

df = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_season_pairs_2015_2025.csv")
df = df[~((df["season_in"] == 2018) & (df["season_out"] == 2019))].copy()
print(f"2018->19 제외 후: {len(df)}")

before = len(df)
df = df.dropna(subset=FEATURES + [TARGET])
df = df[df["IP_in"] > 0]
print(f"IP_in=0/결측 제거 후: {len(df)} (제거 {before - len(df)}개)")

train = df[df["season_in"] <= 2022].copy()
val = df[df["season_in"] == 2023].copy()
test = df[df["season_in"] == 2024].copy()
print(f"Train(2015-2022, 2018->19 제외): {len(train)} / Val(2023): {len(val)} / Test(2024, holdout): {len(test)}")

X_train_raw = train[FEATURES].values
y_train = train[TARGET].values
X_val_raw = val[FEATURES].values
y_val = val[TARGET].values
X_test_raw = test[FEATURES].values
y_test = test[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # 15 fold evaluations


def metrics(y_true, y_pred):
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


results = []

# ---------------------------------------------------------------------
# 1. Baseline: single-variable (FIP_in) linear regression
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("1. Baseline (단일변수 FIP_in)")
print("=" * 90)
bl = LinearRegression().fit(train[["FIP_in"]].values, y_train)
pred_train_bl = bl.predict(train[["FIP_in"]].values)
pred_val_bl = bl.predict(val[["FIP_in"]].values)
pred_test_bl = bl.predict(test[["FIP_in"]].values)
m_train, m_val, m_test = metrics(y_train, pred_train_bl), metrics(y_val, pred_val_bl), metrics(y_test, pred_test_bl)
print(f"Train {m_train} | Val {m_val} | Test {m_test}")
results.append({"model": "Baseline(FIP_in)", "cv_R2": np.nan, "Train_R2": m_train["R2"],
                 "Val_R2": m_val["R2"], "Val_MAE": m_val["MAE"], "Test_R2": m_test["R2"], "Test_MAE": m_test["MAE"]})

# ---------------------------------------------------------------------
# 2. Ridge (7 features)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. Ridge (7변수: FIP/WHIP/K9/BB9/IP/WAR_in/ball_era)")
print("=" * 90)
alphas = np.logspace(-2, 3, 60)
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = RidgeCV(alphas=alphas, cv=cv5, scoring="neg_mean_absolute_error")
ridge.fit(X_train, y_train)
pred_train_r, pred_val_r, pred_test_r = ridge.predict(X_train), ridge.predict(X_val), ridge.predict(X_test)
m_train, m_val, m_test = metrics(y_train, pred_train_r), metrics(y_val, pred_val_r), metrics(y_test, pred_test_r)
cv_r2_ridge = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=rkf, scoring="r2")
print(f"alpha={ridge.alpha_:.4f}")
print(f"Train {m_train} | Val {m_val} | Test {m_test}")
print(f"CV R2 (25회): mean={cv_r2_ridge.mean():.3f}, std={cv_r2_ridge.std():.3f}")
coefs = dict(zip(FEATURES, ridge.coef_))
print("표준화 계수:", {k: round(v, 3) for k, v in coefs.items()})
results.append({"model": f"Ridge(alpha={ridge.alpha_:.2f})", "cv_R2": round(cv_r2_ridge.mean(), 3),
                 "Train_R2": m_train["R2"], "Val_R2": m_val["R2"], "Val_MAE": m_val["MAE"],
                 "Test_R2": m_test["R2"], "Test_MAE": m_test["MAE"]})

# ---------------------------------------------------------------------
# 3. RF / GB with repeated CV grid search (avoid overfit-picking, per 05_tree_retune.py protocol)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. RF / GB -- 좁힌 그리드 x Repeated 5-fold CV (5x5=25회 평가)")
print("=" * 90)

grid = {"max_depth": [2, 3, 4], "n_estimators": [50, 100], "min_samples_leaf": [15, 30]}
combos = list(itertools.product(grid["max_depth"], grid["n_estimators"], grid["min_samples_leaf"]))

rf_rows = []
for depth, n_est, leaf in combos:
    m = RandomForestRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf, random_state=42, n_jobs=-1)
    r2s = cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="r2")
    maes = -cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="neg_mean_absolute_error")
    rf_rows.append({"depth": depth, "n_estimators": n_est, "leaf": leaf, "cv_R2_mean": r2s.mean(),
                     "cv_R2_std": r2s.std(), "cv_MAE_mean": maes.mean()})
rf_table = pd.DataFrame(rf_rows).sort_values("cv_MAE_mean")
print("\n--- RF 상위 8 (cv_MAE_mean 기준) ---")
print(rf_table.head(8).round(4).to_string(index=False))

gb_rows = []
for depth, n_est, leaf in combos:
    for lr in [0.05, 0.1]:
        m = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf,
                                       learning_rate=lr, random_state=42)
        r2s = cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="r2")
        maes = -cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="neg_mean_absolute_error")
        gb_rows.append({"depth": depth, "n_estimators": n_est, "leaf": leaf, "lr": lr,
                         "cv_R2_mean": r2s.mean(), "cv_R2_std": r2s.std(), "cv_MAE_mean": maes.mean()})
gb_table = pd.DataFrame(gb_rows).sort_values("cv_MAE_mean")
print("\n--- GB 상위 8 (cv_MAE_mean 기준) ---")
print(gb_table.head(8).round(4).to_string(index=False))

# pick most stable among top 5 by mean (lowest std), matching established protocol
def pick_stable(table, top_n=5):
    top = table.head(top_n).copy()
    return top.loc[top["cv_R2_std"].idxmin()]


rf_pick = pick_stable(rf_table)
gb_pick = pick_stable(gb_table)
print(f"\nRF 선택: {rf_pick.to_dict()}")
print(f"GB 선택: {gb_pick.to_dict()}")

rf_final = RandomForestRegressor(max_depth=int(rf_pick["depth"]), n_estimators=int(rf_pick["n_estimators"]),
                                  min_samples_leaf=int(rf_pick["leaf"]), random_state=42, n_jobs=-1).fit(X_train_raw, y_train)
gb_final = GradientBoostingRegressor(max_depth=int(gb_pick["depth"]), n_estimators=int(gb_pick["n_estimators"]),
                                      min_samples_leaf=int(gb_pick["leaf"]), learning_rate=gb_pick["lr"],
                                      random_state=42).fit(X_train_raw, y_train)

for name, model, pick, cvr2 in [("RF", rf_final, rf_pick, rf_pick["cv_R2_mean"]), ("GB", gb_final, gb_pick, gb_pick["cv_R2_mean"])]:
    pred_train_m = model.predict(X_train_raw)
    pred_val_m = model.predict(X_val_raw)
    pred_test_m = model.predict(X_test_raw)
    m_train, m_val, m_test = metrics(y_train, pred_train_m), metrics(y_val, pred_val_m), metrics(y_test, pred_test_m)
    print(f"\n{name}: Train {m_train} | Val {m_val} | Test {m_test}")
    results.append({"model": f"{name}(재탐색)", "cv_R2": round(cvr2, 3), "Train_R2": m_train["R2"],
                     "Val_R2": m_val["R2"], "Val_MAE": m_val["MAE"], "Test_R2": m_test["R2"], "Test_MAE": m_test["MAE"]})

# ---------------------------------------------------------------------
# 4. Summary + CV/Val direction agreement
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. 종합 비교 + CV/Val 방향 일치 확인")
print("=" * 90)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

print("\n--- CV 방향 vs Val 방향 (있는 모델만) ---")
for r in results:
    if pd.isna(r["cv_R2"]):
        continue
    direction = "일치" if (r["cv_R2"] > 0) == (r["Val_R2"] > 0) else "불일치(주의)"
    print(f"{r['model']}: CV_R2={r['cv_R2']}, Val_R2={r['Val_R2']} -> {direction}")

res_df.to_csv(f"{ROOT}/reports/modeling/kbo_wide_model_comparison.csv", index=False)
print(f"\n저장: {ROOT}/reports/modeling/kbo_wide_model_comparison.csv")
