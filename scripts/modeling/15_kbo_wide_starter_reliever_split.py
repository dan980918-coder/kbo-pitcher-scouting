#!/usr/bin/env python3
"""
Starter vs reliever split of the KBO-wide "this season -> next season WAR"
Ridge (1yr-feature version, the one shown robust across 5 walk-forward
folds). Split by GS_in/G_in >= 0.5 (starter) vs < 0.5 (reliever), same
FEATURES_7 and same temporal split as the main model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
FEATURES_7 = ["FIP_in", "WHIP_in", "K9_in", "BB9_in", "IP_in", "WAR_in", "ball_era"]


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


pairs = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_season_pairs_2015_2025.csv")
pairs = pairs[~((pairs["season_in"] == 2018) & (pairs["season_out"] == 2019))].copy()
pairs = pairs.dropna(subset=FEATURES_7 + ["WAR_out", "G_in", "GS_in"])
pairs = pairs[pairs["IP_in"] > 0]
pairs["gs_ratio"] = pairs["GS_in"] / pairs["G_in"]
pairs["role"] = np.where(pairs["gs_ratio"] >= 0.5, "선발형(GS/G>=0.5)", "불펜형(GS/G<0.5)")

print("=" * 90)
print("역할별 표본 분포")
print("=" * 90)
print(pairs["role"].value_counts())
print()
print(pairs.groupby("role")[["G_in", "GS_in", "gs_ratio", "IP_in", "WAR_in"]].mean().round(2))

results = []
for role, g in pairs.groupby("role"):
    train = g[g["season_in"] <= 2022]
    val = g[g["season_in"] == 2023]
    test = g[g["season_in"] == 2024]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES_7].values)
    y_train = train["WAR_out"].values
    alphas = np.logspace(-2, 3, 60)
    ridge = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)

    X_val, X_test = scaler.transform(val[FEATURES_7].values), scaler.transform(test[FEATURES_7].values)
    m_val = metrics(val["WAR_out"].values, ridge.predict(X_val))
    m_test = metrics(test["WAR_out"].values, ridge.predict(X_test))
    cv_r2 = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

    coefs = dict(zip(FEATURES_7, np.round(ridge.coef_, 3)))

    print("\n" + "=" * 90)
    print(f"{role}  (n_train={len(train)}, n_val={len(val)}, n_test={len(test)})")
    print("=" * 90)
    print(f"alpha={ridge.alpha_:.3f}, CV R2(5fold)={cv_r2.mean():.3f}")
    print(f"Val {m_val} | Test {m_test}")
    print(f"표준화 계수: {coefs}")

    results.append({"role": role, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                     "alpha": round(ridge.alpha_, 3), "cv_R2": round(cv_r2.mean(), 3),
                     "Val_R2": m_val["R2"], "Val_MAE": m_val["MAE"], "Test_R2": m_test["R2"], "Test_MAE": m_test["MAE"],
                     **{f"coef_{k}": v for k, v in coefs.items()}})

print("\n" + "=" * 90)
print("종합 비교 (전체 모델: Val R2=0.358, Test R2=0.352 -- 앞서 나온 결과)")
print("=" * 90)
res_df = pd.DataFrame(results)
print(res_df[["role", "n_train", "alpha", "cv_R2", "Val_R2", "Val_MAE", "Test_R2", "Test_MAE"]].to_string(index=False))
print()
print("계수 비교:")
print(res_df[["role"] + [c for c in res_df.columns if c.startswith("coef_")]].to_string(index=False))

res_df.to_csv(f"{ROOT}/reports/modeling/kbo_wide_starter_reliever_split.csv", index=False)
print(f"\n저장: {ROOT}/reports/modeling/kbo_wide_starter_reliever_split.csv")
