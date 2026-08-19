#!/usr/bin/env python3
"""
3-way role classification (pure bullpen / swingman / pure starter), boundary
data-driven from 21_gs_ratio_distribution.py (GS/G==0 dominant mode 57.2%,
GS/G>=0.9 second mode 18.7%). Each group gets its own role-matched workload
feature, then per-group predictions are pooled back into the full Val/Test
set to verify the group-level improvements survive at the population level.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


pairs = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_season_pairs_2015_2025.csv")
pairs = pairs[~((pairs["season_in"] == 2018) & (pairs["season_out"] == 2019))].copy()
BASE = ["FIP_in", "WHIP_in", "K9_in", "BB9_in", "WAR_in", "ball_era"]
pairs = pairs.dropna(subset=BASE + ["IP_in", "G_in", "GS_in", "WAR_out"])
pairs = pairs[pairs["IP_in"] > 0]
pairs["gs_ratio"] = pairs["GS_in"] / pairs["G_in"]


def classify(r):
    if r == 0:
        return "순수불펜(GS/G=0)"
    if r >= 0.9:
        return "순수선발(GS/G>=0.9)"
    return "스윙맨(0<GS/G<0.9)"


pairs["role3"] = pairs["gs_ratio"].apply(classify)
print("그룹별 표본수:")
print(pairs["role3"].value_counts())

CONFIGS = {
    "순수불펜(GS/G=0)": BASE + ["G_in"],
    "순수선발(GS/G>=0.9)": BASE + ["IP_in"],
    "스윙맨(0<GS/G<0.9)": BASE + ["IP_in", "G_in"],
}

results = []
all_val_pred, all_val_actual = [], []
all_test_pred, all_test_actual = [], []

for role, features in CONFIGS.items():
    g = pairs[pairs["role3"] == role]
    train = g[g["season_in"] <= 2022]
    val = g[g["season_in"] == 2023]
    test = g[g["season_in"] == 2024]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].values)
    y_train = train["WAR_out"].values
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 60), cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)
    cv_r2 = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

    X_val, X_test = scaler.transform(val[features].values), scaler.transform(test[features].values)
    pred_val, pred_test = ridge.predict(X_val), ridge.predict(X_test)
    m_val = metrics(val["WAR_out"].values, pred_val)
    m_test = metrics(test["WAR_out"].values, pred_test)
    coefs = dict(zip(features, np.round(ridge.coef_, 3)))

    print(f"\n=== {role} (n_train={len(train)}, n_val={len(val)}, n_test={len(test)}) ===")
    print(f"alpha={ridge.alpha_:.3f}, CV R2={cv_r2.mean():.3f}")
    print(f"Val {m_val} | Test {m_test}")
    print(f"계수: {coefs}")

    results.append({"role": role, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                     "cv_R2": round(cv_r2.mean(), 3), "Val_R2": m_val["R2"], "Test_R2": m_test["R2"]})

    all_val_pred.append(pred_val); all_val_actual.append(val["WAR_out"].values)
    all_test_pred.append(pred_test); all_test_actual.append(test["WAR_out"].values)

print("\n" + "=" * 90)
print("그룹별 종합")
print("=" * 90)
print(pd.DataFrame(results).to_string(index=False))

val_pred_3g = np.concatenate(all_val_pred)
val_actual_3g = np.concatenate(all_val_actual)
test_pred_3g = np.concatenate(all_test_pred)
test_actual_3g = np.concatenate(all_test_actual)

print("\n" + "=" * 90)
print("3분류 -- 전체 population 종합 성능 (그룹별 예측 합산)")
print("=" * 90)
print("Val(전체):", metrics(val_actual_3g, val_pred_3g))
print("Test(전체):", metrics(test_actual_3g, test_pred_3g))

# baseline: single pooled 7-var model, same population, for exact apples-to-apples
FEATURES_7 = BASE + ["IP_in"]
train_all = pairs[pairs["season_in"] <= 2022]
val_all = pairs[pairs["season_in"] == 2023]
test_all = pairs[pairs["season_in"] == 2024]
scaler = StandardScaler()
X_train = scaler.fit_transform(train_all[FEATURES_7].values)
ridge = RidgeCV(alphas=np.logspace(-2, 3, 60), cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
ridge.fit(X_train, train_all["WAR_out"].values)
X_val, X_test = scaler.transform(val_all[FEATURES_7].values), scaler.transform(test_all[FEATURES_7].values)

print("\n" + "=" * 90)
print("기존 통합모델(7변수, 전체 하나로 학습) -- 같은 population")
print("=" * 90)
print("Val(전체):", metrics(val_all["WAR_out"].values, ridge.predict(X_val)))
print("Test(전체):", metrics(test_all["WAR_out"].values, ridge.predict(X_test)))
