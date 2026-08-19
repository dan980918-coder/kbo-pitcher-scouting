#!/usr/bin/env python3
"""
Apply empirical-Bayes shrinkage to FIP_in on top of the 3-way role model
(§11.4): FIP_shrunk = (FIP_in*IP_in + league_avg_FIP*k) / (IP_in+k),
k in {60, 100} (common FIP-stabilization range). League average FIP is the
IP-weighted mean of ALL pitchers in that season_in year (already-local
statiz_bulk data, no new collection).

For the pure-bullpen group, also tries a games-weighted variant (G_in
instead of IP_in as the reliability denominator) since that group's actual
regressor is G_in, not IP_in -- exploratory, reported alongside the
standard IP-weighted version.
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


def ip_true(ip):
    whole = int(ip)
    frac = round((ip - whole) * 10)
    return whole + frac / 3.0


# ---------------------------------------------------------------------
# League-average FIP per season_in year (IP-weighted, full population)
# ---------------------------------------------------------------------
season = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_pitching_2015_2025_all.csv")
season["IP_true"] = season["IP"].apply(ip_true)
season = season[season["IP_true"] > 0]
league_avg_fip = season.groupby("year").apply(
    lambda g: (g["FIP"] * g["IP_true"]).sum() / g["IP_true"].sum(), include_groups=False)
print("연도별 리그평균FIP(IP가중):")
print(league_avg_fip.round(3).to_string())

# ---------------------------------------------------------------------
# Build pairs, attach league avg FIP for season_in year
# ---------------------------------------------------------------------
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
pairs["league_avg_fip"] = pairs["season_in"].map(league_avg_fip)

for k in [60, 100]:
    pairs[f"fip_shrunk_ip_k{k}"] = (pairs["FIP_in"] * pairs["IP_in"] + pairs["league_avg_fip"] * k) / (pairs["IP_in"] + k)
    pairs[f"fip_shrunk_g_k{k}"] = (pairs["FIP_in"] * pairs["G_in"] + pairs["league_avg_fip"] * k) / (pairs["G_in"] + k)

CONFIGS = {
    "순수불펜(GS/G=0)": ["G_in"],
    "순수선발(GS/G>=0.9)": ["IP_in"],
    "스윙맨(0<GS/G<0.9)": ["IP_in", "G_in"],
}
OTHER_FEATS = ["WHIP_in", "K9_in", "BB9_in", "WAR_in", "ball_era"]


def fit_eval(g, fip_col, workload_cols):
    features = [fip_col] + OTHER_FEATS + workload_cols
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
    return cv_r2.mean(), metrics(val["WAR_out"].values, pred_val), metrics(test["WAR_out"].values, pred_test), pred_val, val["WAR_out"].values, pred_test, test["WAR_out"].values


print("\n" + "=" * 100)
print("1-2. 그룹별 FIP_in vs FIP_shrunk(k=60,100) 재학습 비교")
print("=" * 100)

best_choice = {}
for role, workload_cols in CONFIGS.items():
    g = pairs[pairs["role3"] == role]
    print(f"\n--- {role} ---")
    variants = [("FIP_in(원본)", "FIP_in")]
    for k in [60, 100]:
        variants.append((f"FIP_shrunk_IP_k{k}", f"fip_shrunk_ip_k{k}"))
    if role == "순수불펜(GS/G=0)":
        for k in [60, 100]:
            variants.append((f"FIP_shrunk_G_k{k}(탐색적)", f"fip_shrunk_g_k{k}"))

    role_results = []
    for label, col in variants:
        cv_r2, m_val, m_test, *_ = fit_eval(g, col, workload_cols)
        print(f"  {label}: CV R2={cv_r2:.3f}, Val R2={m_val['R2']}, Test R2={m_test['R2']}")
        role_results.append((label, col, cv_r2, m_val["R2"], m_test["R2"]))

    best = max(role_results, key=lambda x: x[3])  # best by Val R2
    best_choice[role] = best[1]
    print(f"  -> 최선(Val R2 기준): {best[0]}")

# ---------------------------------------------------------------------
# 3. Combined population check with best-per-group FIP column
# ---------------------------------------------------------------------
print("\n" + "=" * 100)
print("3. 각 그룹 최선 버전으로 종합 population 성능 재확인")
print("=" * 100)

all_val_pred, all_val_actual, all_test_pred, all_test_actual = [], [], [], []
for role, workload_cols in CONFIGS.items():
    g = pairs[pairs["role3"] == role]
    fip_col = best_choice[role]
    _, m_val, m_test, pred_val, y_val, pred_test, y_test = fit_eval(g, fip_col, workload_cols)
    print(f"{role} (사용 FIP컬럼={fip_col}): Val {m_val} | Test {m_test}")
    all_val_pred.append(pred_val); all_val_actual.append(y_val)
    all_test_pred.append(pred_test); all_test_actual.append(y_test)

val_pred_all = np.concatenate(all_val_pred); val_actual_all = np.concatenate(all_val_actual)
test_pred_all = np.concatenate(all_test_pred); test_actual_all = np.concatenate(all_test_actual)

print("\n종합 Val:", metrics(val_actual_all, val_pred_all))
print("종합 Test:", metrics(test_actual_all, test_pred_all))
print("\n(비교 기준) §11.4 기존 3분류(shrinkage 없음): Val R2=0.394, Test R2=0.374")
