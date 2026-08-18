#!/usr/bin/env python3
"""
Robustness checks on the KBO-wide "this season -> next season WAR" Ridge:
1. Walk-forward validation across 5 independent year-transitions (train on
   everything strictly before, test on that one transition) -- is the
   Val/Test R2~0.35-0.36 reported for 2023/2024 a one-off or reproducible?
2. Input window comparison: last-1-season features vs MARCEL-style IP-weighted
   3-season features (same last/3yr pattern used in the foreign-pitcher
   project), on the standard fixed split.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
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
# Build clean per-season table
# ---------------------------------------------------------------------
season = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_pitching_2015_2025_all.csv")
season["IP_true"] = season["IP"].apply(ip_true)
season = season[season["IP_true"] > 0].copy()
season["K9"] = 9 * season["SO"] / season["IP_true"]
season["BB9"] = 9 * season["BB"] / season["IP_true"]
season = season.sort_values(["p_no", "year"])

by_pno = {pno: g.sort_values("year") for pno, g in season.groupby("p_no")}

FEATURES_7 = ["FIP_in", "WHIP_in", "K9_in", "BB9_in", "IP_in", "WAR_in", "ball_era"]

# ---------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------
def build_pairs_last1():
    pairs = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_season_pairs_2015_2025.csv")
    pairs = pairs[~((pairs["season_in"] == 2018) & (pairs["season_out"] == 2019))].copy()
    pairs = pairs.dropna(subset=FEATURES_7 + ["WAR_out"])
    pairs = pairs[pairs["IP_in"] > 0]
    return pairs


def build_pairs_window(n_seasons):
    """For every (p_no, target_year) with a played target_year, use the up-to-n
    prior seasons (whatever is available, most recent first) as input window."""
    rows = []
    for pno, g in by_pno.items():
        years = g["year"].tolist()
        for i, ty in enumerate(years):
            if i == 0:
                continue  # no prior season at all
            window = g.iloc[max(0, i - n_seasons):i].sort_values("year", ascending=False)
            if len(window) == 0:
                continue
            total_ip = window["IP_true"].sum()
            fip = (window["FIP"] * window["IP_true"]).sum() / total_ip
            whip = (window["WHIP"] * window["IP_true"]).sum() / total_ip
            k9 = 9 * window["SO"].sum() / total_ip
            bb9 = 9 * window["BB"].sum() / total_ip
            war_sum = window["WAR"].sum()
            last_season_in = int(window["year"].max())
            rows.append({
                "p_no": pno, "name": g.iloc[i]["name"], "season_in": last_season_in, "season_out": int(ty),
                "n_seasons_used": len(window),
                "FIP_in": fip, "WHIP_in": whip, "K9_in": k9, "BB9_in": bb9, "IP_in": total_ip, "WAR_in": war_sum,
                "ball_era": 1 if last_season_in >= 2019 else 0,
                "WAR_out": g.iloc[i]["WAR"],
            })
    df = pd.DataFrame(rows)
    df = df[df["season_out"] != 2019]  # any window predicting into 2019 crosses the ball-era seam, same exclusion principle as the 1yr version
    return df


def fit_ridge(train, features):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].values)
    y_train = train["WAR_out"].values
    alphas = np.logspace(-2, 3, 60)
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=cv5, scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)
    return ridge, scaler


# ---------------------------------------------------------------------
# 1. Walk-forward across 5 independent year-transitions
# ---------------------------------------------------------------------
print("=" * 90)
print("1. Walk-forward validation (직전 시즌 1개 입력, 매번 그 이전 데이터로만 학습)")
print("=" * 90)

pairs1 = build_pairs_last1()
wf_rows = []
for target_out in [2021, 2022, 2023, 2024, 2025]:
    train = pairs1[pairs1["season_out"] < target_out]
    test = pairs1[pairs1["season_out"] == target_out]
    if len(test) == 0:
        continue
    ridge, scaler = fit_ridge(train, FEATURES_7)
    X_test = scaler.transform(test[FEATURES_7].values)
    pred = ridge.predict(X_test)
    m = metrics(test["WAR_out"].values, pred)
    m["target_year"] = target_out
    m["alpha"] = round(ridge.alpha_, 3)
    m["n_train"] = len(train)
    wf_rows.append(m)

wf_df = pd.DataFrame(wf_rows)[["target_year", "n_train", "n", "alpha", "MAE", "RMSE", "R2"]]
print(wf_df.to_string(index=False))
print(f"\nR2 평균: {wf_df['R2'].mean():.3f}, 표준편차: {wf_df['R2'].std():.3f}, 범위: {wf_df['R2'].min():.3f}~{wf_df['R2'].max():.3f}")

# ---------------------------------------------------------------------
# 2. Input window comparison: last-1-season vs IP-weighted 3-season
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. 입력 시간창 비교: 직전 1개 시즌 vs 최근 3개 시즌 가중평균(MARCEL 방식)")
print("=" * 90)

pairs3 = build_pairs_window(3)
pairs3 = pairs3.dropna(subset=FEATURES_7 + ["WAR_out"])
print(f"3yr 윈도우 학습쌍: {len(pairs3)} (1yr 버전 {len(pairs1)}개와 비교)")
print(f"윈도우 내 실제 사용 시즌 수 분포: {pairs3['n_seasons_used'].value_counts().sort_index().to_dict()}")

for label, pairs in [("1yr(직전 1개 시즌)", pairs1), ("3yr(최근 3개 가중평균)", pairs3)]:
    train = pairs[pairs["season_in"] <= 2022] if "season_in" in pairs.columns else pairs[pairs["season_out"] <= 2023]
    val = pairs[pairs["season_out"] == 2024]
    test = pairs[pairs["season_out"] == 2025]
    ridge, scaler = fit_ridge(train, FEATURES_7)
    X_val, X_test = scaler.transform(val[FEATURES_7].values), scaler.transform(test[FEATURES_7].values)
    m_val = metrics(val["WAR_out"].values, ridge.predict(X_val))
    m_test = metrics(test["WAR_out"].values, ridge.predict(X_test))
    cv_r2 = cross_val_score(Ridge(alpha=ridge.alpha_), scaler.transform(train[FEATURES_7].values), train["WAR_out"].values, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    print(f"\n{label}: n_train={len(train)}, alpha={ridge.alpha_:.3f}")
    print(f"  CV R2(5fold)={cv_r2.mean():.3f} | Val(2024) {m_val} | Test(2025) {m_test}")

print(f"\n저장 없음 (콘솔 출력만; 필요시 재요청)")
