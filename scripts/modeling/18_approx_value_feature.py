#!/usr/bin/env python3
"""
b. approx_value: a rough cumulative "value" stat = (league_avg_FIP - player_FIP) x IP / 10 (runs-per-win ~10).
MLB: use mlb_career_war directly (FanGraphs, already computed, no approximation needed).
AAA: no career WAR available, so approximate from aaa_fip_minus_career (already a
league-relative index) x aaa_career_ip. League-avg FIP per player is backed out from
their own last-season aaa_fip_last / aaa_fip_minus_last ratio where available (more
precise than a flat league constant); falls back to a flat 4.20 AAA-average FIP assumption
otherwise (stated explicitly -- this whole feature is an approximation exercise).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
TARGET = "kbo_first_year_WAR"
FALLBACK_AAA_LEAGUE_FIP = 4.20
RUNS_PER_WIN = 10.0


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)

# AAA approx_value
implied_league_fip = d["aaa_fip_last"] / (d["aaa_fip_minus_last"] / 100)
implied_league_fip = implied_league_fip.where((d["aaa_fip_minus_last"] > 0) & implied_league_fip.notna(), FALLBACK_AAA_LEAGUE_FIP)
raw_diff_per9 = (100 - d["aaa_fip_minus_career"]) / 100 * implied_league_fip
d["aaa_approx_value"] = raw_diff_per9 * (d["aaa_career_ip"] / 9) / RUNS_PER_WIN

print("=" * 90)
print("approx_value 분포 확인")
print("=" * 90)
print(f"mlb_career_war: n={d['mlb_career_war'].notna().sum()}, "
      f"범위 {d['mlb_career_war'].min():.2f}~{d['mlb_career_war'].max():.2f}, 평균 {d['mlb_career_war'].mean():.2f}")
print(f"aaa_approx_value: n={d['aaa_approx_value'].notna().sum()}, "
      f"범위 {d['aaa_approx_value'].min():.2f}~{d['aaa_approx_value'].max():.2f}, 평균 {d['aaa_approx_value'].mean():.2f}")
print(f"(참고) implied_league_fip 사용 가능 비율: {((d['aaa_fip_minus_last']>0)).sum()}/{len(d)}, "
      f"나머지는 고정값 {FALLBACK_AAA_LEAGUE_FIP} 사용")

# unified approx_value: MLB priority (has_mlb_record), else AAA approx
d["approx_value_unified"] = np.where(d["has_mlb_record"] == 1, d["mlb_career_war"], d["aaa_approx_value"])

corr_sub = d[["approx_value_unified", TARGET]].dropna()
from scipy import stats
r, p = stats.pearsonr(corr_sub["approx_value_unified"], corr_sub[TARGET])
print(f"\napprox_value_unified vs {TARGET}: r={r:.3f}, p={p:.4f}, n={len(corr_sub)}")

# ---------------------------------------------------------------------
# Add to (1) original 8var and (2) Davenport-unified 8var
# ---------------------------------------------------------------------
ORIG_LEVEL_FEATURES = {"mlb": ["mlb_fip_last", "mlb_fip_minus_career"], "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"]}
ORIG_OTHER = ["age_at_kbo_entry", "n_pitch_types_recorded"]
ORIG_FEATURES = ORIG_LEVEL_FEATURES["mlb"] + ORIG_LEVEL_FEATURES["aaa"] + ORIG_OTHER + ["has_mlb_record", "has_aaa_record"]
ORIG_PLUS_AV = ORIG_FEATURES + ["approx_value_unified"]

dav_feat = pd.read_csv(f"{ROOT}/reports/modeling/davenport_unified_features.csv")
dav_feat = dav_feat.merge(d[["선수명", "approx_value_unified"]], on="선수명", how="left")
UNIFIED_FEATURES = ["fip_dav_last", "fip_dav_career", "hr9_dav_last", "bb9_dav_3yr",
                     "age_at_kbo_entry", "n_pitch_types_recorded", "has_mlb_record", "has_aaa_record"]
UNIFIED_PLUS_AV = UNIFIED_FEATURES + ["approx_value_unified"]


def run(dataset, features, label, train_impute_cols, level_features=None):
    train = dataset[dataset["연도"] <= 2023].copy()
    val = dataset[dataset["연도"] == 2024].copy()
    test = dataset[dataset["연도"] == 2025].copy()
    train_means = {c: float(train[c].mean()) for c in train_impute_cols}

    level_train_means = {}
    if level_features:
        for level, cols in level_features.items():
            hc = f"has_{level}_record"
            for col in cols:
                level_train_means[col] = float(train.loc[train[hc] == 1, col].mean())

    def build(dd):
        dd = dd.copy()
        if level_features:
            for level, cols in level_features.items():
                hc = f"has_{level}_record"
                for col in cols:
                    dd[col] = dd[col].where(dd[hc] == 1, level_train_means[col])
                    dd[col] = dd[col].fillna(level_train_means[col])
        for c in train_impute_cols:
            dd[c] = dd[c].fillna(train_means[c])
        return dd

    train_f = build(train).dropna(subset=[TARGET])
    val_f = build(val).dropna(subset=[TARGET])
    test_f = build(test).dropna(subset=[TARGET])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_f[features].values)
    y_train = train_f[TARGET].values
    y_val, y_test = val_f[TARGET].values, test_f[TARGET].values
    X_val, X_test = scaler.transform(val_f[features].values), scaler.transform(test_f[features].values)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 60), cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)
    cv_r2 = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    m_val = metrics(y_val, ridge.predict(X_val))
    m_test = metrics(y_test, ridge.predict(X_test))
    print(f"\n=== {label} ===")
    print(f"alpha={ridge.alpha_:.3f}, CV R2={cv_r2.mean():.3f}")
    print(f"Val {m_val} | Test {m_test}")
    coefs = dict(zip(features, np.round(ridge.coef_, 3)))
    print(f"계수: {coefs}")
    return {"model": label, "cv_R2": round(cv_r2.mean(), 3), "Val_R2": m_val["R2"], "Test_R2": m_test["R2"]}


print("\n" + "=" * 90)
print("기존 8변수 계열")
print("=" * 90)
res1 = run(d, ORIG_FEATURES, "기존 8변수(기준)", ORIG_OTHER, level_features=ORIG_LEVEL_FEATURES)
res2 = run(d, ORIG_PLUS_AV, "+approx_value_unified(9변수)", ORIG_OTHER + ["approx_value_unified"], level_features=ORIG_LEVEL_FEATURES)

print("\n" + "=" * 90)
print("Davenport 통일 8변수 계열")
print("=" * 90)
dav_other = ["age_at_kbo_entry", "n_pitch_types_recorded", "fip_dav_last", "fip_dav_career", "hr9_dav_last", "bb9_dav_3yr"]
res3 = run(dav_feat, UNIFIED_FEATURES, "Davenport 통일 8변수(기준)", dav_other)
res4 = run(dav_feat, UNIFIED_PLUS_AV, "+approx_value_unified(9변수)", dav_other + ["approx_value_unified"])

print("\n" + "=" * 90)
print("종합 비교")
print("=" * 90)
print(pd.DataFrame([res1, res2, res3, res4]).to_string(index=False))
