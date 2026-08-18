#!/usr/bin/env python3
"""
Rebuild the 8-variable Ridge using Davenport-translated (MLE) stats as a
single unified column per feature (most-recent-league priority) instead of
separate MLB/AAA tracks gated by has_mlb_record/has_aaa_record dummies.

Davenport's raw HR/BB/K/IP counts are already MLE-translated (comment in
build_analysis_dataset.py), so FIP/K9/BB9/HR9 computed from them are on a
common scale across levels -- unlike the current mlb_fip_last (raw FanGraphs
MLB FIP) vs aaa_hr9_last (raw AAA HR9), which live on different implicit
scales. IP itself is true decimal in Davenport's data (not baseball
notation) -- confirmed in this project previously, no conversion needed.
"""
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
TARGET = "kbo_first_year_WAR"


def metrics(y_true, y_pred):
    return {"n": len(y_true), "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


def load_fip_constants():
    with open(f"{ROOT}/data/rosters/fangraphs_fip_constants.csv", encoding="utf-8") as f:
        return {int(r["season"]): float(r["cFIP"]) for r in csv.DictReader(f)}


def load_davenport_combined():
    """Per (player, level, season): summed HR/BB/K/IP (already MLE-translated),
    plus FIP_DAV/K9_DAV/BB9_DAV/HR9_DAV computed from those sums."""
    constants = load_fip_constants()
    with open(f"{ROOT}/data/raw/davenport_career_stats.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    combined = {}
    for r in rows:
        level = r["level"]
        if level not in ("MLB", "AAA"):
            continue
        try:
            ip = float(r["IP"])
            hr, bb, k = float(r["HR"]), float(r["BB"]), float(r["K"])
            season = int(r["season"])
        except (ValueError, KeyError):
            continue
        if ip <= 0:
            continue
        key = (r["선수명"], level, season)
        c = combined.setdefault(key, {"IP": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0})
        c["IP"] += ip
        c["HR"] += hr
        c["BB"] += bb
        c["K"] += k

    by_player = {}
    for (name, level, season), c in combined.items():
        const = constants.get(season)
        if const is None:
            continue
        fip_dav = (13 * c["HR"] + 3 * c["BB"] - 2 * c["K"]) / c["IP"] + const
        k9_dav = 9 * c["K"] / c["IP"]
        bb9_dav = 9 * c["BB"] / c["IP"]
        hr9_dav = 9 * c["HR"] / c["IP"]
        by_player.setdefault(name, []).append({
            "level": level, "season": season, "IP": c["IP"],
            "FIP_DAV": fip_dav, "K9_DAV": k9_dav, "BB9_DAV": bb9_dav, "HR9_DAV": hr9_dav,
        })
    for name in by_player:
        by_player[name].sort(key=lambda x: x["season"], reverse=True)
    return by_player


def unified_features(seasons):
    """seasons: all pre-KBO (MLB+AAA combined) Davenport rows for one player,
    sorted desc by season. Returns most-recent-league last/3yr features."""
    if not seasons:
        return {"fip_dav_last": np.nan, "hr9_dav_last": np.nan, "bb9_dav_3yr": np.nan,
                "fip_dav_career": np.nan, "primary_level": None, "primary_last_season": np.nan}

    primary_level = seasons[0]["level"]
    primary_seasons = [s for s in seasons if s["level"] == primary_level]

    last = primary_seasons[0]
    window3 = primary_seasons[:3]
    w_ip = sum(s["IP"] for s in window3)
    bb9_3yr = sum(s["BB9_DAV"] * s["IP"] for s in window3) / w_ip if w_ip > 0 else np.nan

    all_ip = sum(s["IP"] for s in seasons)
    fip_career = sum(s["FIP_DAV"] * s["IP"] for s in seasons) / all_ip if all_ip > 0 else np.nan

    return {
        "fip_dav_last": round(last["FIP_DAV"], 3),
        "hr9_dav_last": round(last["HR9_DAV"], 3),
        "bb9_dav_3yr": round(bb9_3yr, 3) if not np.isnan(bb9_3yr) else np.nan,
        "fip_dav_career": round(fip_career, 3) if not np.isnan(fip_career) else np.nan,
        "primary_level": primary_level,
        "primary_last_season": last["season"],
    }


# ---------------------------------------------------------------------
# Build unified features for all 167 players
# ---------------------------------------------------------------------
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)

dav = load_davenport_combined()

# filter each player's davenport rows to strictly pre-KBO seasons
rows = []
for _, r in d.iterrows():
    name, kbo_year = r["선수명"], r["연도"]
    seasons = [s for s in dav.get(name, []) if s["season"] < kbo_year]
    feats = unified_features(seasons)
    feats["선수명"] = name
    rows.append(feats)

feat_df = pd.DataFrame(rows)
d = d.merge(feat_df, on="선수명", how="left")

print("=" * 90)
print("통일 컬럼 결측/커버리지")
print("=" * 90)
print(f"fip_dav_last 결측: {d['fip_dav_last'].isna().sum()} / {len(d)}")
print(f"primary_level 분포: {d['primary_level'].value_counts(dropna=False).to_dict()}")

# ---------------------------------------------------------------------
# Retrain: original 8var vs Davenport-unified 8var
# ---------------------------------------------------------------------
train = d[d["연도"] <= 2023].copy()
val = d[d["연도"] == 2024].copy()
test = d[d["연도"] == 2025].copy()

ORIG_LEVEL_FEATURES = {"mlb": ["mlb_fip_last", "mlb_fip_minus_career"], "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"]}
ORIG_OTHER = ["age_at_kbo_entry", "n_pitch_types_recorded"]
ORIG_FEATURES = ORIG_LEVEL_FEATURES["mlb"] + ORIG_LEVEL_FEATURES["aaa"] + ORIG_OTHER + ["has_mlb_record", "has_aaa_record"]

UNIFIED_FEATURES = ["fip_dav_last", "fip_dav_career", "hr9_dav_last", "bb9_dav_3yr",
                     "age_at_kbo_entry", "n_pitch_types_recorded", "has_mlb_record", "has_aaa_record"]


def build_orig(dd, train_means, train_other_means):
    dd = dd.copy()
    for level, cols in ORIG_LEVEL_FEATURES.items():
        hc = f"has_{level}_record"
        for col in cols:
            dd[col] = dd[col].where(dd[hc] == 1, train_means[col])
            dd[col] = dd[col].fillna(train_means[col])
    for col in ORIG_OTHER:
        dd[col] = dd[col].fillna(train_other_means[col])
    return dd


def build_unified(dd, train_means):
    dd = dd.copy()
    for col in UNIFIED_FEATURES:
        if col in ("has_mlb_record", "has_aaa_record"):
            continue
        dd[col] = dd[col].fillna(train_means[col])
    return dd


train_tmp = train.copy()
orig_train_means = {}
for level, cols in ORIG_LEVEL_FEATURES.items():
    hc = f"has_{level}_record"
    for col in cols:
        orig_train_means[col] = float(train_tmp.loc[train_tmp[hc] == 1, col].mean())
orig_train_other_means = {col: float(train_tmp[col].mean()) for col in ORIG_OTHER}

unified_train_means = {col: float(train_tmp[col].mean()) for col in UNIFIED_FEATURES if col not in ("has_mlb_record", "has_aaa_record")}

results = []
for label, features, builder in [
    ("기존 8변수(MLB/AAA 분리)", ORIG_FEATURES, lambda dd: build_orig(dd, orig_train_means, orig_train_other_means)),
    ("Davenport 통일 8변수", UNIFIED_FEATURES, lambda dd: build_unified(dd, unified_train_means)),
]:
    train_f = builder(train).dropna(subset=[TARGET])
    val_f = builder(val).dropna(subset=[TARGET])
    test_f = builder(test).dropna(subset=[TARGET])

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
    coefs = dict(zip(features, np.round(ridge.coef_, 3)))

    print(f"\n=== {label} ===")
    print(f"n_train={len(train_f)}, alpha={ridge.alpha_:.3f}, CV R2(5fold)={cv_r2.mean():.3f}")
    print(f"Val {m_val} | Test {m_test}")
    print(f"계수: {coefs}")
    pred_val = ridge.predict(X_val)
    print(f"예측범위(Val): {pred_val.min():.2f}~{pred_val.max():.2f}")

    results.append({"model": label, "n_train": len(train_f), "alpha": round(ridge.alpha_, 3),
                     "cv_R2": round(cv_r2.mean(), 3), "Val_R2": m_val["R2"], "Val_MAE": m_val["MAE"],
                     "Test_R2": m_test["R2"], "Test_MAE": m_test["MAE"]})

print("\n" + "=" * 90)
print("종합 비교")
print("=" * 90)
print(pd.DataFrame(results).to_string(index=False))

pd.DataFrame(results).to_csv(f"{ROOT}/reports/modeling/davenport_unified_comparison.csv", index=False)
d.to_csv(f"{ROOT}/reports/modeling/davenport_unified_features.csv", index=False)
print(f"\n저장 완료")
