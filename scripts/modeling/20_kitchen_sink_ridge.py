#!/usr/bin/env python3
"""
Kitchen-sink Ridge: original 8 variables + Davenport-translated K9/BB9/HR9
(MLB and AAA, last pre-KBO season -- level-specific, no unification tie-break
issue since each stays in its own has_record-gated column) + nationality
(one-hot, small-sample countries grouped into "기타"). Height/weight skipped
-- no such data exists anywhere in this project's local files (checked).
Let Ridge's own regularization decide what survives, all added at once.
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


def load_davenport_by_level():
    """Per (player, level, season): combined HR/BB/K/IP (MLE-translated) -> K9/BB9/HR9_DAV."""
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

    by_player_level = {}
    for (name, level, season), c in combined.items():
        by_player_level.setdefault((name, level), []).append({
            "season": season, "IP": c["IP"],
            "K9_DAV": 9 * c["K"] / c["IP"], "BB9_DAV": 9 * c["BB"] / c["IP"], "HR9_DAV": 9 * c["HR"] / c["IP"],
        })
    for key in by_player_level:
        by_player_level[key].sort(key=lambda x: x["season"], reverse=True)
    return by_player_level


dav = load_davenport_by_level()

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)

new_cols = {c: [] for c in ["mlb_k9_dav_last", "mlb_bb9_dav_last", "mlb_hr9_dav_last",
                             "aaa_k9_dav_last", "aaa_bb9_dav_last", "aaa_hr9_dav_last"]}
for _, r in d.iterrows():
    name, kbo_year = r["선수명"], r["연도"]
    for level, prefix in [("MLB", "mlb"), ("AAA", "aaa")]:
        seasons = [s for s in dav.get((name, level), []) if s["season"] < kbo_year]
        if seasons:
            last = seasons[0]
            new_cols[f"{prefix}_k9_dav_last"].append(last["K9_DAV"])
            new_cols[f"{prefix}_bb9_dav_last"].append(last["BB9_DAV"])
            new_cols[f"{prefix}_hr9_dav_last"].append(last["HR9_DAV"])
        else:
            new_cols[f"{prefix}_k9_dav_last"].append(np.nan)
            new_cols[f"{prefix}_bb9_dav_last"].append(np.nan)
            new_cols[f"{prefix}_hr9_dav_last"].append(np.nan)

for c, vals in new_cols.items():
    d[c] = vals

print("=" * 90)
print("신규 Davenport K9/BB9/HR9 커버리지")
print("=" * 90)
for c in new_cols:
    print(f"  {c}: n={d[c].notna().sum()}/{len(d)}")

# nationality: group n<5 into 기타
nat_counts = d["nationality"].value_counts()
big_nats = nat_counts[nat_counts >= 5].index.tolist()
d["nationality_grouped"] = d["nationality"].where(d["nationality"].isin(big_nats), "기타")
print(f"\nnationality_grouped 분포: {d['nationality_grouped'].value_counts().to_dict()}")
nat_dummies = pd.get_dummies(d["nationality_grouped"], prefix="nat", drop_first=True).astype(float)
d = pd.concat([d, nat_dummies], axis=1)
NAT_COLS = nat_dummies.columns.tolist()

# ---------------------------------------------------------------------
# Feature set assembly
# ---------------------------------------------------------------------
ORIG_LEVEL_FEATURES = {"mlb": ["mlb_fip_last", "mlb_fip_minus_career"], "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"]}
NEW_LEVEL_FEATURES = {"mlb": ["mlb_k9_dav_last", "mlb_bb9_dav_last", "mlb_hr9_dav_last"],
                       "aaa": ["aaa_k9_dav_last", "aaa_bb9_dav_last", "aaa_hr9_dav_last"]}
ALL_LEVEL_FEATURES = {"mlb": ORIG_LEVEL_FEATURES["mlb"] + NEW_LEVEL_FEATURES["mlb"],
                       "aaa": ORIG_LEVEL_FEATURES["aaa"] + NEW_LEVEL_FEATURES["aaa"]}
OTHER_FEATURES = ["age_at_kbo_entry", "n_pitch_types_recorded"] + NAT_COLS
BASE_COLS = ["has_mlb_record", "has_aaa_record"]

FEATURES = ALL_LEVEL_FEATURES["mlb"] + ALL_LEVEL_FEATURES["aaa"] + OTHER_FEATURES + BASE_COLS
print(f"\n총 변수 개수: {len(FEATURES)}")
print(FEATURES)

train = d[d["연도"] <= 2023].copy()
val = d[d["연도"] == 2024].copy()
test = d[d["연도"] == 2025].copy()

level_train_means = {}
for level, cols in ALL_LEVEL_FEATURES.items():
    hc = f"has_{level}_record"
    for col in cols:
        level_train_means[col] = float(train.loc[train[hc] == 1, col].mean())
other_train_means = {col: float(train[col].mean()) for col in OTHER_FEATURES if col not in NAT_COLS}


def build(dd):
    dd = dd.copy()
    for level, cols in ALL_LEVEL_FEATURES.items():
        hc = f"has_{level}_record"
        for col in cols:
            dd[col] = dd[col].where(dd[hc] == 1, level_train_means[col])
            dd[col] = dd[col].fillna(level_train_means[col])
    for col in OTHER_FEATURES:
        if col in NAT_COLS:
            dd[col] = dd[col].fillna(0.0)
        else:
            dd[col] = dd[col].fillna(other_train_means[col])
    return dd


train_f = build(train).dropna(subset=[TARGET])
val_f = build(val).dropna(subset=[TARGET])
test_f = build(test).dropna(subset=[TARGET])

scaler = StandardScaler()
X_train = scaler.fit_transform(train_f[FEATURES].values)
y_train = train_f[TARGET].values
X_val, X_test = scaler.transform(val_f[FEATURES].values), scaler.transform(test_f[FEATURES].values)
y_val, y_test = val_f[TARGET].values, test_f[TARGET].values

alphas = np.logspace(-2, 4, 100)
ridge = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
ridge.fit(X_train, y_train)
cv_r2 = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

pred_val = ridge.predict(X_val)
pred_test = ridge.predict(X_test)
m_val = metrics(y_val, pred_val)
m_test = metrics(y_test, pred_test)

print("\n" + "=" * 90)
print("종합 Ridge (kitchen sink) 학습 결과")
print("=" * 90)
print(f"alpha={ridge.alpha_:.3f} (탐색범위 {alphas.min():.3f}~{alphas.max():.1f})")
print(f"CV R2(5fold)={cv_r2.mean():.3f}, std={cv_r2.std():.3f}")
print(f"Val {m_val} | Test {m_test}")

coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values(key=np.abs, ascending=False)
print("\n계수 크기 순위 (절대값 기준):")
print(coefs.round(3).to_string())

# ---------------------------------------------------------------------
# Bootstrap Test R2 CI
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)
n = len(y_test)
boot_r2 = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    yt, yp = y_test[idx], pred_test[idx]
    if np.var(yt) == 0:
        continue
    boot_r2.append(r2_score(yt, yp))
boot_r2 = np.array(boot_r2)
ci_low, ci_high = np.percentile(boot_r2, [2.5, 97.5])
print(f"\n부트스트랩 Test R2 95% CI: [{ci_low:.4f}, {ci_high:.4f}], R2<0 비율={100*(boot_r2<0).mean():.1f}%")

# ---------------------------------------------------------------------
# Comparison vs original 8-var
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("기존 8변수와 비교")
print("=" * 90)
print(f"기존 8변수: Val R2=0.068, Test R2=-0.052, Test 95%CI=[-0.822, 0.286]")
print(f"종합(kitchen sink, {len(FEATURES)}변수): Val R2={m_val['R2']}, Test R2={m_test['R2']}, "
      f"Test 95%CI=[{ci_low:.3f}, {ci_high:.3f}]")

pd.DataFrame({"coef": coefs}).to_csv(f"{ROOT}/reports/modeling/kitchen_sink_ridge_coefs.csv")
print(f"\n저장 완료")
