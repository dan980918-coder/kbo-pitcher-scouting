#!/usr/bin/env python3
"""EDA-5: clean up workload_spike (log-transform + drop tiny-denominator cases),
check n_pitch_types_recorded vs statcast_metrics_available redundancy, then
retrain Ridge with candidate new features and compare against the 8-feature
baseline on Val (2024).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
TARGET = "kbo_first_year_WAR"

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()

# =====================================================================
# 1. workload_spike 정리: prev_ip < 5이닝이면 결측 처리, 나머지는 log(ratio)
# =====================================================================
print("=" * 78)
print("1. mlb_workload_spike 정리")
print("=" * 78)

fg = pd.read_csv(f"{ROOT}/data/raw/fangraphs_career_stats.csv")
fg = fg[fg["is_split_row"] == 0].copy()


def ip_true_decimal(v):
    if pd.isna(v):
        return np.nan
    whole = int(v)
    frac = round((v - whole) * 10)
    return whole + frac / 3.0


fg["IP_true"] = fg["IP"].apply(ip_true_decimal)

MIN_PREV_IP = 5.0
spike_rows = []
for _, r in df.iterrows():
    kor = r["선수명"]
    kbo_year = r["연도"]
    if pd.isna(kbo_year):
        continue
    kbo_year = int(kbo_year)
    for level in ("MLB", "AAA"):
        seasons = fg[(fg["선수명"] == kor) & (fg["level"] == level) & (fg["season"] < kbo_year)]
        seasons = seasons.sort_values("season", ascending=False)
        if len(seasons) >= 2:
            last_ip = seasons.iloc[0]["IP_true"]
            prev_ip = seasons.iloc[1]["IP_true"]
            if prev_ip < MIN_PREV_IP:
                # denominator too small -> ratio is an artifact, treat as missing
                ratio, log_ratio, diff = np.nan, np.nan, np.nan
            else:
                ratio = last_ip / prev_ip
                log_ratio = np.log(ratio)
                diff = last_ip - prev_ip
        else:
            ratio, log_ratio, diff = np.nan, np.nan, np.nan
        spike_rows.append({"선수명": kor, "level": level, "spike_ratio": ratio,
                            "spike_log": log_ratio, "spike_diff": diff})

spike_df = pd.DataFrame(spike_rows)
mlb_spike = spike_df[spike_df["level"] == "MLB"].set_index("선수명")[["spike_ratio", "spike_log", "spike_diff"]]
aaa_spike = spike_df[spike_df["level"] == "AAA"].set_index("선수명")[["spike_ratio", "spike_log", "spike_diff"]]
mlb_spike.columns = ["mlb_workload_spike_ratio", "mlb_workload_spike_log", "mlb_workload_spike_diff"]
aaa_spike.columns = ["aaa_workload_spike_ratio", "aaa_workload_spike_log", "aaa_workload_spike_diff"]

# drop old (unfiltered) spike columns before rejoining cleaned versions
old_cols = ["mlb_workload_spike_ratio", "mlb_workload_spike_diff", "mlb_workload_spike_log",
            "aaa_workload_spike_ratio", "aaa_workload_spike_diff", "aaa_workload_spike_log"]
df = df.drop(columns=[c for c in old_cols if c in df.columns])
df = df.set_index("선수명").join(mlb_spike).join(aaa_spike).reset_index()

d = df[df["kbo_no_appearance"] != 1].copy()

for col in ["mlb_workload_spike_ratio", "mlb_workload_spike_log", "mlb_workload_spike_diff"]:
    n_valid = d[col].notna().sum()
    sub = d[[col, TARGET]].dropna()
    r, p = stats.pearsonr(sub[col], sub[TARGET]) if len(sub) >= 3 else (np.nan, np.nan)
    print(f"{col}: n={n_valid}/{len(d)}  r={r:.3f}  p={p:.4f}")

print(f"\n(prev_ip<{MIN_PREV_IP}이닝으로 결측 처리된 건: "
      f"{spike_df[(spike_df['level']=='MLB')]['spike_ratio'].isna().sum()} "
      f"(원래 유효 119명 대비))")

# =====================================================================
# 2. n_pitch_types_recorded vs statcast_metrics_available 중복성 체크
# =====================================================================
print("\n" + "=" * 78)
print("2. n_pitch_types_recorded vs statcast_metrics_available")
print("=" * 78)
grp = d.groupby("statcast_metrics_available")["n_pitch_types_recorded"].agg(["size", "mean", "median", "std"]).round(3)
print(grp.to_string())

groups = [d[d["statcast_metrics_available"] == g]["n_pitch_types_recorded"].dropna()
          for g in d["statcast_metrics_available"].dropna().unique()]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\nANOVA (statcast_metrics_available 그룹별 n_pitch_types_recorded 차이): F={f_stat:.3f}, p={p_val:.4f}")

# eta-squared style: how much of statcast group is "explained" by pitch-type count alone
corr_sub = d[["n_pitch_types_recorded", TARGET]].dropna()
r_pt, p_pt = stats.pearsonr(corr_sub["n_pitch_types_recorded"], corr_sub[TARGET])
print(f"\n참고: n_pitch_types_recorded vs WAR r={r_pt:.3f} (이미 Ridge에 포함된 변수)")

# =====================================================================
# 3 & 4. Ridge 재학습: 기존 8개 + mlb_ip_last + workload_spike_log(선택) + statcast/pitch_types(선택)
# =====================================================================
print("\n" + "=" * 78)
print("3-4. Ridge 재학습 및 비교")
print("=" * 78)

# redundancy check result decides this at run time (see printed ANOVA above);
# statcast_metrics_available is a 3-level categorical while n_pitch_types_recorded already
# captures overlapping information as a continuous var already in the model -> keep pitch_types,
# drop statcast_metrics_available to avoid redundancy/collinearity.
USE_STATCAST_CATEGORY = False

LEVEL_FEATURES_BASE = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career"],
    "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES_BASE = ["age_at_kbo_entry", "n_pitch_types_recorded"]

# new features
LEVEL_FEATURES_NEW = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career", "mlb_ip_last"],
    "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES_NEW = ["age_at_kbo_entry", "n_pitch_types_recorded", "mlb_workload_spike_log"]

LEVEL_FEATURES_IPONLY = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career", "mlb_ip_last"],
    "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES_IPONLY = ["age_at_kbo_entry", "n_pitch_types_recorded"]

FEATURE_SETS = {
    "baseline_8var": (LEVEL_FEATURES_BASE, OTHER_FEATURES_BASE),
    "extended_new": (LEVEL_FEATURES_NEW, OTHER_FEATURES_NEW),
    "ip_last_only_9var": (LEVEL_FEATURES_IPONLY, OTHER_FEATURES_IPONLY),
}


def build_features(d_in, level_features, other_features, train_means, train_other_means):
    dd = d_in.copy()
    dd["has_mlb_record"] = (dd["mlb_career_ip"].fillna(0) > 0).astype(int)
    dd["has_aaa_record"] = (dd["aaa_career_ip"].fillna(0) > 0).astype(int)
    for level, cols in level_features.items():
        has_col = f"has_{level}_record"
        for col in cols:
            dd[col] = dd[col].where(dd[has_col] == 1, train_means[col])
            dd[col] = dd[col].fillna(train_means[col])
    for col in other_features:
        dd[col] = dd[col].fillna(train_other_means[col])
    return dd


modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()

results = []
pred_ranges = {}
for name, (level_features, other_features) in FEATURE_SETS.items():
    feature_cols = level_features["mlb"] + level_features["aaa"] + other_features + ["has_mlb_record", "has_aaa_record"]

    train_tmp = train.copy()
    train_tmp["has_mlb_record_tmp"] = (train_tmp["mlb_career_ip"].fillna(0) > 0).astype(int)
    train_tmp["has_aaa_record_tmp"] = (train_tmp["aaa_career_ip"].fillna(0) > 0).astype(int)
    train_means = {}
    for level, cols in level_features.items():
        has_col = f"has_{level}_record_tmp"
        for col in cols:
            train_means[col] = float(train_tmp.loc[train_tmp[has_col] == 1, col].mean())
    train_other_means = {col: float(train_tmp[col].mean()) for col in other_features}

    train_f = build_features(train, level_features, other_features, train_means, train_other_means)
    train_fit = train_f.dropna(subset=[TARGET])
    val_f = build_features(val, level_features, other_features, train_means, train_other_means)
    val_fit = val_f.dropna(subset=[TARGET])

    X_train_raw = train_fit[feature_cols].values
    y_train = train_fit[TARGET].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    alphas = np.logspace(-2, 3, 60)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
    ridge.fit(X_train, y_train)

    X_val = scaler.transform(val_fit[feature_cols].values)
    pred_train = ridge.predict(X_train)
    pred_val = ridge.predict(X_val)

    def metrics(y_true, y_pred):
        return {"MAE": round(mean_absolute_error(y_true, y_pred), 3),
                "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
                "R2": round(r2_score(y_true, y_pred), 3)}

    train_m = metrics(y_train, pred_train)
    val_m = metrics(val_fit[TARGET].values, pred_val)

    results.append({"model": name, "alpha": round(float(ridge.alpha_), 4), "n_features": len(feature_cols),
                     "n_train": len(train_fit), "n_val": len(val_fit),
                     "Train_MAE": train_m["MAE"], "Train_R2": train_m["R2"],
                     "Val_MAE": val_m["MAE"], "Val_RMSE": val_m["RMSE"], "Val_R2": val_m["R2"],
                     "gap(Train-Val R2)": round(train_m["R2"] - val_m["R2"], 3)})
    pred_ranges[name] = {"pred_min": round(float(pred_val.min()), 3), "pred_max": round(float(pred_val.max()), 3),
                          "pred_std": round(float(pred_val.std()), 3), "features": feature_cols}

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

print("\n--- Val 예측값 범위 비교 ---")
for name, r in pred_ranges.items():
    print(f"{name}: min={r['pred_min']}, max={r['pred_max']}, std={r['pred_std']}, features={r['features']}")

# persist cleaned workload-spike columns (log/ratio/diff) back to csv
df.to_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv", index=False)
print("\n(정리된 workload_spike 컬럼들을 analysis_dataset_v1.csv에 저장함)")
