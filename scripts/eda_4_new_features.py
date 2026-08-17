#!/usr/bin/env python3
"""EDA-4: untried feature candidates, workload-spike construction, correlations."""
import pandas as pd
import numpy as np
from scipy import stats

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
TARGET = "kbo_first_year_WAR"

# ---- 2. correlation recheck for untried numeric candidates ----
print("=" * 78)
print("2. 아직 안 써본 숫자형 후보 -- WAR 상관관계")
print("=" * 78)
numeric_candidates = [
    "mlb_ip_last", "mlb_k9_last", "mlb_bb9_last", "mlb_hr9_last", "mlb_fip_minus_last",
    "mlb_ip_3yr", "mlb_fip_3yr", "mlb_k9_3yr", "mlb_bb9_3yr", "mlb_hr9_3yr", "mlb_fip_minus_3yr",
    "mlb_career_ip", "mlb_fip_davenport_translated_last", "mlb_fip_davenport_translated_3yr",
    "aaa_ip_last", "aaa_fip_last", "aaa_k9_last", "aaa_bb9_last", "aaa_fip_minus_last",
    "aaa_ip_3yr", "aaa_fip_3yr", "aaa_k9_3yr", "aaa_hr9_3yr", "aaa_fip_minus_3yr",
    "aaa_career_ip", "aaa_fip_minus_career",
    "aaa_fip_davenport_translated_last", "aaa_fip_davenport_translated_3yr",
    "mlb_ip_share", "mlb_career_war", "mlb_n_seasons_pre_kbo",
]
rows = []
for col in numeric_candidates:
    sub = d[[col, TARGET]].dropna()
    if len(sub) < 3:
        continue
    r, p = stats.pearsonr(sub[col], sub[TARGET])
    rows.append({"column": col, "r": round(r, 3), "p": round(p, 4), "n": len(sub)})
t2 = pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
print(t2.to_string(index=False))

# ---- categorical group comparisons ----
print("\n" + "=" * 78)
print("2b. 범주형 후보 -- 그룹별 WAR 평균/중앙값")
print("=" * 78)

print("\n--- nationality (n>=5인 국적만) ---")
nat_counts = d["nationality"].value_counts()
big_nats = nat_counts[nat_counts >= 5].index
nat_grp = d[d["nationality"].isin(big_nats)].groupby("nationality")[TARGET].agg(["size", "mean", "median", "std"]).round(3)
print(nat_grp.sort_values("mean", ascending=False).to_string())
groups = [d[d["nationality"] == n][TARGET].dropna() for n in big_nats]
f_stat, p_val = stats.f_oneway(*groups)
print(f"ANOVA (국적 그룹간 차이): F={f_stat:.3f}, p={p_val:.4f}")

print("\n--- statcast_metrics_available ---")
sc_grp = d.groupby("statcast_metrics_available")[TARGET].agg(["size", "mean", "median", "std"]).round(3)
print(sc_grp.to_string())
groups_sc = [d[d["statcast_metrics_available"] == g][TARGET].dropna() for g in d["statcast_metrics_available"].dropna().unique()]
f_stat2, p_val2 = stats.f_oneway(*groups_sc)
print(f"ANOVA: F={f_stat2:.3f}, p={p_val2:.4f}")

print("\n--- 실전출전여부 (분산 있는지 확인) ---")
print(d["실전출전여부"].value_counts().to_string())

# ---- 3. workload spike ----
print("\n" + "=" * 78)
print("3. Workload spike (직전시즌 IP / 그전시즌 IP)")
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

spike_rows = []
for _, r in d.iterrows():
    kor = r["선수명"]
    kbo_year = int(r["연도"])
    for level in ("MLB", "AAA"):
        seasons = fg[(fg["선수명"] == kor) & (fg["level"] == level) & (fg["season"] < kbo_year)]
        seasons = seasons.sort_values("season", ascending=False)
        if len(seasons) >= 2:
            last_ip = seasons.iloc[0]["IP_true"]
            prev_ip = seasons.iloc[1]["IP_true"]
            spike = last_ip / prev_ip if prev_ip > 0 else np.nan
            diff = last_ip - prev_ip
        else:
            spike, diff = np.nan, np.nan
        spike_rows.append({"선수명": kor, "level": level, "spike_ratio": spike, "spike_diff": diff})

spike_df = pd.DataFrame(spike_rows)
mlb_spike = spike_df[spike_df["level"] == "MLB"].set_index("선수명")[["spike_ratio", "spike_diff"]]
aaa_spike = spike_df[spike_df["level"] == "AAA"].set_index("선수명")[["spike_ratio", "spike_diff"]]
mlb_spike.columns = ["mlb_workload_spike_ratio", "mlb_workload_spike_diff"]
aaa_spike.columns = ["aaa_workload_spike_ratio", "aaa_workload_spike_diff"]

d2 = d.set_index("선수명").join(mlb_spike).join(aaa_spike).reset_index()

for col in ["mlb_workload_spike_ratio", "mlb_workload_spike_diff", "aaa_workload_spike_ratio", "aaa_workload_spike_diff"]:
    n_valid = d2[col].notna().sum()
    print(f"\n{col}: 유효 표본 {n_valid}/{len(d2)}")
    sub = d2[[col, TARGET]].dropna()
    if len(sub) >= 3:
        r, p = stats.pearsonr(sub[col], sub[TARGET])
        print(f"  vs WAR: r={r:.3f}, p={p:.4f}, n={len(sub)}")
        print(f"  분포: min={sub[col].min():.2f}, median={sub[col].median():.2f}, max={sub[col].max():.2f}")

# persist workload spike columns onto full df for later reuse
full = df.set_index("선수명").join(mlb_spike).join(aaa_spike).reset_index()
full.to_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv", index=False)
print("\n(workload spike 4개 컬럼을 analysis_dataset_v1.csv에 저장함)")
