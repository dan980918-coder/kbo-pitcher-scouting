#!/usr/bin/env python3
"""
d. Pitch-arsenal clustering (unsupervised) -- do MLB Statcast pitch-mix
archetypes relate to KBO first-year WAR / Translation Gap?

Uses pitch_arsenal_full.csv (Baseball Savant, already local, no STATIZ
dependency). Restricted to statcast_metrics_available=='full' (95 players)
since usage%/whiff%/run_value only exist at that tier.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

pa = pd.read_csv(f"{ROOT}/data/rosters/pitch_arsenal_full.csv")
pa = pa[pa["statcast_metrics_available"] == "full"].copy()

FASTBALL = {"FF", "SI", "FC"}
BREAKING = {"SL", "CU", "ST", "SV", "KC"}
OFFSPEED = {"CH", "FS"}


def bucket(pt):
    if pt in FASTBALL:
        return "fastball"
    if pt in BREAKING:
        return "breaking"
    if pt in OFFSPEED:
        return "offspeed"
    return "other"


pa["bucket"] = pa["pitch_type"].apply(bucket)

rows = []
for name, g in pa.groupby("선수명"):
    total_usage = g["pitch_usage_pct"].sum()
    if total_usage <= 0:
        continue
    bucket_usage = g.groupby("bucket")["pitch_usage_pct"].sum()
    # primary (highest-usage) fastball velocity as the "stuff" signal
    ff_rows = g[g["pitch_type"].isin(FASTBALL)]
    primary_velo = ff_rows.sort_values("pitch_usage_pct", ascending=False)["avg_velocity"].iloc[0] if len(ff_rows) else np.nan
    whiff_w = (g["whiff_pct"] * g["pitch_usage_pct"]).sum() / total_usage
    rv_w = (g["run_value_per_100"] * g["pitch_usage_pct"]).sum() / total_usage
    n_types = g["pitch_type"].nunique()
    rows.append({
        "선수명": name,
        "fastball_pct": bucket_usage.get("fastball", 0.0),
        "breaking_pct": bucket_usage.get("breaking", 0.0),
        "offspeed_pct": bucket_usage.get("offspeed", 0.0),
        "primary_fb_velo": primary_velo,
        "whiff_pct_weighted": whiff_w,
        "run_value_weighted": rv_w,
        "n_pitch_types": n_types,
    })

feat = pd.DataFrame(rows).dropna()
print(f"군집분석 대상: {len(feat)}명 (통계 full tier {pa['선수명'].nunique()}명 중 결측 제외)")

CLUSTER_FEATURES = ["fastball_pct", "breaking_pct", "offspeed_pct", "primary_fb_velo", "whiff_pct_weighted", "n_pitch_types"]
scaler = StandardScaler()
X = scaler.fit_transform(feat[CLUSTER_FEATURES].values)

print("\n" + "=" * 90)
print("k별 inertia (elbow 참고용)")
print("=" * 90)
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    print(f"k={k}: inertia={km.inertia_:.1f}")

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
feat["cluster"] = km.labels_

print("\n" + "=" * 90)
print(f"k={K} 군집별 특징 (원 단위 평균)")
print("=" * 90)
summary = feat.groupby("cluster")[CLUSTER_FEATURES + []].mean().round(2)
summary["n"] = feat.groupby("cluster").size()
print(summary.to_string())

# ---------------------------------------------------------------------
# Merge with WAR / Translation Gap
# ---------------------------------------------------------------------
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1][["선수명", "연도", "kbo_first_year_WAR", "translation_gap"]].copy()
d["translation_gap"] = pd.to_numeric(d["translation_gap"], errors="coerce")

merged = feat.merge(d, on="선수명", how="left").dropna(subset=["kbo_first_year_WAR"])
print(f"\nWAR/Gap 매칭된 인원: {len(merged)}명")

print("\n" + "=" * 90)
print("군집별 KBO 첫해 WAR / Translation Gap")
print("=" * 90)
grp = merged.groupby("cluster")[["kbo_first_year_WAR", "translation_gap"]].agg(["size", "mean", "median", "std"])
print(grp.round(3).to_string())

groups_war = [merged[merged["cluster"] == c]["kbo_first_year_WAR"].dropna() for c in sorted(merged["cluster"].unique())]
f_war, p_war = stats.f_oneway(*groups_war)
print(f"\nANOVA (군집별 WAR 차이): F={f_war:.3f}, p={p_war:.4f}")

groups_gap = [merged[merged["cluster"] == c]["translation_gap"].dropna() for c in sorted(merged["cluster"].unique())]
groups_gap = [g for g in groups_gap if len(g) > 0]
if len(groups_gap) >= 2:
    f_gap, p_gap = stats.f_oneway(*groups_gap)
    print(f"ANOVA (군집별 Translation Gap 차이): F={f_gap:.3f}, p={p_gap:.4f}")

feat.to_csv(f"{ROOT}/reports/modeling/pitch_arsenal_clusters.csv", index=False)
merged.to_csv(f"{ROOT}/reports/modeling/pitch_arsenal_clusters_with_outcomes.csv", index=False)
print(f"\n저장 완료")
