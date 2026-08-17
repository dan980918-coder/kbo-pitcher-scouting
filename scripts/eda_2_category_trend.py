#!/usr/bin/env python3
"""EDA-2: category/trend comparisons. Text/table output only."""
import pandas as pd
from scipy import stats

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()  # WAR-bearing population, 165

pd.set_option("display.width", 130)

# ---- 1. Yearly trend ----
print("=" * 70)
print("1. 연도별 트렌드")
print("=" * 70)
d["gs_share"] = d["kbo_first_year_GS"] / d["kbo_first_year_G"]
yearly = d.groupby("연도").agg(
    n=("kbo_first_year_WAR", "size"),
    war_mean=("kbo_first_year_WAR", "mean"),
    gs_share_mean=("gs_share", "mean"),
    ip_mean=("kbo_first_year_IP", "mean"),
).round(3)
print(yearly.to_string())

pre = d[d["연도"] <= 2019]
post = d[d["연도"] >= 2020]
print("\n--- 2014-2019 vs 2020-2025 ---")
for label, grp in [("2014-2019", pre), ("2020-2025", post)]:
    print(f"{label}: n={len(grp)}, WAR평균={grp['kbo_first_year_WAR'].mean():.3f}, "
          f"GS/G평균={grp['gs_share'].mean():.3f}, IP평균={grp['kbo_first_year_IP'].mean():.1f}")

# ---- 2. throws ----
print("\n" + "=" * 70)
print("2. throws(좌/우완)별 비교")
print("=" * 70)
throws_grp = d.groupby("throws")["kbo_first_year_WAR"].agg(["size", "mean", "median", "std"]).round(3)
print(throws_grp.to_string())
l = d[d["throws"] == "L"]["kbo_first_year_WAR"].dropna()
r = d[d["throws"] == "R"]["kbo_first_year_WAR"].dropna()
if len(l) > 1 and len(r) > 1:
    t, p = stats.ttest_ind(l, r, equal_var=False)
    print(f"Welch t-test (L vs R): t={t:.3f}, p={p:.4f}")

# ---- 3. 대체영입여부 ----
print("\n" + "=" * 70)
print("3. 대체영입여부(N/Y)별 비교")
print("=" * 70)
sub_grp = d.groupby("대체영입여부")["kbo_first_year_WAR"].agg(["size", "mean", "median", "std"]).round(3)
print(sub_grp.to_string())

print("\n--- outcome_category 분포 (대체영입여부별, 열%) ---")
ct = pd.crosstab(d["outcome_category"], d["대체영입여부"], normalize="columns").round(3) * 100
ct_n = pd.crosstab(d["outcome_category"], d["대체영입여부"])
print(ct_n.to_string())
print()
print("(열 기준 비율 %)")
print(ct.to_string())

# ---- 4. asia_league_experience ----
print("\n" + "=" * 70)
print("4. asia_league_experience(Y/N)별 비교")
print("=" * 70)
asia_grp = d.groupby("asia_league_experience")["kbo_first_year_WAR"].agg(["size", "mean", "median", "std"]).round(3)
print(asia_grp.to_string())
y = d[d["asia_league_experience"] == "Y"]["kbo_first_year_WAR"].dropna()
n = d[d["asia_league_experience"] == "N"]["kbo_first_year_WAR"].dropna()
if len(y) > 1 and len(n) > 1:
    t, p = stats.ttest_ind(y, n, equal_var=False)
    print(f"Welch t-test (Y vs N): t={t:.3f}, p={p:.4f}  (주의: Y 표본 {len(y)}명, 작음)")

# ---- 5. mlb_ip_share ----
print("\n" + "=" * 70)
print("5. mlb_ip_share (메이저 의존도) vs WAR")
print("=" * 70)
d["mlb_ip_share"] = d["mlb_career_ip"] / (d["mlb_career_ip"] + d["aaa_career_ip"])
valid = d.dropna(subset=["mlb_ip_share", "kbo_first_year_WAR"])
print(f"mlb_ip_share 결측: {d['mlb_ip_share'].isna().sum()}/{len(d)} (둘 다 없는 선수)")
print(f"유효 표본: {len(valid)}")
r_corr, p_corr = stats.pearsonr(valid["mlb_ip_share"], valid["kbo_first_year_WAR"])
print(f"Pearson r = {r_corr:.3f}, p = {p_corr:.4f}")
print(valid["mlb_ip_share"].describe().round(3).to_string())

# save mlb_ip_share to the main dataset
df["mlb_ip_share"] = df["mlb_career_ip"] / (df["mlb_career_ip"] + df["aaa_career_ip"])
df.to_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv", index=False)
print("\n(mlb_ip_share 컬럼을 analysis_dataset_v1.csv에 저장함)")
