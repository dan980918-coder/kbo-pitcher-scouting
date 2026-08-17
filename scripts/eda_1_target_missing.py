#!/usr/bin/env python3
"""EDA-1: target variable distribution + missingness-vs-target check. Text/table output only."""
import pandas as pd
from scipy import stats

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 30)

# ---- 1. KBO first-season WAR distribution (excluding kbo_no_appearance==1) ----
d = df[df["kbo_no_appearance"] != 1].copy()
war = d["kbo_first_year_WAR"]

print("=" * 70)
print(f"1. KBO 첫 시즌 WAR 분포 (n={war.notna().sum()}, kbo_no_appearance=1 제외)")
print("=" * 70)
desc = war.describe(percentiles=[0.25, 0.5, 0.75])
print(desc.to_string())
print(f"skewness (왜도): {war.skew():.3f}")

print("\n상위 5명 (WAR 기준):")
print(d.nlargest(5, "kbo_first_year_WAR")[["선수명", "kbo_first_year_WAR"]].to_string(index=False))

print("\n하위 5명 (WAR 기준):")
print(d.nsmallest(5, "kbo_first_year_WAR")[["선수명", "kbo_first_year_WAR"]].to_string(index=False))

# ---- 2. Missingness vs target ----
print("\n" + "=" * 70)
print("2. 결측 패턴과 타깃(WAR)의 관계")
print("=" * 70)

def compare_groups(flag_col, label):
    has = d[d[flag_col].notna()]
    missing = d[d[flag_col].isna()]
    print(f"\n--- {label} ---")
    print(f"결측 그룹: n={len(missing)}, WAR 평균={missing['kbo_first_year_WAR'].mean():.3f}, "
          f"중앙값={missing['kbo_first_year_WAR'].median():.3f}, std={missing['kbo_first_year_WAR'].std():.3f}")
    print(f"보유 그룹: n={len(has)}, WAR 평균={has['kbo_first_year_WAR'].mean():.3f}, "
          f"중앙값={has['kbo_first_year_WAR'].median():.3f}, std={has['kbo_first_year_WAR'].std():.3f}")
    t, p = stats.ttest_ind(missing["kbo_first_year_WAR"].dropna(), has["kbo_first_year_WAR"].dropna(), equal_var=False)
    print(f"Welch t-test: t={t:.3f}, p={p:.4f}  -> {'유의미(p<0.05)' if p < 0.05 else '유의미하지 않음(p>=0.05)'}")

compare_groups("mlb_ip_last", "mlb_ip_last 결측(AAA 전용 선수) vs 보유")
compare_groups("aaa_ip_last", "aaa_ip_last 결측(MLB 위주 선수) vs 보유")

# ---- 3. Bottom 10 by KBO first-year IP ----
print("\n" + "=" * 70)
print("3. KBO 첫 시즌 IP 하위 10명 (조기방출/실패 사례 재확인)")
print("=" * 70)
bottom_ip = df.nsmallest(10, "kbo_first_year_IP")[
    ["선수명", "kbo_first_year_IP", "kbo_first_year_WAR", "kbo_no_appearance"]
]
print(bottom_ip.to_string(index=False))
print(f"\n(참고: kbo_no_appearance=1인 2명은 IP/WAR 자체가 결측이라 이 목록에는 안 잡힘 -- 별도 확인)")
print(df[df["kbo_no_appearance"] == 1][["선수명", "kbo_first_year_IP", "kbo_first_year_WAR", "kbo_no_appearance"]].to_string(index=False))
