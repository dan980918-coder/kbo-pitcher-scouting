#!/usr/bin/env python3
"""EDA-2 visualizations: yearly GS/G trend, outcome_category by 대체영입여부."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
OUT = f"{ROOT}/reports/eda"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
d["gs_share"] = d["kbo_first_year_GS"] / d["kbo_first_year_G"]

# ---- 1. yearly GS/G trend ----
yearly = d.groupby("연도")["gs_share"].mean()
pre_mean = d[d["연도"] <= 2019]["gs_share"].mean()
post_mean = d[d["연도"] >= 2020]["gs_share"].mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly.index, yearly.values, marker="o", color="#4C72B0", linewidth=2, label="연도별 GS/G 평균")
ax.axvline(2019.5, color="gray", linestyle=":", linewidth=1.5, label="2020년 (3인 동시출전 도입)")
ax.hlines(pre_mean, 2014, 2019.5, color="#C44E52", linestyle="--", linewidth=2,
          label=f"2014-2019 평균 = {pre_mean:.3f}")
ax.hlines(post_mean, 2019.5, 2025, color="#55A868", linestyle="--", linewidth=2,
          label=f"2020-2025 평균 = {post_mean:.3f}")
ax.set_title("연도별 GS/G(선발 비중) 추이", fontsize=14)
ax.set_xlabel("연도")
ax.set_ylabel("GS/G")
ax.set_xticks(range(2014, 2026))
ax.set_ylim(0.75, 1.05)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/eda2_1_gs_share_trend.png", dpi=150)
plt.close(fig)
print("saved eda2_1_gs_share_trend.png")

# ---- 2. outcome_category by 대체영입여부, 100% stacked bar ----
CAT_ORDER = ["조기방출", "시즌후_자연종료", "완주-재계약안됨",
             "완주-재계약됨(일반슬롯)", "완주-재계약됨(아시아쿼터)"]
COLORS = ["#C44E52", "#DD8452", "#8C8C8C", "#4C72B0", "#55A868"]

ct = pd.crosstab(d["대체영입여부"], d["outcome_category"], normalize="index")
ct = ct.reindex(columns=CAT_ORDER, fill_value=0)
ct = ct.reindex(["N", "Y"])
counts = pd.crosstab(d["대체영입여부"], d["outcome_category"]).reindex(["N", "Y"])
n_by_group = counts.sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
bottom = pd.Series([0.0, 0.0], index=["N", "Y"])
bar_positions = [0, 1]
for cat, color in zip(CAT_ORDER, COLORS):
    vals = ct[cat].values * 100
    bars = ax.bar(bar_positions, vals, bottom=bottom.values, color=color, width=0.55, label=cat)
    for i, (v, b) in enumerate(zip(vals, bottom.values)):
        if v >= 3:
            ax.text(bar_positions[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
    bottom += ct[cat].values * 100

ax.set_xticks(bar_positions)
ax.set_xticklabels([f"정식영입 N\n(n={n_by_group['N']})", f"대체영입 Y\n(n={n_by_group['Y']})"], fontsize=11)
ax.set_ylabel("비율 (%)")
ax.set_title("대체영입여부별 outcome_category 분포", fontsize=14)
ax.set_ylim(0, 100)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/eda2_2_outcome_by_substitution.png", dpi=150)
plt.close(fig)
print("saved eda2_2_outcome_by_substitution.png")
