#!/usr/bin/env python3
"""EDA-1 visualizations: WAR distribution, missingness vs WAR, IP vs WAR."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
OUT = f"{ROOT}/reports/eda"
import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
war = d["kbo_first_year_WAR"]

# ---- 1. WAR histogram ----
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(war, bins=20, color="#4C72B0", edgecolor="white", alpha=0.9)
mean_v, med_v = war.mean(), war.median()
ax.axvline(mean_v, color="#C44E52", linestyle="--", linewidth=2, label=f"평균 = {mean_v:.2f}")
ax.axvline(med_v, color="#55A868", linestyle="--", linewidth=2, label=f"중앙값 = {med_v:.2f}")
ax.set_title(f"KBO 첫 시즌 WAR 분포 (n=165, 왜도={war.skew():.3f})", fontsize=14)
ax.set_xlabel("KBO 첫 시즌 WAR")
ax.set_ylabel("선수 수")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eda1_1_war_histogram.png", dpi=150)
plt.close(fig)
print("saved eda1_1_war_histogram.png")

# ---- 2 & 3. missingness boxplots ----
def missing_boxplot(flag_col, label, fname):
    grp_missing = d.loc[d[flag_col].isna(), "kbo_first_year_WAR"].dropna()
    grp_has = d.loc[d[flag_col].notna(), "kbo_first_year_WAR"].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([grp_missing, grp_has], tick_labels=[
        f"결측 (n={len(grp_missing)})", f"보유 (n={len(grp_has)})"
    ], patch_artist=True, widths=0.5)
    colors = ["#C44E52", "#4C72B0"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(f"{label}\nKBO 첫 시즌 WAR 비교", fontsize=13)
    ax.set_ylabel("KBO 첫 시즌 WAR")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=150)
    plt.close(fig)
    print(f"saved {fname}")

missing_boxplot("mlb_ip_last", "mlb_ip_last 결측 여부 (AAA 전용 vs MLB 경력 보유)", "eda1_2_mlb_missing_boxplot.png")
missing_boxplot("aaa_ip_last", "aaa_ip_last 결측 여부 (MLB 위주 vs AAA 경력 보유)", "eda1_3_aaa_missing_boxplot.png")

# ---- 4. IP vs WAR scatter, bottom 10 highlighted ----
bottom10_names = set(df.nsmallest(10, "kbo_first_year_IP")["선수명"])
df_plot = df.dropna(subset=["kbo_first_year_IP", "kbo_first_year_WAR"]).copy()
df_plot["is_bottom10"] = df_plot["선수명"].isin(bottom10_names)

fig, ax = plt.subplots(figsize=(10, 7))
normal = df_plot[~df_plot["is_bottom10"]]
bottom = df_plot[df_plot["is_bottom10"]]
ax.scatter(normal["kbo_first_year_IP"], normal["kbo_first_year_WAR"],
           color="#4C72B0", alpha=0.6, s=40, label=f"나머지 (n={len(normal)})")
ax.scatter(bottom["kbo_first_year_IP"], bottom["kbo_first_year_WAR"],
           color="#C44E52", alpha=0.9, s=70, edgecolor="black", linewidth=0.5,
           label=f"IP 하위 10명")
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.set_title(f"KBO 첫 시즌 IP vs WAR (n={len(df_plot)})", fontsize=14)
ax.set_xlabel("KBO 첫 시즌 IP")
ax.set_ylabel("KBO 첫 시즌 WAR")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/eda1_4_ip_vs_war_scatter.png", dpi=150)
plt.close(fig)
print("saved eda1_4_ip_vs_war_scatter.png")
