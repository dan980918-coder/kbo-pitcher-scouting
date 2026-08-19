#!/usr/bin/env python3
"""
Reverse-engineer whether Davenport Translation pulls extreme performances
toward league average, using ONLY already-local data (no new scraping):
match each player's raw (untranslated) FanGraphs AAA-season FIP against
Davenport's translated FIP for the SAME player-season.

distance = raw_FIP - implied_league_avg_FIP (positive = worse than average)
  implied_league_avg_FIP backed out from FanGraphs' own FIP_minus:
  league_avg = raw_FIP * 100 / FIP_minus
delta = FIP_DAV(translated) - raw_FIP (positive = translation made it worse)

Regression-to-mean hypothesis: delta should be negatively correlated with
signed distance (bad players pulled down/better, good players pulled
up/worse), and |distance| positively correlated with |delta| (farther from
average -> pulled harder).
"""
import csv
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"


def load_fip_constants():
    with open(f"{ROOT}/data/rosters/fangraphs_fip_constants.csv", encoding="utf-8") as f:
        return {int(r["season"]): float(r["cFIP"]) for r in csv.DictReader(f)}


def load_davenport_aaa_by_season():
    constants = load_fip_constants()
    with open(f"{ROOT}/data/raw/davenport_career_stats.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    combined = {}
    for r in rows:
        if r["level"] != "AAA":
            continue
        try:
            ip = float(r["IP"])
            hr, bb, k = float(r["HR"]), float(r["BB"]), float(r["K"])
            season = int(r["season"])
        except (ValueError, KeyError):
            continue
        if ip <= 0:
            continue
        key = (r["선수명"], season)
        c = combined.setdefault(key, {"IP": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0})
        c["IP"] += ip
        c["HR"] += hr
        c["BB"] += bb
        c["K"] += k
    out = {}
    for (name, season), c in combined.items():
        const = constants.get(season)
        if const is None or c["IP"] <= 0:
            continue
        out[(name, season)] = (13 * c["HR"] + 3 * c["BB"] - 2 * c["K"]) / c["IP"] + const
    return out


dav_aaa = load_davenport_aaa_by_season()

fg = pd.read_csv(f"{ROOT}/data/raw/fangraphs_career_stats.csv")
aaa = fg[(fg["level"] == "AAA") & (fg["is_split_row"] == 0)].copy()
aaa = aaa.dropna(subset=["FIP", "FIP_minus", "IP"])
aaa = aaa[aaa["IP"] > 0]

aaa["implied_league_avg_fip"] = aaa["FIP"] * 100 / aaa["FIP_minus"]
aaa["distance"] = aaa["FIP"] - aaa["implied_league_avg_fip"]  # positive = worse than average

aaa["fip_dav"] = aaa.apply(lambda r: dav_aaa.get((r["선수명"], int(r["season"])), np.nan), axis=1)
matched = aaa.dropna(subset=["fip_dav"]).copy()
matched["delta"] = matched["fip_dav"] - matched["FIP"]

print("=" * 90)
print(f"매칭된 선수-시즌: {len(matched)} (원본 FanGraphs AAA {len(aaa)}개 중 Davenport translated 매칭)")
print("=" * 90)
print(matched[["선수명", "season", "IP", "FIP", "implied_league_avg_fip", "distance", "fip_dav", "delta"]].head(10).round(3).to_string(index=False))

print("\n" + "=" * 90)
print("1. 리그평균 거리 분포")
print("=" * 90)
print(matched["distance"].describe().round(3))

print("\n" + "=" * 90)
print("2. 거리 vs 변화폭 상관관계")
print("=" * 90)
r_signed, p_signed = stats.pearsonr(matched["distance"], matched["delta"])
print(f"부호 있는 distance vs delta: r={r_signed:.3f}, p={p_signed:.4f}, n={len(matched)}")
print("  (평균으로의 정규화 가설이 맞다면 음의 상관 -- 평균보다 나쁜 선수(distance>0)일수록 delta<0(번역후 더 좋아짐) 방향)")

r_abs, p_abs = stats.pearsonr(matched["distance"].abs(), matched["delta"].abs())
print(f"\n|distance| vs |delta|: r={r_abs:.3f}, p={p_abs:.4f}")
print("  (평균에서 멀수록 더 세게 당겨진다는 가설이 맞다면 양의 상관)")

# quick visual bucket check
matched["distance_bucket"] = pd.cut(matched["distance"], bins=[-10, -1.5, -0.5, 0.5, 1.5, 10],
                                     labels=["<-1.5(매우좋음)", "-1.5~-0.5(좋음)", "-0.5~0.5(평균)", "0.5~1.5(나쁨)", ">1.5(매우나쁨)"])
print("\n구간별 평균 delta (평균으로의 정규화라면 매우좋음=+delta 커야, 매우나쁨=-delta 커야):")
print(matched.groupby("distance_bucket", observed=True)["delta"].agg(["size", "mean", "std"]).round(3))
