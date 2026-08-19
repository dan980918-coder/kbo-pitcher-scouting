#!/usr/bin/env python3
"""
GS/G distribution check (2015-2025, all 2,979 pitcher-season) to find a
data-driven starter/reliever/swingman boundary instead of an arbitrary 0.5
cutoff. Uses only already-local STATIZ bulk data.
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

d = pd.read_csv(f"{ROOT}/data/raw/statiz_bulk/statiz_pitching_2015_2025_all.csv")
d = d[d["G"] > 0].copy()
d["gs_ratio"] = d["GS"] / d["G"]

print(f"전체 pitcher-season: {len(d)}")
print(d["gs_ratio"].describe().round(3))

bins = np.arange(0, 1.1, 0.1)
counts, edges = np.histogram(d["gs_ratio"], bins=bins)
print("\n구간별 인원수 (0.1 단위):")
for i in range(len(counts)):
    print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}: {counts[i]:>4}")

print(f"\ngs_ratio==0 비율: {(d['gs_ratio']==0).mean():.3f}")
print(f"gs_ratio>=0.9 비율: {(d['gs_ratio']>=0.9).mean():.3f}")
print(f"중간(0.05~0.95) 비율: {((d['gs_ratio']>0.05)&(d['gs_ratio']<0.95)).mean():.3f}")

kde = gaussian_kde(d["gs_ratio"], bw_method=0.08)
xs = np.linspace(0.01, 0.99, 500)
density = kde(xs)
minima_idx = argrelextrema(density, np.less)[0]
maxima_idx = argrelextrema(density, np.greater)[0]
print("\nKDE 극댓값(모드):", [round(xs[i], 3) for i in maxima_idx])
print("KDE 극솟값(valley):", [round(xs[i], 3) for i in minima_idx])
