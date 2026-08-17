#!/usr/bin/env python3
"""EDA-3: input-target correlations, multicollinearity, ranking. Text/table only."""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
d = df[df["kbo_no_appearance"] != 1].copy()
TARGET = "kbo_first_year_WAR"

SEASON_START = {
    2014: date(2014, 3, 29), 2015: date(2015, 3, 28), 2016: date(2016, 4, 1),
    2017: date(2017, 3, 31), 2018: date(2018, 3, 24), 2019: date(2019, 3, 23),
    2020: date(2020, 5, 5), 2021: date(2021, 4, 3), 2022: date(2022, 4, 2),
    2023: date(2023, 4, 1), 2024: date(2024, 3, 23), 2025: date(2025, 3, 22),
}
d["birth_date"] = pd.to_datetime(d["birth_date"])
def age_at_entry(row):
    start = SEASON_START.get(int(row["연도"]))
    if start is None or pd.isna(row["birth_date"]):
        return np.nan
    bd = row["birth_date"].date()
    return (start - bd).days / 365.25
d["age_at_kbo_entry"] = d.apply(age_at_entry, axis=1)
df["age_at_kbo_entry"] = np.nan
df.loc[d.index, "age_at_kbo_entry"] = d["age_at_kbo_entry"]

def corr(col):
    sub = d[[col, TARGET]].dropna()
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    r, p = stats.pearsonr(sub[col], sub[TARGET])
    return r, p, len(sub)

# ---- 1. time-window comparison ----
print("=" * 78)
print("1. 시간창(last/3yr/career) 비교 -- MLB/AAA x 지표 x WAR 상관")
print("=" * 78)
metrics = [
    ("fip", "FIP", "낮을수록 좋음(음의 상관=좋은신호)"),
    ("fip_minus", "FIP-", "낮을수록 좋음(음의 상관=좋은신호)"),
    ("k9", "K9", "높을수록 좋음(양의 상관=좋은신호)"),
    ("bb9", "BB9", "낮을수록 좋음(음의 상관=좋은신호)"),
    ("hr9", "HR9", "낮을수록 좋음(음의 상관=좋은신호)"),
]
rows = []
for level in ("mlb", "aaa"):
    for key, label, direction in metrics:
        for window in ("last", "3yr", "career"):
            col = f"{level}_{key}_{window}"
            if col not in df.columns:
                continue
            r, p, n = corr(col)
            rows.append({"level": level.upper(), "metric": label, "window": window,
                         "column": col, "r": r, "p": p, "n": n, "방향해석": direction})
    for window in ("last", "3yr"):
        col = f"{level}_fip_davenport_translated_{window}"
        r, p, n = corr(col)
        rows.append({"level": level.upper(), "metric": "Davenport translated FIP", "window": window,
                     "column": col, "r": r, "p": p, "n": n, "방향해석": "낮을수록 좋음(음의 상관=좋은신호)"})

t1 = pd.DataFrame(rows)
t1["r"] = t1["r"].round(3)
t1["p"] = t1["p"].round(4)
print(t1.to_string(index=False))

# ---- 2. multicollinearity ----
print("\n" + "=" * 78)
print("2. 다중공선성 체크 (FIP 계열, last/3yr/career/davenport)")
print("=" * 78)
for level in ("mlb", "aaa"):
    cols = [f"{level}_fip_last", f"{level}_fip_3yr", f"{level}_fip_minus_career",
            f"{level}_fip_davenport_translated_last"]
    present = [c for c in cols if c in df.columns]
    note = ""
    if f"{level}_fip_career" not in df.columns:
        note = f"  (주: {level}_fip_career 컬럼 없음 -> {level}_fip_minus_career로 대체)"
    print(f"\n--- {level.upper()} ---{note}")
    sub = d[present].dropna()
    cm = sub.corr().round(3)
    print(f"(pairwise-complete n={len(sub)})")
    print(cm.to_string())
    high = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            v = cm.iloc[i, j]
            if abs(v) > 0.9:
                high.append((present[i], present[j], v))
    if high:
        print("0.9 초과 쌍:")
        for a, b, v in high:
            print(f"  {a} <-> {b}: r={v}")
    else:
        print("0.9 초과 쌍: 없음")

# ---- 3. FanGraphs FIP- vs Davenport translated FIP ----
print("\n" + "=" * 78)
print("3. FanGraphs FIP- vs Davenport translated FIP")
print("=" * 78)
for level in ("mlb", "aaa"):
    fgm_col, dav_col = f"{level}_fip_minus_last", f"{level}_fip_davenport_translated_last"
    r_fg, p_fg, n_fg = corr(fgm_col)
    r_dav, p_dav, n_dav = corr(dav_col)
    sub = d[[fgm_col, dav_col]].dropna()
    r_between, p_between = stats.pearsonr(sub[fgm_col], sub[dav_col]) if len(sub) > 2 else (np.nan, np.nan)
    print(f"\n--- {level.upper()} ---")
    print(f"{fgm_col} vs WAR: r={r_fg:.3f}, p={p_fg:.4f}, n={n_fg}")
    print(f"{dav_col} vs WAR: r={r_dav:.3f}, p={p_dav:.4f}, n={n_dav}")
    print(f"둘 사이 상관 ({fgm_col} vs {dav_col}): r={r_between:.3f}, n={len(sub)}")

# ---- 4. other variables ----
print("\n" + "=" * 78)
print("4. 기타 변수 (mlb_n_seasons_pre_kbo, n_pitch_types_recorded, age_at_kbo_entry)")
print("=" * 78)
other_cols = ["mlb_n_seasons_pre_kbo", "n_pitch_types_recorded", "age_at_kbo_entry", "mlb_ip_share"]
other_rows = []
for col in other_cols:
    r, p, n = corr(col)
    other_rows.append({"column": col, "r": round(r, 3), "p": round(p, 4), "n": n})
t4 = pd.DataFrame(other_rows)
print(t4.to_string(index=False))

# ---- 5. overall ranking ----
print("\n" + "=" * 78)
print("5. 종합 랭킹 (|r| 기준 정렬)")
print("=" * 78)
all_rows = t1[["column", "r", "p", "n"]].to_dict("records") + other_rows
t5 = pd.DataFrame(all_rows).dropna(subset=["r"])
t5["abs_r"] = t5["r"].abs()
t5 = t5.sort_values("abs_r", ascending=False).reset_index(drop=True)
t5.index += 1
print(t5[["column", "r", "p", "n"]].to_string())

df.to_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv", index=False)
print("\n(age_at_kbo_entry 컬럼을 analysis_dataset_v1.csv에 저장함)")
