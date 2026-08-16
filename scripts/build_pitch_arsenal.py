#!/usr/bin/env python3
"""
Build per-player pitch-arsenal (Statcast) table for the KBO-entry-immediate-
prior MLB season, for all 167 players.

Data source: Baseball Savant public leaderboard CSVs (cached in
/tmp/savant_cache/ as arsenal_stats_<year>.csv and velo_<year>.csv).

Year-boundary rule (per user decision):
- pitch_usage%, whiff%, run_value_per_100: real values only for seasons
  >= 2017 (Savant leaderboard has zero rows before 2017). NaN otherwise.
- avg_velocity: real values for seasons >= 2015 (Statcast full rollout).
  NaN otherwise.
- statcast_metrics_available: 'full' (season >= 2017), 'velo_only'
  (season in 2015-2016), 'none' (season <= 2014, or no MLB record at all
  before KBO entry).

AAA is not included: Baseball Savant's public leaderboard endpoints return
identical data with or without a minors=true param, and the raw
statcast_search endpoint returns zero pitch-level rows for AAA-only
players -- there is no accessible AAA Statcast source. See
PROJECT_GUIDELINES.md.
"""
import csv

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
CACHE = "/tmp/savant_cache"

PITCH_TYPE_VELO_COL = {
    "FF": "ff_avg_speed", "SI": "si_avg_speed", "FC": "fc_avg_speed", "SL": "sl_avg_speed",
    "CH": "ch_avg_speed", "CU": "cu_avg_speed", "FS": "fs_avg_speed", "KN": "kn_avg_speed",
    "ST": "st_avg_speed", "SV": "sv_avg_speed",
}

_velo_cache = {}
_arsenal_cache = {}


def load_velo(year):
    if year not in _velo_cache:
        d = {}
        with open(f"{CACHE}/velo_{year}.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                d[row["pitcher"]] = row
        _velo_cache[year] = d
    return _velo_cache[year]


def load_arsenal(year):
    if year not in _arsenal_cache:
        rows_by_pid = {}
        with open(f"{CACHE}/arsenal_stats_{year}.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows_by_pid.setdefault(row["player_id"], []).append(row)
        _arsenal_cache[year] = rows_by_pid
    return _arsenal_cache[year]


def tier_for(year):
    if year is None:
        return "none"
    if year >= 2017:
        return "full"
    if year in (2015, 2016):
        return "velo_only"
    return "none"


def main():
    with open(f"{ROOT}/data/rosters/mlb_aaa_career_stats.csv", encoding="utf-8") as f:
        stat_rows = list(csv.DictReader(f))
    with open(f"{ROOT}/data/rosters/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        roster = list(csv.DictReader(f))

    last_mlb_season = {}
    pid_of = {}
    eng_of = {}
    for r in stat_rows:
        if r["level"] == "MLB" and r["is_split_row"] == "0":
            yr = int(float(r["season"]))
            n = r["선수명"]
            if n not in last_mlb_season or yr > last_mlb_season[n]:
                last_mlb_season[n] = yr
                pid_of[n] = r["mlb_person_id"]
                eng_of[n] = r["english_name"]

    coverage_rows = []
    pitch_rows = []

    for r in roster:
        kor = r["선수명"]
        year = last_mlb_season.get(kor)
        tier = tier_for(year)
        pid = pid_of.get(kor, "")
        eng = eng_of.get(kor, r["english_name"])

        n_pitch_types = 0
        if tier in ("full", "velo_only"):
            arsenal = load_arsenal(year) if tier == "full" else {}
            velo = load_velo(year)
            my_arsenal_rows = arsenal.get(pid, []) if tier == "full" else []
            my_velo_row = velo.get(pid)

            if tier == "full" and my_arsenal_rows:
                for row in my_arsenal_rows:
                    pt = row["pitch_type"]
                    velo_col = PITCH_TYPE_VELO_COL.get(pt)
                    avg_v = my_velo_row.get(velo_col) if my_velo_row and velo_col else ""
                    pitch_rows.append({
                        "선수명": kor, "english_name": eng, "mlb_person_id": pid,
                        "savant_season": year, "statcast_metrics_available": tier,
                        "pitch_type": pt, "pitch_name": row["pitch_name"],
                        "pitch_usage_pct": row["pitch_usage"], "avg_velocity": avg_v or "",
                        "whiff_pct": row["whiff_percent"], "run_value_per_100": row["run_value_per_100"],
                        "pitches": row["pitches"],
                    })
                n_pitch_types = len(my_arsenal_rows)
            elif tier == "velo_only" and my_velo_row:
                # no usage/whiff/RV100 breakdown available pre-2017; emit one
                # row per pitch type that has a recorded velocity, metrics blank
                for pt, col in PITCH_TYPE_VELO_COL.items():
                    v = my_velo_row.get(col)
                    if v:
                        pitch_rows.append({
                            "선수명": kor, "english_name": eng, "mlb_person_id": pid,
                            "savant_season": year, "statcast_metrics_available": tier,
                            "pitch_type": pt, "pitch_name": "",
                            "pitch_usage_pct": "", "avg_velocity": v,
                            "whiff_pct": "", "run_value_per_100": "",
                            "pitches": "",
                        })
                n_pitch_types = sum(1 for c in PITCH_TYPE_VELO_COL.values() if my_velo_row.get(c))
            elif tier == "full" and not my_arsenal_rows:
                # season >=2017 but this specific pitcher has no leaderboard row
                # (e.g. traded mid-season with too few pitches to qualify, or
                # min=1 threshold not met in the raw feed for some reason)
                tier = "none"

        coverage_rows.append({
            "선수명": kor, "english_name": eng, "mlb_person_id": pid,
            "last_pre_kbo_mlb_season": year if year is not None else "",
            "statcast_metrics_available": tier, "n_pitch_types_recorded": n_pitch_types,
        })

    pitch_cols = ["선수명", "english_name", "mlb_person_id", "savant_season",
                  "statcast_metrics_available", "pitch_type", "pitch_name",
                  "pitch_usage_pct", "avg_velocity", "whiff_pct", "run_value_per_100", "pitches"]
    with open(f"{ROOT}/data/rosters/pitch_arsenal_full.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pitch_cols)
        w.writeheader()
        w.writerows(pitch_rows)

    cov_cols = ["선수명", "english_name", "mlb_person_id", "last_pre_kbo_mlb_season",
                "statcast_metrics_available", "n_pitch_types_recorded"]
    with open(f"{ROOT}/data/rosters/pitch_arsenal_coverage.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cov_cols)
        w.writeheader()
        w.writerows(coverage_rows)

    from collections import Counter
    tiers = Counter(r["statcast_metrics_available"] for r in coverage_rows)
    print(f"pitch_arsenal_full.csv: {len(pitch_rows)} rows")
    print(f"pitch_arsenal_coverage.csv: {len(coverage_rows)} rows")
    print("tier distribution:", dict(tiers))


if __name__ == "__main__":
    main()
