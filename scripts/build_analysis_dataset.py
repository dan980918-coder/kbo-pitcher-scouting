#!/usr/bin/env python3
"""
Merge the per-season/per-source collection files into one player-level
analysis table: analysis_dataset_v1.csv (167 rows, one row per player).

Design decisions (documented here since the merge instructions named two
season-level source files but did not pin every column to one of them):

- FanGraphs' fangraphs_career_stats.csv is used as the SOLE source for the
  MLB/AAA last-season, 3yr-weighted, and career-cumulative aggregate
  columns (IP, FIP, K/9, BB/9, HR/9). It already carries every one of these
  fields pre-computed in one consistent row per (player, level, season),
  and FanGraphs was separately confirmed as this project's final FIP
  source (0.0000 mean abs error vs MLB Stats API raw stats at the MLB
  level -- see fip_verification.csv). Mixing IP from one source with rate
  stats from another risks a subtle unit/rounding mismatch within a single
  weighted average, so all five metrics for a given aggregate are always
  pulled from the same row. mlb_aaa_career_stats.csv was checked for
  coverage gaps FanGraphs doesn't have; there are none (both sources hit
  the same structural "no pre-KBO record" cases for the same players), so
  no fallback is needed for these columns.
- Only is_split_row == 0 rows are aggregated (season/level totals), never
  the team-split reference rows, per the project's established rule.
- MLB career WAR is pulled directly from fangraphs_mlb_war_summary.csv
  (already computed there) rather than re-derived.
- Pitch-arsenal detail (per-pitch-type usage%/velocity/whiff%/RV100) is
  deliberately NOT merged in -- only the two coverage summary fields are,
  per the instruction to keep per-pitch-type detail in a separate
  reference file for case-by-case lookup.
"""
import csv

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
RAW = f"{ROOT}/data/raw"
ROSTERS = f"{ROOT}/data/rosters"

METRICS = ["IP", "FIP", "K_9", "BB_9", "HR_9", "FIP_minus"]
OUT_METRIC_NAMES = {"IP": "ip", "FIP": "fip", "K_9": "k9", "BB_9": "bb9", "HR_9": "hr9",
                     "FIP_minus": "fip_minus"}


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ip_to_true_decimal(v):
    # FanGraphs (like all box scores) reports IP in baseball notation: the
    # digit after the decimal point is OUTS, not tenths -- "6.2" = 6 and
    # 2/3 innings = 6.667, not 6.2. Summing or comparing raw notation
    # values is wrong (this bit earlier in verify_fip.py's FIP check too).
    f = to_float(v)
    if f is None:
        return None
    whole = int(f)
    frac_digit = round((f - whole) * 10)
    return whole + frac_digit / 3.0


def load_fg_season_rows():
    with open(f"{RAW}/fangraphs_career_stats.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_player_level = {}
    for r in rows:
        if r["is_split_row"] != "0":
            continue
        level = r["level"]
        if level not in ("MLB", "AAA"):
            continue
        ip = ip_to_true_decimal(r["IP"])
        if ip is None:
            continue
        season = to_float(r["season"])
        by_player_level.setdefault((r["선수명"], level), []).append({
            "season": season, "IP": ip,
            "FIP": to_float(r["FIP"]), "K_9": to_float(r["K_9"]),
            "BB_9": to_float(r["BB_9"]), "HR_9": to_float(r["HR_9"]),
            "FIP_minus": to_float(r.get("FIP_minus")),
        })
    for key in by_player_level:
        by_player_level[key].sort(key=lambda x: x["season"], reverse=True)
    return by_player_level


def last_season_values(seasons):
    if not seasons:
        return {m: "" for m in METRICS}
    s = seasons[0]
    return {m: (round(s[m], 3) if s[m] is not None else "") for m in METRICS}


def weighted_3yr_values(seasons):
    window = seasons[:3]
    if not window:
        return {m: "" for m in METRICS}
    total_ip = sum(s["IP"] for s in window)
    out = {"IP": round(total_ip, 1)}
    if total_ip <= 0:
        for m in METRICS[1:]:
            out[m] = ""
        return out
    for m in METRICS[1:]:
        vals = [(s[m], s["IP"]) for s in window if s[m] is not None]
        if not vals:
            out[m] = ""
            continue
        weighted_ip = sum(ip for _, ip in vals)
        out[m] = round(sum(v * ip for v, ip in vals) / weighted_ip, 3) if weighted_ip > 0 else ""
    return out


def career_ip(seasons):
    if not seasons:
        return ""
    return round(sum(s["IP"] for s in seasons), 1)


def career_weighted_fip_minus(seasons):
    # Same IP-weighted-average logic as the 3yr window, but over every
    # pre-KBO season on record for that level (debut through the season
    # before KBO entry), not just the most recent 3.
    vals = [(s["FIP_minus"], s["IP"]) for s in seasons if s["FIP_minus"] is not None]
    if not vals:
        return ""
    weighted_ip = sum(ip for _, ip in vals)
    if weighted_ip <= 0:
        return ""
    return round(sum(v * ip for v, ip in vals) / weighted_ip, 3)


def main():
    with open(f"{RAW}/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fieldnames = reader.fieldnames
        roster = list(reader)

    fg_seasons = load_fg_season_rows()

    with open(f"{RAW}/fangraphs_mlb_war_summary.csv", encoding="utf-8") as f:
        war_by_player = {r["선수명"]: r for r in csv.DictReader(f)}

    with open(f"{RAW}/pitch_arsenal_coverage.csv", encoding="utf-8") as f:
        pitch_by_player = {r["선수명"]: r for r in csv.DictReader(f)}

    new_cols = []
    for level_prefix, level_key in (("mlb", "MLB"), ("aaa", "AAA")):
        for form in ("last", "3yr"):
            for m in METRICS:
                new_cols.append(f"{level_prefix}_{OUT_METRIC_NAMES[m]}_{form}")
        new_cols.append(f"{level_prefix}_career_ip")
        new_cols.append(f"{level_prefix}_fip_minus_career")
    new_cols.append("mlb_career_war")
    new_cols.append("mlb_n_seasons_pre_kbo")
    new_cols.append("statcast_metrics_available")
    new_cols.append("n_pitch_types_recorded")

    out_fieldnames = base_fieldnames + new_cols

    for r in roster:
        name = r["선수명"]
        for level_prefix, level_key in (("mlb", "MLB"), ("aaa", "AAA")):
            seasons = fg_seasons.get((name, level_key), [])
            last_v = last_season_values(seasons)
            yr3_v = weighted_3yr_values(seasons)
            for m in METRICS:
                r[f"{level_prefix}_{OUT_METRIC_NAMES[m]}_last"] = last_v[m]
                r[f"{level_prefix}_{OUT_METRIC_NAMES[m]}_3yr"] = yr3_v[m]
            r[f"{level_prefix}_career_ip"] = career_ip(seasons)
            r[f"{level_prefix}_fip_minus_career"] = career_weighted_fip_minus(seasons)

        war_row = war_by_player.get(name)
        r["mlb_career_war"] = war_row["mlb_war_career_pre_kbo"] if war_row else ""
        r["mlb_n_seasons_pre_kbo"] = war_row["n_mlb_seasons_pre_kbo"] if war_row else ""

        pitch_row = pitch_by_player.get(name)
        r["statcast_metrics_available"] = pitch_row["statcast_metrics_available"] if pitch_row else ""
        r["n_pitch_types_recorded"] = pitch_row["n_pitch_types_recorded"] if pitch_row else ""

    with open(f"{ROSTERS}/analysis_dataset_v1.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        w.writerows(roster)

    print(f"Done. {len(roster)} rows, {len(out_fieldnames)} columns -> analysis_dataset_v1.csv")


if __name__ == "__main__":
    main()
