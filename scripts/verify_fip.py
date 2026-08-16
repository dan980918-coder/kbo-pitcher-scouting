#!/usr/bin/env python3
"""
Verification (검산): compare FIP computed from MLB Stats API raw counting
stats (mlb_aaa_career_stats.csv) against FanGraphs' own official FIP
(fangraphs_career_stats.csv) for the same player-season-level, restricted
to rows both sources agree are season totals (is_split_row == 0).

FIP formula: ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + cFIP
using FanGraphs' own published per-season constant (fangraphs_fip_constants.csv)
so the comparison isolates real data/methodology differences rather than an
unrelated constant choice.

Matching key: (선수명, level, season). AAA rows are matched too (FanGraphs'
per-season cFIP is an MLB-average constant; applying it to AAA innings is a
known simplification -- flagged in the output, not hidden).
"""
import csv
from collections import defaultdict

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting/data/rosters"


def load_constants():
    with open(f"{ROOT}/fangraphs_fip_constants.csv", encoding="utf-8") as f:
        return {int(r["season"]): float(r["cFIP"]) for r in csv.DictReader(f)}


def main():
    constants = load_constants()

    with open(f"{ROOT}/mlb_aaa_career_stats.csv", encoding="utf-8") as f:
        mlb_rows = list(csv.DictReader(f))
    with open(f"{ROOT}/fangraphs_career_stats.csv", encoding="utf-8") as f:
        fg_rows = list(csv.DictReader(f))

    fg_by_key = {}
    for r in fg_rows:
        if r["is_split_row"] != "0":
            continue
        key = (r["선수명"], r["level"], r["season"])
        fg_by_key[key] = r

    out_rows = []
    for r in mlb_rows:
        if r["is_split_row"] != "0":
            continue
        key = (r["선수명"], r["level"], r["season"])
        fg = fg_by_key.get(key)
        if not fg or not fg.get("FIP"):
            continue

        try:
            ip_raw = r["IP"]
            ip_whole = int(float(ip_raw))
            ip_frac_digit = round((float(ip_raw) - ip_whole) * 10)
            # MLB Stats API (like FanGraphs and all official box scores) reports
            # IP in baseball notation: the digit after the decimal point is
            # OUTS, not tenths -- ".1" = 1 out = 1/3 inning, ".2" = 2 outs =
            # 2/3 inning. Treating it as a true decimal (e.g. 0.1 as 0.1
            # innings instead of 0.333) explodes FIP for short appearances.
            ip = ip_whole + ip_frac_digit / 3.0
            hr = float(r["HR"])
            bb = float(r["BB"])
            hbp = float(r["HBP"]) if r["HBP"] else 0.0
            k = float(r["SO"])
        except (ValueError, KeyError):
            continue
        if ip <= 0:
            continue

        season = int(float(r["season"]))
        const = constants.get(season)
        if const is None:
            continue

        my_fip = ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + const
        fg_fip = float(fg["FIP"])
        diff = my_fip - fg_fip

        fg_ip_raw = fg.get("IP", "")
        if fg_ip_raw:
            fg_ip_whole = int(float(fg_ip_raw))
            fg_ip_frac_digit = round((float(fg_ip_raw) - fg_ip_whole) * 10)
            fg_ip_true = round(fg_ip_whole + fg_ip_frac_digit / 3.0, 3)
        else:
            fg_ip_true = ""

        out_rows.append({
            "선수명": r["선수명"], "level": r["level"], "season": season,
            "IP_mlbstatsapi": round(ip, 3), "IP_fangraphs": fg_ip_true,
            "my_fip": round(my_fip, 3), "fangraphs_fip": round(fg_fip, 3),
            "diff": round(diff, 3), "cFIP_used": const,
        })

    out_cols = ["선수명", "level", "season", "IP_mlbstatsapi", "IP_fangraphs",
                "my_fip", "fangraphs_fip", "diff", "cFIP_used"]
    with open(f"{ROOT}/fip_verification.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)

    diffs = [r["diff"] for r in out_rows]
    if diffs:
        import statistics
        print(f"n compared rows: {len(diffs)}")
        print(f"mean diff: {statistics.mean(diffs):.4f}")
        print(f"median diff: {statistics.median(diffs):.4f}")
        print(f"mean abs diff: {statistics.mean(abs(d) for d in diffs):.4f}")
        print(f"stdev: {statistics.pstdev(diffs):.4f}")
        print(f"max abs diff: {max(abs(d) for d in diffs):.4f}")
        by_level = defaultdict(list)
        for r in out_rows:
            by_level[r["level"]].append(r["diff"])
        for lvl, ds in by_level.items():
            print(f"  {lvl}: n={len(ds)}, mean_abs_diff={statistics.mean(abs(d) for d in ds):.4f}")
        # flag IP mismatches (would indicate a season/level matching problem)
        ip_mismatches = [r for r in out_rows if r["IP_fangraphs"] and abs(r["IP_mlbstatsapi"] - float(r["IP_fangraphs"])) > 0.2]
        print(f"rows where IP itself disagrees between sources by >0.2: {len(ip_mismatches)}")
    else:
        print("No overlapping rows found.")


if __name__ == "__main__":
    main()
