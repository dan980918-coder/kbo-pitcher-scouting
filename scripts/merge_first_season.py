#!/usr/bin/env python3
"""
Merge profile_data.csv + kbo_yearly_stats_all.csv into
new_import_pitchers_2014_2025_draft_v2.csv.

- Adds profile columns: english_name, birth_date, throws, nationality
- Adds KBO first-season performance columns computed from the yearly table:
  kbo_first_year_WAR, kbo_first_year_IP, kbo_first_year_GS,
  kbo_first_year_ERA, kbo_first_year_FIP, kbo_first_year_WHIP,
  kbo_first_year_K9 (=SO/IP*9), kbo_first_year_BB9 (=BB/IP*9)

"KBO 첫 시즌" = the earliest Year present for that player in
kbo_yearly_stats_all.csv (that file only contains seasons with real
game appearances, consistent with the project's "실전 출전 연도" rule).

If a player's first season is split across two teams (mid-season trade),
the counting stats (G, GS, IP, ER, H, BB, SO, WAR) are summed and ERA/WHIP/
K9/BB9 are recomputed from the summed components. FIP has no simple additive
formula (depends on a league constant), so it is IP-weighted-averaged across
the split rows as a reasonable approximation.
"""
import csv
from collections import defaultdict

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting/data/rosters"


def ip_to_outs(ip_str):
    """Convert baseball-notation IP (e.g. '65.2') to total outs (int)."""
    ip_str = ip_str.strip()
    if not ip_str:
        return 0
    whole, _, frac = ip_str.partition(".")
    whole = int(whole)
    frac = int(frac) if frac else 0
    return whole * 3 + frac


def outs_to_ip(outs):
    """Convert total outs back to baseball-notation IP string."""
    whole = outs // 3
    frac = outs % 3
    return f"{whole}.{frac}"


def outs_to_float(outs):
    return outs / 3.0


# 1. Load profile data
profiles = {}
with open(f"{ROOT}/profile_data.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        profiles[row["선수명"]] = row

# 2. Load yearly stats, grouped by player
yearly_by_player = defaultdict(list)
with open(f"{ROOT}/kbo_yearly_stats_all.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        yearly_by_player[row["선수명"]].append(row)

# 3. Load main roster
with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    roster_fieldnames = reader.fieldnames
    roster = list(reader)

profile_cols = ["english_name", "birth_date", "throws", "nationality"]
perf_cols = [
    "kbo_first_year_WAR", "kbo_first_year_IP", "kbo_first_year_GS",
    "kbo_first_year_ERA", "kbo_first_year_FIP", "kbo_first_year_WHIP",
    "kbo_first_year_K9", "kbo_first_year_BB9",
]
new_fieldnames = roster_fieldnames + profile_cols + perf_cols

missing_profile = []
missing_perf = []

for r in roster:
    name = r["선수명"]

    # profile
    prof = profiles.get(name)
    if prof:
        for c in profile_cols:
            r[c] = prof.get(c, "")
    else:
        for c in profile_cols:
            r[c] = ""
        missing_profile.append(name)

    # first-season performance
    rows = yearly_by_player.get(name)
    if not rows:
        for c in perf_cols:
            r[c] = ""
        missing_perf.append(name)
        continue

    first_year = min(row["Year"] for row in rows)
    fy_rows = [row for row in rows if row["Year"] == first_year]

    total_outs = sum(ip_to_outs(row["IP"]) for row in fy_rows)
    total_gs = sum(int(row["GS"]) for row in fy_rows)
    total_er = sum(int(row["ER"]) for row in fy_rows)
    total_h = sum(int(row["H"]) for row in fy_rows)
    total_bb = sum(int(row["BB"]) for row in fy_rows)
    total_hp = sum(int(row["HP"]) for row in fy_rows)
    total_hr = sum(int(row["HR"]) for row in fy_rows)
    total_so = sum(int(row["SO"]) for row in fy_rows)
    total_war = round(sum(float(row["WAR"]) for row in fy_rows), 2)

    ip_float = outs_to_float(total_outs)
    era = round(total_er * 9 / ip_float, 2) if ip_float else ""
    whip = round((total_bb + total_h) / ip_float, 2) if ip_float else ""
    k9 = round(total_so * 9 / ip_float, 2) if ip_float else ""
    bb9 = round(total_bb * 9 / ip_float, 2) if ip_float else ""

    if len(fy_rows) == 1:
        # single team all season: use STATIZ's own FIP as-is, no recompute
        fip = fy_rows[0]["FIP"]
    else:
        # mid-season trade split across teams: back out each row's implied
        # FIP constant (FIP = (13*HR+3*(BB+HBP)-2*K)/IP + constant), take the
        # IP-weighted average of those constants (same year/league, so they
        # should agree up to STATIZ's own rounding), then apply it to the
        # combined HR/BB/HBP/K/IP to compute FIP directly from the formula.
        def implied_constant(row):
            outs = ip_to_outs(row["IP"])
            ip = outs_to_float(outs)
            component = (13 * int(row["HR"]) + 3 * (int(row["BB"]) + int(row["HP"])) - 2 * int(row["SO"])) / ip
            return float(row["FIP"]) - component, outs

        weighted_const_sum = 0.0
        for row in fy_rows:
            c, outs = implied_constant(row)
            weighted_const_sum += c * outs
        avg_constant = weighted_const_sum / total_outs

        combined_component = (13 * total_hr + 3 * (total_bb + total_hp) - 2 * total_so) / ip_float
        fip = round(combined_component + avg_constant, 2)

    r["kbo_first_year_WAR"] = total_war
    r["kbo_first_year_IP"] = outs_to_ip(total_outs)
    r["kbo_first_year_GS"] = total_gs
    r["kbo_first_year_ERA"] = era
    r["kbo_first_year_FIP"] = fip
    r["kbo_first_year_WHIP"] = whip
    r["kbo_first_year_K9"] = k9
    r["kbo_first_year_BB9"] = bb9

with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(roster)

print(f"Merged {len(roster)} players.")
print(f"Missing profile ({len(missing_profile)}): {missing_profile}")
print(f"Missing first-season perf ({len(missing_perf)}): {missing_perf}")
