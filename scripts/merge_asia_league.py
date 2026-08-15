#!/usr/bin/env python3
"""
Merge asia_league_check.csv into new_import_pitchers_2014_2025_draft_v2.csv.

Adds two columns:
- asia_league_experience (Y/N): NPB/CPBL (or other Asian pro league)
  experience *before* the player's first KBO season. Blank (missing) if
  we couldn't confirm identity (Wikipedia page missing, or birth date
  conflicts with STATIZ and wasn't independently resolved).
- asia_league_detail: league/team/year text, blank if none or unconfirmed.

Rows flagged needs_review=Y in asia_league_check.csv are deliberately left
blank in the main CSV rather than auto-decided, per the review policy.
"""
import csv

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting/data/rosters"

with open(f"{ROOT}/asia_league_check.csv", encoding="utf-8") as f:
    checks = {row["선수명"]: row for row in csv.DictReader(f)}

with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    roster = list(reader)

new_cols = ["asia_league_experience", "asia_league_detail"]
new_fieldnames = fieldnames + new_cols

for r in roster:
    c = checks.get(r["선수명"])
    if not c or c["needs_review"] == "Y":
        r["asia_league_experience"] = ""
        r["asia_league_detail"] = ""
    else:
        r["asia_league_experience"] = c["asia_league_experience"]
        r["asia_league_detail"] = c["asia_league_detail"]

with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(roster)

y = sum(1 for r in roster if r["asia_league_experience"] == "Y")
n = sum(1 for r in roster if r["asia_league_experience"] == "N")
blank = sum(1 for r in roster if r["asia_league_experience"] == "")
print(f"Merged. asia_league_experience: Y={y}, N={n}, blank(review-pending)={blank}")
