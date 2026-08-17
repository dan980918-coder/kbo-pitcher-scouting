#!/usr/bin/env python3
"""
Classify all 167 players into 5 outcome categories:
  조기방출, 시즌후_자연종료, 완주-재계약안됨,
  완주-재계약됨(일반슬롯), 완주-재계약됨(아시아쿼터)

Sources (no new web scraping needed beyond the namu.wiki waiver list
already fetched to /tmp/namu_waiver_section.txt this session):
- Release status + date: parsed from the namu.wiki "KBO 리그 웨이버
  공시/선수 목록" document (validated 15/15 against manually-verified
  cases before running on all 167).
- resigned_next_year: derived from data/rosters/kbo_yearly_stats_all.csv
  by checking whether a player has a row for kbo_year+1.
- resigned_as_asia_quota: the KBO Asia-quota slot did NOT exist before
  2026 (approved by the KBO board January 2025, first applied in the
  2026 season) -- confirmed after an earlier pass in this project
  wrongly assumed 2022. Every resigned_next_year candidate whose resign
  year is < 2026 is therefore automatically N (the rule couldn't have
  applied). The 9 candidates whose resign year is exactly 2026 (all
  players whose kbo_year == 2025) were each individually verified via
  news search rather than inferred from nationality -- nationality alone
  isn't a reliable proxy for this (Lachlan Wells, Australian, IS an
  Asia-quota signee, but so is e.g. Miyamori Satoshi, Japanese, while
  most regular-slot re-signings that year are American). Confirmed: only
  Lachlan Wells (LG, 2026) is Asia-quota; the other 8 (Zach Logue/두산,
  Yonny Chirinos/LG, Anders Tolhurst/LG, Adam Oller/KIA, Logan Allen/KT,
  Riley Thompson/NC, Mitch White/SSG, Kenny Rosenberg/키움) were each
  explicitly confirmed in news coverage as regular foreign-slot
  signings, distinct from that team's separate Asia-quota player.
"""
import csv
import json
import re
from datetime import date, timedelta

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

SEASON_END = {
    2014: date(2014, 10, 17), 2015: date(2015, 10, 6), 2016: date(2016, 9, 19),
    2017: date(2017, 10, 3), 2018: date(2018, 10, 14), 2019: date(2019, 10, 1),
    2020: date(2020, 10, 31), 2021: date(2021, 10, 30), 2022: date(2022, 10, 11),
    2023: date(2023, 10, 17), 2024: date(2024, 10, 1), 2025: date(2025, 10, 4),
}
NATURAL_END_WINDOW_DAYS = 30

TEAM_ALIAS = {
    "넥센": {"넥센", "키움"}, "키움": {"넥센", "키움"},
    "SK": {"SK", "SSG"}, "SSG": {"SK", "SSG"},
}
TEAMS = {"넥센", "한화", "SK", "LG", "두산", "KIA", "NC", "kt", "롯데", "삼성", "SSG", "키움"}
DATE_RE = re.compile(r"^\d{4}\.\d{2}\.(\d{2}\.)?$")


def team_match(roster_team, found_team):
    if found_team is None:
        return False
    if found_team == roster_team:
        return True
    return found_team in TEAM_ALIAS.get(roster_team, {roster_team})


def find_release(lines, kor_name, kbo_year, roster_team):
    parts = kor_name.replace(".", "").split()
    candidates = [parts[-1]] if len(parts) < 2 else [parts[-1], parts[0]]
    matches = []
    for cand in candidates:
        if len(cand) < 2:
            continue
        for idx, line in enumerate(lines):
            if line != cand:
                continue
            date_str = None
            for j in range(idx, -1, -1):
                if DATE_RE.match(lines[j]):
                    date_str = lines[j]
                    break
            if not date_str:
                continue
            team = None
            for j in range(idx + 1, min(idx + 6, len(lines))):
                if DATE_RE.match(lines[j]):
                    break
                if lines[j] in TEAMS:
                    team = lines[j]
                    break
            if team is None:
                for j in range(idx - 1, max(idx - 25, -1), -1):
                    if lines[j] in TEAMS:
                        team = lines[j]
                        break
            yr = int(date_str[:4])
            if yr == kbo_year and team_match(roster_team, team):
                mm, dd = int(date_str[5:7]), int(date_str[8:10])
                matches.append(date(yr, mm, dd))
    return min(matches) if matches else None


def main():
    with open(f"{ROOT}/data/raw/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        roster = list(csv.DictReader(f))
    with open("/tmp/namu_waiver_section.txt", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    with open(f"{ROOT}/data/rosters/kbo_yearly_stats_all.csv", encoding="utf-8") as f:
        yearly = list(csv.DictReader(f))
    years_by_player = {}
    for r in yearly:
        years_by_player.setdefault(r["선수명"], set()).add(int(r["Year"]))

    # (nationality is intentionally not loaded here as an asia-quota proxy --
    # see rationale below)
    # Asia-quota didn't exist before 2026 (KBO board approved it Jan 2025,
    # first applied 2026 season). Every resigned_next_year candidate with
    # resign year < 2026 is auto-N. The 9 candidates with resign year
    # exactly 2026 were each individually verified via news search (see
    # module docstring) -- nationality is not a reliable proxy for this.
    ASIA_QUOTA_2026_VERIFIED = {
        "라클란 웰스": "Y",       # LG, confirmed Asia-quota signee
        "잭 로그": "N",           # 두산, explicitly confirmed regular slot
        "요니 치리노스": "N",      # LG, regular slot (Wells is LG's separate AQ signee)
        "앤더스 톨허스트": "N",    # LG, explicitly confirmed regular slot
        "애덤 올러": "N",         # KIA, explicitly confirmed regular slot (KIA's AQ is Jared Dale)
        "로건 앨런": "N",         # KT, regular foreign contract
        "라일리 톰프슨": "N",     # NC, regular slot (NC's AQ is Toda Natsuki)
        "미치 화이트": "N",       # SSG, regular slot (SSG's AQ is Takeda)
        "케니 로젠버그": "N",     # 키움, regular injury-replacement foreign slot
    }

    out_rows = []
    for r in roster:
        kor = r["선수명"]
        if r["kbo_no_appearance"] == "1":
            out_rows.append({"선수명": kor, "연도": r["연도"], "팀": r["팀"],
                              "release_date": "", "outcome_category": "분류제외(kbo_no_appearance)",
                              "resigned_next_year": "", "resigned_as_asia_quota": ""})
            continue

        kbo_year = int(r["연도"])
        team = r["팀"]
        release_dt = find_release(lines, kor, kbo_year, team)

        next_year_present = (kbo_year + 1) in years_by_player.get(kor, set())
        resigned = "Y" if next_year_present else "N"
        asia_quota = ""
        if resigned == "Y" and (kbo_year + 1) >= 2026:
            asia_quota = ASIA_QUOTA_2026_VERIFIED.get(kor, "N")
        elif resigned == "Y":
            asia_quota = "N"  # program didn't exist yet (pre-2026)

        if release_dt:
            season_end = SEASON_END[kbo_year]
            days_before_end = (season_end - release_dt).days
            if 0 <= days_before_end <= NATURAL_END_WINDOW_DAYS:
                category = "시즌후_자연종료"
            else:
                category = "조기방출"
        else:
            if resigned == "N":
                category = "완주-재계약안됨"
            elif asia_quota == "Y":
                category = "완주-재계약됨(아시아쿼터)"
            else:
                category = "완주-재계약됨(일반슬롯)"

        out_rows.append({
            "선수명": kor, "연도": r["연도"], "팀": team,
            "release_date": release_dt.isoformat() if release_dt else "",
            "outcome_category": category,
            "resigned_next_year": resigned, "resigned_as_asia_quota": asia_quota,
        })

    cols = ["선수명", "연도", "팀", "release_date", "outcome_category", "resigned_next_year", "resigned_as_asia_quota"]
    with open(f"{ROOT}/data/rosters/outcome_category.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    from collections import Counter
    print(Counter(r["outcome_category"] for r in out_rows))
    print(f"\n{len(out_rows)} rows -> data/rosters/outcome_category.csv")


if __name__ == "__main__":
    main()
