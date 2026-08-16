#!/usr/bin/env python3
"""
Collect MLB + AAA career (debut through the season before KBO entry)
pitching stats for all 167 players via the MLB Stats API.

Matching: name search against /api/v1/people/search, disambiguated by
exact birth_date match against STATIZ's birth_date for that player.
Multiple name variants are tried until one yields a birth_date match.

Parsing rule (per user spec):
- Every row the API returns with a "team" key is kept as-is with
  is_split_row=1 (team-specific stint).
- Every row without a "team" key (the API's own multi-team aggregate)
  is kept with is_split_row=0.
- For seasons where the player was with a single team all year, the API
  returns only one (team) row and no aggregate row. To keep "always
  aggregate from is_split_row=0" a uniform rule for downstream use, we
  synthesize a duplicate is_split_row=0 row for those seasons (identical
  stats, team field replaced with the joined team name(s) for that
  season-level).
"""
import csv
import json
import time
import urllib.request
import urllib.parse

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting/data/rosters"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def api_get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def search_people(name):
    url = "https://statsapi.mlb.com/api/v1/people/search?names=" + urllib.parse.quote(name)
    try:
        d = api_get(url)
    except Exception:
        return []
    return d.get("people", [])


def name_variants(english_name):
    parts = english_name.replace(".", ". ").split()
    parts = [p for p in parts if p]
    variants = []
    if english_name not in variants:
        variants.append(english_name)
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        if first_last not in variants:
            variants.append(first_last)
    # collapse "J. D." style back into "J.D."
    collapsed = english_name.replace(". ", ".")
    if collapsed not in variants:
        variants.append(collapsed)
    # some players (e.g. J.D. Martin, born "John Dale Martin") are registered
    # under first+middle initials rather than full given names
    if len(parts) >= 3:
        initials = "".join(p[0].upper() + "." for p in parts[:-1]) + " " + parts[-1]
        if initials not in variants:
            variants.append(initials)
    return variants


def find_person(english_name, birth_date):
    tried = set()
    for variant in name_variants(english_name):
        if variant in tried:
            continue
        tried.add(variant)
        people = search_people(variant)
        for p in people:
            if p.get("birthDate") == birth_date:
                return p, variant, "birth_date_match"
        time.sleep(0.3)
    # last name only fallback, filter by birth_date among returned candidates
    last = english_name.split()[-1]
    if last not in tried:
        people = search_people(last)
        for p in people:
            if p.get("birthDate") == birth_date:
                return p, last, "lastname_fallback_birth_date_match"
    return None, None, "unresolved"


def fetch_year_by_year(person_id, sport_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
        f"?stats=yearByYear&group=pitching&sportId={sport_id}"
    )
    try:
        d = api_get(url)
    except Exception:
        return []
    stats = d.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return []
    return stats[0]["splits"]


STAT_FIELDS = ["gamesPlayed", "gamesStarted", "inningsPitched", "era",
               "strikeOuts", "baseOnBalls", "homeRuns", "hitBatsmen"]
OUT_COLS = ["G", "GS", "IP", "ERA", "SO", "BB", "HR", "HBP"]


def parse_level(splits, kbo_year, level_label, player_name, kor_name, person_id):
    """Return list of output rows for one sport level (MLB or AAA)."""
    by_season = {}
    for s in splits:
        # MLB Stats API uses non-integer "season" labels (e.g. "2018.2") for
        # Mexican League (LMB) split-season stints filed under sportId=11
        # "AAA" (LMB is an official MLB Partner League at AAA classification).
        # Use the whole-year part for the KBO-year cutoff filter, but keep the
        # original season string in the output so these are identifiable.
        season_year = int(float(s["season"]))
        if season_year >= kbo_year:
            continue
        by_season.setdefault(s["season"], []).append(s)

    def league_of(s):
        return s.get("league", {}).get("name", "")

    rows = []
    for season in sorted(by_season):
        splits_for_season = by_season[season]
        team_rows = [s for s in splits_for_season if "team" in s]
        agg_rows = [s for s in splits_for_season if "team" not in s]

        for s in team_rows:
            row = {
                "선수명": kor_name, "english_name": player_name, "mlb_person_id": person_id,
                "level": level_label, "season": season, "league": league_of(s),
                "team": s["team"]["name"], "is_split_row": 1,
            }
            for f, col in zip(STAT_FIELDS, OUT_COLS):
                row[col] = s["stat"].get(f, "")
            rows.append(row)

        if agg_rows:
            s = agg_rows[0]
            leagues = sorted({league_of(t) for t in team_rows if league_of(t)})
            row = {
                "선수명": kor_name, "english_name": player_name, "mlb_person_id": person_id,
                "level": level_label, "season": season, "league": " / ".join(leagues),
                "team": " / ".join(t["team"]["name"] for t in team_rows) if team_rows else "",
                "is_split_row": 0,
            }
            for f, col in zip(STAT_FIELDS, OUT_COLS):
                row[col] = s["stat"].get(f, "")
            rows.append(row)
        elif len(team_rows) == 1:
            # single-team season: synthesize the is_split_row=0 total row
            s = team_rows[0]
            row = {
                "선수명": kor_name, "english_name": player_name, "mlb_person_id": person_id,
                "level": level_label, "season": season, "league": league_of(s),
                "team": s["team"]["name"], "is_split_row": 0,
            }
            for f, col in zip(STAT_FIELDS, OUT_COLS):
                row[col] = s["stat"].get(f, "")
            rows.append(row)
        # if len(team_rows) > 1 but no agg_row was returned (shouldn't happen), skip synth to avoid guessing

    return rows


def main():
    with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        roster = list(csv.DictReader(f))

    all_rows = []
    match_log = []

    for i, r in enumerate(roster, 1):
        kor_name = r["선수명"]
        eng_name = r["english_name"].strip()
        birth_date = r["birth_date"].strip()
        kbo_year = int(r["연도"])

        if not eng_name or not birth_date:
            match_log.append({"선수명": kor_name, "status": "skipped_no_english_name_or_birthdate"})
            print(f"[{i}/167] {kor_name}: SKIP (no english_name/birth_date)")
            continue

        person, used_variant, status = find_person(eng_name, birth_date)
        if not person:
            match_log.append({"선수명": kor_name, "status": "unresolved", "tried": eng_name})
            print(f"[{i}/167] {kor_name} ({eng_name}): UNRESOLVED")
            continue

        pid = person["id"]
        match_log.append({
            "선수명": kor_name, "status": status, "matched_variant": used_variant,
            "mlb_person_id": pid, "mlb_fullName": person.get("fullName"),
        })

        mlb_splits = fetch_year_by_year(pid, 1)
        time.sleep(0.2)
        aaa_splits = fetch_year_by_year(pid, 11)
        time.sleep(0.2)

        mlb_rows = parse_level(mlb_splits, kbo_year, "MLB", eng_name, kor_name, pid)
        aaa_rows = parse_level(aaa_splits, kbo_year, "AAA", eng_name, kor_name, pid)
        all_rows.extend(mlb_rows)
        all_rows.extend(aaa_rows)

        n_mlb_seasons = len({row["season"] for row in mlb_rows if row["is_split_row"] == 0})
        n_aaa_seasons = len({row["season"] for row in aaa_rows if row["is_split_row"] == 0})
        print(f"[{i}/167] {kor_name} ({eng_name}) -> id={pid} via '{used_variant}' | MLB {n_mlb_seasons}szn, AAA {n_aaa_seasons}szn")

    out_cols = ["선수명", "english_name", "mlb_person_id", "level", "season", "league", "team",
                "is_split_row", "G", "GS", "IP", "ERA", "SO", "BB", "HR", "HBP"]
    with open(f"{ROOT}/mlb_aaa_career_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(all_rows)

    log_cols = ["선수명", "status", "matched_variant", "mlb_person_id", "mlb_fullName", "tried"]
    with open(f"{ROOT}/mlb_stats_match_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_cols)
        writer.writeheader()
        for row in match_log:
            writer.writerow({k: row.get(k, "") for k in log_cols})

    print(f"\nDone. {len(all_rows)} rows written to mlb_aaa_career_stats.csv")
    print(f"Match log: {len(match_log)} entries written to mlb_stats_match_log.csv")


if __name__ == "__main__":
    main()
