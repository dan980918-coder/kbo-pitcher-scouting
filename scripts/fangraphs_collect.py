#!/usr/bin/env python3
"""
Collect season-by-season pitching stats (all levels: MLB/AAA/AA/A+/A/Rookie)
from FanGraphs player pages for all 167 players, through the season before
KBO entry.

Matching: FanGraphs' own player-search API (/api/search/players/) returns a
birthdate per candidate, so we match the same way as the MLB Stats API phase
-- exact birth_date against the roster's birth_date column.

Data source: FanGraphs' own Next.js data endpoint for a player's stats page
(discovered via the site's network requests), which returns the exact same
season-by-season table the page renders, as clean JSON -- no HTML scraping
needed. Confirmed via browser: KBO stat lines are also present in this feed,
but are deliberately excluded from output here since KBO performance is this
project's *target* variable, not a predictor.

Multi-team seasons: FanGraphs' own feed already carries a `type` field per
row -- type==0 is always the season's TOTAL row (whether the player was on
one team all year or several), and type>=1 rows are the individual team
splits that additionally appear for a traded player. We use the type==0
total row directly, exactly as instructed, with no manual weighted-average
recomputation. Both are kept in the output; type==0 rows are flagged
is_split_row=0, type>=1 rows is_split_row=1 (mirrors the MLB Stats API
schema from the previous phase).
"""
import csv
import json
import time
import urllib.request
import urllib.parse

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
UA = {"User-Agent": "Mozilla/5.0 (research script; polite rate-limited academic data collection)"}
NEXT_BUILD_ID = "qbZ3eQb-JrNVTfH49yXdu"
REQUEST_DELAY = 1.2

# birth_date is blank in the main roster for these two (STATIZ never had it);
# their identity was already confirmed against MLB Stats API in the prior
# phase, so we reuse that authoritative birthdate here for FanGraphs matching.
BIRTHDATE_OVERRIDE = {
    "파커 마켈": "1990-09-15",
    "에니 로메로": "1991-01-24",
}

# Players whose FanGraphs search-based match required manual resolution in
# the first collection pass (nickname-form registered names the automated
# name_variants() doesn't generate, e.g. "John Dale Martin" -> "J.D. Martin";
# or a birth_date in our roster that differs from FanGraphs' by 1 day/1-3
# years for a handful of players, cross-checked at the time against
# MLB Stats API's mlbDebutDate for plausibility). Hardcoded here so a re-run
# resolves them directly instead of repeating the two-pass manual patch.
FANGRAPHS_ID_OVERRIDE = {
    "재크 스튜어트": ("7397", "/players/zach-stewart/7397/stats/pitching"),
    "알렉산드로 마에스트리": ("sa328310", "/players/alessandro-maestri/sa328310/stats/pitching"),
    "제이콥 터너": ("10185", "/players/jacob-turner/10185/stats/pitching"),
    "채드 벨": ("10297", "/players/chad-bell/10297/stats/pitching"),
    "마이크 라이트": ("12586", "/players/mike-wright-jr/12586/stats/pitching"),
    "조시 스미스": ("10946", "/players/josh-a-smith/10946/stats/pitching"),
    "토마스 파노니": ("17281", "/players/thomas-pannone/17281/stats/pitching"),
    "케니 로젠버그": ("20009", "/players/kenny-rosenberg/20009/stats/pitching"),
    "파커 마켈": ("12106", "/players/parker-markel/12106/stats/pitching"),
    "에니 로메로": ("4001", "/players/enny-romero/4001/stats/pitching"),
}

# levels we keep in the season-by-season output; KBO is the project's target
# variable so it is explicitly excluded even though FanGraphs includes it in
# the same feed. MiLB is FanGraphs' own combined-level rollup for players who
# crossed levels in-season -- a different aggregation axis than the team
# is_split_row flag, so we skip it to avoid double counting.
KEEP_LEVELS = {"MLB", "AAA", "AA", "A+", "A", "A-", "R", "Rookie"}

STAT_FIELDS = ["G", "GS", "IP", "ERA", "K/9", "BB/9", "HR/9", "FIP", "xFIP", "WAR",
               "ERA-", "FIP-", "xFIP-"]

FIELD_KEY_MAP = {"ERA-": "ERA_minus", "FIP-": "FIP_minus", "xFIP-": "xFIP_minus"}


def field_key(f):
    return FIELD_KEY_MAP.get(f, f.replace("/", "_"))


LOW_SAMPLE_IP_THRESHOLD = 30


def ip_true_decimal(ip_raw):
    if ip_raw is None:
        return None
    whole = int(ip_raw)
    frac_digit = round((ip_raw - whole) * 10)
    return whole + frac_digit / 3.0


def api_get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def search_players(name):
    url = "https://www.fangraphs.com/api/search/players/?search=" + urllib.parse.quote(name)
    try:
        d = api_get(url)
    except Exception:
        return []
    return d.get("hits", [])


def name_variants(english_name):
    parts = [p for p in english_name.replace(".", ". ").split() if p]
    variants = [english_name]
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        if first_last not in variants:
            variants.append(first_last)
    collapsed = english_name.replace(". ", ".")
    if collapsed not in variants:
        variants.append(collapsed)
    if len(parts) >= 3:
        initials = "".join(p[0].upper() + "." for p in parts[:-1]) + " " + parts[-1]
        if initials not in variants:
            variants.append(initials)
    return variants


def find_player(english_name, birth_date):
    tried = set()
    for variant in name_variants(english_name):
        if variant in tried:
            continue
        tried.add(variant)
        for hit in search_players(variant):
            if hit.get("birthdate") == birth_date:
                return hit, variant, "birth_date_match"
        time.sleep(REQUEST_DELAY)
    last = english_name.split()[-1]
    if last not in tried:
        for hit in search_players(last):
            if hit.get("birthdate") == birth_date:
                return hit, last, "lastname_fallback_birth_date_match"
        time.sleep(REQUEST_DELAY)
    return None, None, "unresolved"


def fetch_player_stats(url_path):
    # url_path like "/players/erick-fedde/17425/stats/pitching"
    parts = url_path.strip("/").split("/")
    # ['players', 'erick-fedde', '17425', 'stats', 'pitching']
    slug, pid = parts[1], parts[2]
    next_url = (
        f"https://www.fangraphs.com/_next/data/{NEXT_BUILD_ID}/players/{slug}/{pid}/stats/pitching.json"
        f"?playerNameRoute={slug}&playerId={pid}"
    )
    d = api_get(next_url)
    return d.get("pageProps", {}).get("dataStats", {}).get("data", []), pid


def parse_rows(raw_rows, kbo_year, kor_name, eng_name, fg_id):
    # The `type` field means different things at different levels:
    # - AbbLevel == "MLB": type 0 is always the season TOTAL row (whether
    #   single- or multi-team); type >= 1 rows are the individual team
    #   splits that additionally appear when the player was traded.
    # - Any MiLB level (AAA/AA/A+/A/A-/R): type is a fixed negative
    #   level-rank code (e.g. AAA=-50, AA=-51, ...), NOT a split indicator.
    #   Usually there is exactly one row per (season, level) here (FanGraphs
    #   collapses an in-season same-level team change into one row, `team`
    #   showing the last team). BUT this does not always hold: a player who
    #   changed ORGANIZATIONS mid-season at the same level (e.g. Josh A.
    #   Smith, 2018: Red Sox AAA 74.0 IP + Mariners AAA 10.1 IP, found via
    #   the season/level FIP cross-check against MLB Stats API) gets two
    #   separate same-level rows with no matching same-level total -- the
    #   only total FanGraphs provides for that case is the AbbLevel=="MiLB"
    #   row, which rolls up across LEVELS too (not just teams), so it is not
    #   usable as a same-level total. When multiple rows share (season,
    #   level), we flag all of them is_split_row=1 rather than guessing
    #   which is picked as "the" total or synthesizing one -- rate stats
    #   (K/9, FIP, ...) cannot be validly summed across rows the way raw
    #   counting stats can.
    counts = {}
    for row in raw_rows:
        level = row.get("AbbLevel")
        season = row.get("aseason")
        if level in KEEP_LEVELS and level != "MLB" and season not in (None, 0):
            counts[(season, level)] = counts.get((season, level), 0) + 1

    out = []
    for row in raw_rows:
        level = row.get("AbbLevel")
        season = row.get("aseason")
        rtype = row.get("type")
        if level not in KEEP_LEVELS:
            continue
        if season is None or season == 0:
            continue
        if season >= kbo_year:
            continue
        if level == "MLB":
            if rtype is None or rtype < 0:
                continue
            is_split = 1 if rtype >= 1 else 0
        else:
            is_split = 1 if counts.get((season, level), 1) > 1 else 0
        ip_true = ip_true_decimal(row.get("IP"))
        out_row = {
            "선수명": kor_name, "english_name": eng_name, "fangraphs_id": fg_id,
            "level": level, "season": season, "team": row.get("ateam", ""),
            "is_split_row": is_split,
            "low_sample_flag": 1 if (ip_true is not None and ip_true < LOW_SAMPLE_IP_THRESHOLD) else 0,
        }
        for f in STAT_FIELDS:
            key = field_key(f)
            v = row.get(f, "")
            out_row[key] = v if v is not None else ""
        out.append(out_row)
    return out


def career_pre_kbo_mlb_war(raw_rows, kbo_year):
    total = 0.0
    n = 0
    has_any = False
    for row in raw_rows:
        if row.get("AbbLevel") != "MLB":
            continue
        if row.get("type") != 0:
            continue
        season = row.get("aseason")
        if season is None or season == 0 or season >= kbo_year:
            continue
        war = row.get("WAR")
        if war is None:
            continue
        has_any = True
        total += war
        n += 1
    return (round(total, 3) if has_any else ""), n


def main():
    with open(f"{ROOT}/data/raw/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        roster = list(csv.DictReader(f))

    all_rows = []
    war_summary = []
    match_log = []

    for i, r in enumerate(roster, 1):
        kor = r["선수명"]
        eng = r["english_name"].strip()
        bd = BIRTHDATE_OVERRIDE.get(kor, r["birth_date"].strip())
        kbo_year = int(r["연도"])

        if kor in FANGRAPHS_ID_OVERRIDE:
            pid, url = FANGRAPHS_ID_OVERRIDE[kor]
            hit, variant, status = {"id": pid, "url": url, "level": ["minor", "mlb"]}, eng, "id_override_from_prior_manual_resolution"
        elif not eng or not bd:
            match_log.append({"선수명": kor, "status": "skipped_no_english_name_or_birthdate"})
            print(f"[{i}/167] {kor}: SKIP (no english_name/birth_date)")
            continue
        else:
            hit, variant, status = find_player(eng, bd)

        if not hit:
            match_log.append({"선수명": kor, "status": "unresolved", "tried": eng})
            print(f"[{i}/167] {kor} ({eng}): UNRESOLVED")
            continue

        match_log.append({
            "선수명": kor, "status": status, "matched_variant": variant,
            "fangraphs_id": hit["id"], "fangraphs_name": hit.get("name", eng),
        })

        levels = hit.get("level") or []
        if "minor" not in levels and "mlb" not in levels:
            print(f"[{i}/167] {kor} ({eng}): matched but no minor/mlb level flag (KBO-only on FanGraphs) -- skipping stats fetch")
            war_summary.append({"선수명": kor, "english_name": eng, "fangraphs_id": hit["id"],
                                 "mlb_war_career_pre_kbo": "", "n_mlb_seasons_pre_kbo": 0})
            time.sleep(REQUEST_DELAY)
            continue

        try:
            raw_rows, pid = fetch_player_stats(hit["url"])
        except Exception as e:
            match_log[-1]["status"] = f"stats_fetch_failed:{e}"
            print(f"[{i}/167] {kor} ({eng}): stats fetch FAILED ({e})")
            time.sleep(REQUEST_DELAY)
            continue

        rows = parse_rows(raw_rows, kbo_year, kor, eng, pid)
        all_rows.extend(rows)

        war_total, n_war_szn = career_pre_kbo_mlb_war(raw_rows, kbo_year)
        war_summary.append({"선수명": kor, "english_name": eng, "fangraphs_id": pid,
                             "mlb_war_career_pre_kbo": war_total, "n_mlb_seasons_pre_kbo": n_war_szn})

        n_levels = len({row["level"] for row in rows if row["is_split_row"] == 0})
        print(f"[{i}/167] {kor} ({eng}) -> fg_id={pid} via '{variant}' | {len(rows)} rows across {n_levels} levels, MLB career WAR(pre-KBO)={war_total}")
        time.sleep(REQUEST_DELAY)

    out_cols = ["선수명", "english_name", "fangraphs_id", "level", "season", "team", "is_split_row",
                "low_sample_flag", "G", "GS", "IP", "ERA", "K_9", "BB_9", "HR_9", "FIP", "xFIP", "WAR",
                "ERA_minus", "FIP_minus", "xFIP_minus"]
    with open(f"{ROOT}/data/raw/fangraphs_career_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(all_rows)

    war_cols = ["선수명", "english_name", "fangraphs_id", "mlb_war_career_pre_kbo", "n_mlb_seasons_pre_kbo"]
    with open(f"{ROOT}/data/raw/fangraphs_mlb_war_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=war_cols)
        w.writeheader()
        w.writerows(war_summary)

    log_cols = ["선수명", "status", "matched_variant", "fangraphs_id", "fangraphs_name", "tried"]
    with open(f"{ROOT}/docs/fangraphs_match_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_cols)
        w.writeheader()
        for row in match_log:
            w.writerow({k: row.get(k, "") for k in log_cols})

    n_low = sum(1 for row in all_rows if row["low_sample_flag"] == 1)
    print(f"\nDone. {len(all_rows)} season rows -> fangraphs_career_stats.csv ({n_low} low_sample_flag=1, IP<{LOW_SAMPLE_IP_THRESHOLD})")
    print(f"{len(war_summary)} WAR summary rows -> fangraphs_mlb_war_summary.csv")
    print(f"{len(match_log)} match log entries -> fangraphs_match_log.csv")


if __name__ == "__main__":
    main()
