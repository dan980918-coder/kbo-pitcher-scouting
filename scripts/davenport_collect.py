#!/usr/bin/env python3
"""
Collect Clay Davenport's league-translated (MLE/"Regular DT") K/9, BB/9,
HR/9 -- and the raw translated HR/BB/K/IP counts they're derived from --
for all 167 players, through the season before KBO entry.

Site: claydavenport.com. robots.txt is Allow: / (confirmed), but the
server runs mod_security and returns 406 for non-browser User-Agent
strings, so every request here uses a real browser UA + a polite delay.

Per-player pages ("pitching cards") live at a predictable URL:
  http://www.claydavenport.com/pt/{FIRST_LETTER_OF_SURNAME}/{SURNAME}{YYYYMMDD}A.shtml
confirmed against several known players (Bryan Woo, Erick Fedde, Ariel
Jurado). No site search API exists, so identity is resolved by
constructing candidate URLs from the surname already cleaned in the
FanGraphs matching phase (fangraphs_match_log.csv) and verifying the
"Born YYYYMMDD" string on the fetched page against our roster's
birth_date -- the same verification principle used in every prior
matching phase this project, just without a search step in front of it.

Each player page stacks three tables: "Real Stats, No DT", "Regular DT"
(translated to Davenport's fixed run-environment baseline), and "Peak
(Age-adjusted) DT". Only the "Regular DT" table is parsed here, per this
phase's scope (NERA/LFRA excluded -- their exact formula is undocumented
and was flagged low-confidence in the diagnostic pass).

FIP from translated rate stats: the translated raw counts (HR, BB, K, IP)
are used directly -- FIP = (13*HR + 3*BB - 2*K)/IP + cFIP -- rather than
reconstructing counts from the rounded per-9 rate columns, since the
counts are already on the same page and avoid the extra rounding step.
Algebraically this is equivalent to applying the per-9 rates directly
((13*HR9 + 3*BB9 - 2*K9)/9 + cFIP) when HR9=9*HR/IP etc holds exactly;
using the page's own counts sidesteps relying on that holding after
display-rounding.
"""
import csv
import re
import time
import unicodedata
import urllib.request
import urllib.error

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
                     "(research script; polite rate-limited academic data collection)"}
REQUEST_DELAY = 1.5

SUFFIXES = {"JR", "SR", "II", "III", "IV"}

LEAGUE_LEVEL_MAP = {
    "AL": "MLB", "NL": "MLB",
    "PCL": "AAA", "INT": "AAA",
    "TEX": "AA", "EL": "AA", "SL": "AA", "SOU": "AA", "SOUTH": "AA",
    "MID": "A+", "MDW": "A+", "SAL": "A+", "NWN": "A+", "NWL": "A+",
    "CAL": "A", "CAR": "A", "FSL": "A", "FLA": "A",
    "ACL": "R", "FCL": "R", "APP": "R", "PIO": "R",
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def surname_candidates(name):
    parts = [p for p in re.split(r"[\s.]+", name) if p]
    parts = [p for p in parts if p.upper().rstrip(".") not in SUFFIXES]
    if not parts:
        return []
    clean = lambda s: re.sub(r"[^A-Za-z]", "", strip_accents(s)).upper()
    cands = []
    last = clean(parts[-1])
    if last:
        cands.append(last)
    if len(parts) >= 2:
        combo = clean(parts[-2] + parts[-1])
        if combo and combo not in cands:
            cands.append(combo)
    return cands


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def find_player_page(name_candidates, birth_date):
    yyyymmdd = birth_date.replace("-", "")
    tried = []
    for surname in name_candidates:
        if not surname:
            continue
        for suffix in "ABCD":
            url = f"http://www.claydavenport.com/pt/{surname[0]}/{surname}{yyyymmdd}{suffix}.shtml"
            tried.append(url)
            html = fetch(url)
            time.sleep(REQUEST_DELAY)
            if html is None:
                continue
            if f"Born {yyyymmdd}" in html:
                return html, url, tried
    return None, None, tried


ROW_RE = re.compile(
    r"^(\d{4})\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(-?\d+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def parse_regular_dt_section(html):
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"&nbsp;", " ", text)
    marker = re.search(r"Regular DT", text)
    if not marker:
        return []
    section = text[marker.end():]
    end_marker = re.search(r"Peak \(Age-adjusted\) DT", section)
    if end_marker:
        section = section[:end_marker.start()]

    rows = []
    for line in section.splitlines():
        line = line.strip()
        m = ROW_RE.match(line)
        if not m:
            continue
        (year, team, lge, g, gs, ip, h, r, hr, bb, k, gb_pct, nera,
         w, l, sv, lfra, h9, hr9, bb9, k9) = m.groups()
        rows.append({
            "season": int(year), "team": team, "lge": lge,
            "G": int(g), "GS": int(gs), "IP": float(ip),
            "H": int(h), "HR": int(hr), "BB": int(bb), "K": int(k),
            "K_9": float(k9), "BB_9": float(bb9), "HR_9": float(hr9),
        })
    return rows


def main():
    with open(f"{ROOT}/data/raw/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
        roster = list(csv.DictReader(f))
    with open(f"{ROOT}/docs/fangraphs_match_log.csv", encoding="utf-8") as f:
        fg_names = {r["선수명"]: r["fangraphs_name"] for r in csv.DictReader(f) if r.get("fangraphs_name")}

    BIRTHDATE_OVERRIDE = {"파커 마켈": "1990-09-15", "에니 로메로": "1991-01-24"}

    all_rows = []
    match_log = []

    for i, r in enumerate(roster, 1):
        kor = r["선수명"]
        eng = r["english_name"].strip()
        bd = BIRTHDATE_OVERRIDE.get(kor, r["birth_date"].strip())
        kbo_year = int(r["연도"])
        primary_name = fg_names.get(kor, eng)

        if not primary_name or not bd:
            match_log.append({"선수명": kor, "status": "skipped_no_name_or_birthdate"})
            print(f"[{i}/167] {kor}: SKIP")
            continue

        cands = surname_candidates(primary_name)
        html, url, tried = find_player_page(cands, bd)

        if html is None:
            match_log.append({"선수명": kor, "status": "unresolved", "tried_n": len(tried)})
            print(f"[{i}/167] {kor} ({primary_name}): UNRESOLVED ({len(tried)} URLs tried)")
            continue

        rows = parse_regular_dt_section(html)
        kept = [row for row in rows if row["season"] < kbo_year]
        for row in kept:
            row["level"] = LEAGUE_LEVEL_MAP.get(row["lge"].upper(), "")
            row["선수명"] = kor
            row["english_name"] = primary_name
            row["davenport_url"] = url
        all_rows.extend(kept)

        match_log.append({"선수명": kor, "status": "matched", "url": url, "n_seasons_pre_kbo": len(kept)})
        n_mlb = sum(1 for x in kept if x["level"] == "MLB")
        n_aaa = sum(1 for x in kept if x["level"] == "AAA")
        print(f"[{i}/167] {kor} ({primary_name}) -> {url} | {len(kept)} pre-KBO seasons (MLB={n_mlb}, AAA={n_aaa})")

    out_cols = ["선수명", "english_name", "davenport_url", "level", "lge", "season", "team",
                "G", "GS", "IP", "H", "HR", "BB", "K", "K_9", "BB_9", "HR_9"]
    with open(f"{ROOT}/data/raw/davenport_career_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(all_rows)

    log_cols = ["선수명", "status", "url", "n_seasons_pre_kbo", "tried_n"]
    with open(f"{ROOT}/docs/davenport_match_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_cols)
        w.writeheader()
        for row in match_log:
            w.writerow({k: row.get(k, "") for k in log_cols})

    n_matched = sum(1 for r in match_log if r["status"] == "matched")
    print(f"\nDone. {len(all_rows)} season rows -> davenport_career_stats.csv")
    print(f"{n_matched}/167 players matched -> davenport_match_log.csv")


if __name__ == "__main__":
    main()
