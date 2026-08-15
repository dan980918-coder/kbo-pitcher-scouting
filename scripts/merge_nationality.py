#!/usr/bin/env python3
"""
Add nationality + nationality_source columns to
new_import_pitchers_2014_2025_draft_v2.csv.

Values compiled from the English Wikipedia infobox/lead-sentence info
already gathered during the asia_league_check.csv research pass (birthplace
/ "American professional baseball pitcher" style phrasing). Marked
"wiki_provisional" since it will be re-verified against MLB Stats API's
birthCountry field in the next phase and overwritten with the authoritative
value then. Players with no Wikipedia page (이안 맥키니) are left blank.
"""
import csv

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting/data/rosters"

NATIONALITY = {
    "크리스 볼스테드": "United States",
    "유네스키 마야": "Cuba",
    "J.D. 마틴": "United States",
    "코리 리오단": "United States",
    "에버렛 티포드": "United States",
    "하이로 어센시오": "Dominican Republic",
    "저스틴 토마스": "United States",
    "케일럽 클레이": "United States",
    "앤드루 앨버스": "Canada",
    "라이언 타투스코": "United States",
    "테드 웨버": "United States",
    "트레비스 밴와트": "United States",
    "로스 울프": "United States",
    "앤서니 스와잭": "United States",
    "타일러 클로이드": "United States",
    "알프레도 피가로": "Dominican Republic",
    "루카스 하렐": "United States",
    "조쉬 스틴슨": "United States",
    "에반 믹": "United States",
    "브룩스 레일리": "United States",
    "조쉬 린드블럼": "United States",
    "에스밀 로저스": "Dominican Republic",
    "재크 스튜어트": "United States",
    "앤드루 시스코": "United States",
    "필 어윈": "United States",
    "메릴 켈리": "United States",
    "콜린 벨레스터": "United States",
    "앨런 웹스터": "United States",
    "아놀드 레온": "Mexico",
    "요한 플란데": "Dominican Republic",
    "데이비드 허프": "United States",
    "헥터 노에시": "Dominican Republic",
    "지크 스프루일": "United States",
    "알렉산드로 마에스트리": "Italy",
    "파비오 카스티요": "Dominican Republic",
    "에릭 서캠프": "United States",
    "슈가 마리몬": "Colombia",
    "요한 피노": "Venezuela",
    "조쉬 로위": "United States",
    "브라울리오 라라": "Dominican Republic",
    "로버트 코엘로": "United States",
    "스콧 맥그레거": "United States",
    "재크 페트릭": "United States",
    "앤서니 레나도": "United States",
    "팻 딘": "United States",
    "파커 마켈": "United States",
    "닉 애디튼": "United States",
    "제프 맨쉽": "United States",
    "돈 로치": "United States",
    "스캇 다이아몬드": "Canada",
    "션 오설리반": "United States",
    "제이크 브리검": "United States",
    "세스 프랭코프": "United States",
    "팀 아델만": "United States",
    "리살베르토 보니야": "Dominican Republic",
    "타일러 윌슨": "United States",
    "펠릭스 듀브론트": "Venezuela",
    "키버스 샘슨": "United States",
    "제이슨 휠러": "United States",
    "데이비드 헤일": "United States",
    "로건 베렛": "United States",
    "왕웨이중": "Taiwan",
    "앙헬 산체스": "Dominican Republic",
    "덱 맥과이어": "United States",
    "저스틴 헤일리": "United States",
    "벤 라이블리": "United States",
    "케이시 켈리": "United States",
    "제이콥 터너": "United States",
    "조 윌랜드": "United States",
    "제이크 톰슨": "United States",
    "브록 다익손": "Canada",
    "워릭 서폴드": "Australia",
    "채드 벨": "United States",
    "드류 루친스키": "United States",
    "에디 버틀러": "United States",
    "크리스천 프리드릭": "United States",
    "윌리엄 쿠에바스": "Venezuela",
    "라울 알칸타라": "Dominican Republic",
    "에릭 요키시": "United States",
    "크리스 플렉센": "United States",
    "데이비드 뷰캐넌": "United States",
    "애런 브룩스": "United States",
    "드류 가뇽": "United States",
    "댄 스트레일리": "United States",
    "애드리안 샘슨": "United States",
    "마이크 라이트": "United States",
    "오드리사머 데스파이네": "Cuba",
    "닉 킹엄": "United States",
    "리카르도 핀토": "Venezuela",
    "아리엘 미란다": "Cuba",
    "워커 로켓": "United States",
    "앤드류 수아레즈": "United States",
    "대니얼 멩덴": "United States",
    "보 다카하시": "Brazil",
    "앤더슨 프랑코": "Venezuela",
    "라이언 카펜터": "United States",
    "웨스 파슨스": "United States",
    "윌머 폰트": "Venezuela",
    "아티 르위키": "United States",
    "샘 가빌리오": "United States",
    "조시 스미스": "United States",
    "로버트 스탁": "United States",
    "브랜던 워델": "United States",
    "앨버트 수아레즈": "Venezuela",
    "로니 윌리엄스": "United States",
    "션 놀린": "United States",
    "토마스 파노니": "United States",
    "찰리 반스": "United States",
    "글렌 스파크맨": "United States",
    "예프리 라미레즈": "Dominican Republic",
    "펠릭스 페냐": "Dominican Republic",
    "맷 더모디": "United States",
    "웨스 벤자민": "United States",
    "이반 노바": "Dominican Republic",
    "숀 모리만도": "United States",
    "타일러 애플러": "United States",
    "딜런 파일": "United States",
    "테일러 와이드너": "United States",
    "아도니스 메디나": "Dominican Republic",
    "숀 앤더슨": "United States",
    "마리오 산체스": "Venezuela",
    "애런 윌커슨": "United States",
    "버치 스미스": "United States",
    "리카르도 산체스": "Venezuela",
    "에릭 페디": "United States",
    "태너 털리": "United States",
    "보 슐서": "United States",
    "커크 맥카티": "United States",
    "에니 로메로": "Dominican Republic",
    "로에니스 엘리아스": "Cuba",
    "아리엘 후라도": "Panama",
    "이안 맥키니": None,
    "시라카와 케이쇼": "Japan",
    "조던 발라조빅": "Canada",
    "코너 시볼드": "United States",
    "데니 레예스": "Dominican Republic",
    "디트릭 엔스": "United States",
    "엘리에이저 에르난데스": "Venezuela",
    "윌 크로우": "United States",
    "캠 알드레드": "United States",
    "에릭 스타우트": "United States",
    "에릭 라우어": "United States",
    "제임스 네일": "United States",
    "다니엘 카스타노": "United States",
    "카일 하트": "United States",
    "라이언 와이스": "United States",
    "하이메 바리아": "Panama",
    "로버트 더거": "United States",
    "드류 앤더슨": "United States",
    "엔마누엘 데 헤이수스": "Venezuela",
    "콜 어빈": "United States",
    "잭 로그": "United States",
    "헤르손 가라비토": "Dominican Republic",
    "요니 치리노스": "Venezuela",
    "앤더스 톨허스트": "United States",
    "애덤 올러": "United States",
    "터커 데이비드슨": "United States",
    "빈스 벨라스케즈": "United States",
    "알렉 감보아": "United States",
    "코디 폰세": "United States",
    "로건 앨런": "United States",
    "라일리 톰프슨": "United States",
    "패트릭 머피": "United States",
    "미치 화이트": "United States",
    "케니 로젠버그": "United States",
    "라클란 웰스": "Australia",
    "C.C 메르세데스": "Dominican Republic",
}

with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    roster = list(reader)

assert set(NATIONALITY) == {r["선수명"] for r in roster}, "name set mismatch"

new_fieldnames = fieldnames + ["nationality_source"]

for r in roster:
    nat = NATIONALITY[r["선수명"]]
    if nat is None:
        r["nationality"] = ""
        r["nationality_source"] = ""
    else:
        r["nationality"] = nat
        r["nationality_source"] = "wiki_provisional"

with open(f"{ROOT}/new_import_pitchers_2014_2025_draft_v2.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(roster)

filled = sum(1 for r in roster if r["nationality"])
print(f"Filled nationality for {filled}/{len(roster)} players (blank: {len(roster)-filled})")
