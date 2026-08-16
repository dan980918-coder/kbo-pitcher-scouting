# PROJECT_GUIDELINES.md

## 1. 프로젝트 개요

### 프로젝트명

**KBO 외국인 투수 성과 예측 / 스카우팅 분석**

### 프로젝트 목적

프로야구단 전력분석팀 및 외국인 선수 스카우트 직무를 염두에 두고, KBO 입성 이전 MLB/MiLB 성적과 선수 특성으로 신규 외국인 투수의 KBO 첫 시즌 성과를 예측할 수 있는지 검증한다.

### 핵심 질문

> KBO 입성 이전 MLB/MiLB 성적과 선수 특성으로 신규 외국인 투수의 KBO 첫 시즌 성과(주 타깃: STATIZ WAR)를 예측할 수 있는가?

### 분석 단위

* 한 행 = KBO에 **처음** 입성하는 외국인 투수의 KBO 첫 시즌
* 재계약 선수는 제외한다. 재계약 선수는 전년도 KBO 성적이라는 강한 정보를 이미 갖고 있어, 해외 기록만으로 기대 성과를 추정해야 하는 신규 영입 스카우팅 문제와 성격이 다르다.
* **"KBO 첫 시즌"의 기준일은 계약/신인지명 시점이 아니라 실제 1군 출전 기록이 존재하는 시즌이다.** STATIZ는 선수의 "신인지명" 연도(계약 시점)와 "활약연도"(실전 기록이 있는 시즌)를 다르게 표기하는 경우가 있는데, 이 프로젝트의 타깃 변수(WAR)가 붙는 시즌은 후자이므로 후자를 기준으로 한다.
  * 예: kt wiz 앤드루 시스코는 2014년에 kt와 계약(신인지명)했지만, 2014년 kt는 아직 KBO 1군에 진입하지 않아 2군 기록만 있다. 실제 1군 출전 기록은 2015년부터 존재하므로, 그의 KBO 첫 시즌은 **2015년**으로 확정한다.

### 대상 기간

* 2014~2025년 신규 외국인 투수 (167명 — 위키백과/나무위키 재구축 후 STATIZ 전수 대조로 확정. 상세 이력은 `docs/phase1_roster_draft_notes.md` 참고)
* 2026년 신규 외국인 투수는 학습에 포함하지 않고 실전 적용(Application) 사례로만 사용한다. 2025년은 완전한 holdout test로 유지한다.
* 2026년 아시아쿼터 투수는 기존 외국인 슬롯과 모집단(계약 조건, 리그 사용 방식 등)이 다를 수 있으므로, 메인 분석에서 분리하거나 제외를 검토한다. Phase 1 데이터 확인 후 결정한다.

---

## 2. 데이터 소스 원칙

### 수집 대상

* **MLB/MiLB(AAA 중심)**: MLB Stats API(`statsapi.mlb.com`)로 raw stat(ERA, IP, K, BB, HR, HBP, GS 등)을 수집한다.
* **파생지표(FIP/xFIP) 및 WAR**: FanGraphs 개별 선수 페이지(`/players/.../stats/pitching`)의 Minor Leagues 탭에서 공식값을 그대로 수집한다. robots.txt로 `/players/...` 경로가 막혀있지 않고, 무명 선수 다수로 확인한 결과 로그인/구독 요구 없이 K/9·BB/9·HR/9·FIP·xFIP가 AAA까지 전부 채워지는 것을 확인했다(WAR만 예외 — §데이터 처리 원칙의 AAA WAR 미제공 항목 참고). 최종 FIP/xFIP 값은 FanGraphs 공식값을 사용하며, MLB Stats API raw stat 기반 자체 계산 FIP는 최종 데이터셋 컬럼에 포함하지 않고 **검증 용도로만** 사용했다.
* **FIP 검증(raw stat 수집 로직 정합성 확인)**: MLB Stats API의 raw stat(HR/BB/HBP/K/IP)으로 FIP를 직접 계산(FanGraphs 공식 연도별 상수 cFIP 사용)해 FanGraphs 공식 FIP와 대조한 결과(`data/rosters/fip_verification.csv`, n=1,025행) — **MLB 레벨(n=473)은 평균 절대오차 0.0000으로 완전 일치**, raw stat 수집 로직이 정확함을 확인했다. **AAA 레벨(n=552)은 평균 절대오차 0.361, 부호있는 평균차 -0.360으로 거의 같고 552건 중 550건(99.6%)이 전부 같은 방향(자체 계산값이 낮음)으로 치우쳐 있다** — 노이즈가 아니라 사실상 일정한 상수 오프셋이며, MLB용 cFIP를 AAA 이닝에 그대로 적용해 생긴 체계적 차이로 판단한다(raw stat 자체의 문제가 아님, AAA 전용 상수를 다시 추정하는 작업은 진행하지 않음 — 최종 데이터셋은 FanGraphs 공식 FIP를 그대로 쓰므로 불필요).
* **KBO 첫 시즌 성과**(WAR 등): STATIZ에서 수집한다. 정적 HTML 스크래핑만 사용하고, robots.txt를 확인한 뒤 rate limit을 적용한다.
* **Baseball-Reference**: 이용약관상 자동 수집 대상에서 제외한다.

### 데이터 처리 원칙

* MLB와 AAA 기록은 리그 수준이 달라 합산하지 않고 별도 변수로 유지한다.
* 2020년 MiLB 기록은 코로나로 시즌이 취소된 구조적 결측이다. 일반 결측과 다르게 처리한다(`milb_2020_cancelled` 플래그 등).
* **KBO 정규시즌 0경기(가용성 실패) 처리**: 신규 외국인 투수로 계약했지만 부상 등으로 정규시즌에 단 한 경기도 출전하지 못한 선수는 모집단에서 제외하지 않는다(§3의 survivorship bias 방지 원칙과 동일한 이유). 다만 STATIZ는 이런 선수의 개인 페이지 자체를 생성하지 않는 것으로 확인되어(예: 2023 SSG 에니 로메로), WAR 값을 0으로 채우지 않고 `kbo_no_appearance=1` 플래그로 표시한 뒤 **결측(missing)** 으로 처리한다. 메인 WAR 회귀 타깃에서는 제외하고, "가용성(availability) 실패" 사례로 별도 집계한다.
* **nationality 잠정값**: `nationality` 컬럼은 STATIZ 프로필 박스에 해당 항목이 없어, 대신 영문 위키백과 인포박스에서 166/167명분을 채웠다(`nationality_source=wiki_provisional`). MLB Stats API 수집 단계에서 공식 필드(`birthCountry`)로 재검증 후 덮어쓸 예정이며, 그 전까지는 참고용 잠정값으로 취급한다.
* **birth_date 출처 구분**: `birth_date_source` 컬럼으로 값의 출처를 표시한다. 기본값은 `statiz`(165명). 파커 마켈·에니 로메로는 STATIZ 개인 페이지 자체가 없어(위 `kbo_no_appearance` 항목 참고) STATIZ에서 birth_date를 가져올 수 없었던 케이스로, MLB Stats API의 선수 검색 결과(고유 후보 확인)에서 가져와 `birth_date_source=mlb_stats_api`로 표시했다. 두 선수의 `english_name`·`throws`도 같은 MLB Stats API 조회에서 함께 채웠다.
* **AAA Statcast(구종별 usage%/구속/whiff%/run value) 미수집**: Baseball Savant의 공개 리더보드(`pitch-arsenal-stats`, `pitch-arsenals`)는 `minors=true` 파라미터를 붙여도 MLB 데이터와 완전히 동일한 응답을 반환하며, `statcast_search` raw pitch-level 엔드포인트도 AAA 전용 선수(예: 알렉 감보아, MLB person id 687941)를 조회하면 0건이 나온다. 즉 AAA(Triple-A) 수준의 Statcast/Hawk-Eye 구종 데이터는 공개 소스로 접근할 방법을 찾지 못했다. 구종 데이터는 MLB 레벨에서만 수집하며, AAA는 처음부터 전부 결측으로 처리한다.
* **MLB Statcast 구종 데이터의 연도 경계**: Baseball Savant 리더보드를 연도별로 직접 조회한 결과, `pitch-arsenal-stats`(usage%/whiff%/run value per 100)는 **2017 시즌부터** 데이터가 존재하고 2015~2016은 완전히 빈 응답이다. `pitch-arsenals`(구종별 평균 구속)는 Statcast 전면 도입 첫해인 **2015 시즌부터** 데이터가 존재한다. 이 때문에 선수의 "KBO 입성 직전 MLB 시즌"이 어느 구간에 속하느냐에 따라 구종 피처의 가용 범위가 다르며, 이는 `statcast_metrics_available` 플래그(`full`=2017+, `velo_only`=2015-2016, `none`=~2014 또는 MLB 기록 자체 없음)로 표시한다.
* **AAA WAR 미제공**: MLB Stats API·FanGraphs 둘 다 AAA(마이너리그) 레벨에는 WAR을 제공하지 않는다(FanGraphs 개별 선수 페이지의 Minor Leagues 탭에서 무명 선수 다수를 확인한 결과, K/9·BB/9·HR/9·FIP·xFIP는 AAA까지 전부 채워지지만 WAR 컬럼만 모든 마이너리그 행에서 예외 없이 공란이었다 — 페이월/결측이 아니라 애초에 계산·게시되지 않는 지표). 이에 따라 AAA WAR은 하나의 값으로 압축해서 대체하지 않고, 대신 AAA FIP·K-BB%·HR/9·커리어 누적 IP 등 원재료 지표를 모델 입력변수로 그대로 사용한다. 이 대체가 타당했는지는 이후 모델링 단계에서 feature importance로 검증한다.

---

## 3. 방법론 원칙

* 이 프로젝트는 "적응 성공 예측"이 아니라 **"해외 기록으로 설명 가능한 KBO 기대성과"**를 예측하는 문제로 설계한다. 실제 성과와의 차이는 **KBO Translation Gap(실제 − 예상)**으로 정의하며, 데이터 근거 없는 인과적 주장(멘탈, 적응력 등)은 하지 않는다.
* KBO 첫 시즌 IP(이닝) 기준으로 표본을 걸러내지 않는다. 조기 퇴출된 실패 사례를 survivorship bias로 제거하지 않는다. 기본 모집단은 "KBO 구단이 외국인 투수 슬롯으로 신규 영입한 선수" 전체다.
* 표본이 100~170명대로 작을 것으로 예상된다. 복잡한 딥러닝 대신 다음을 비교한다.
  * Baseline (최근 성적 + 나이 + 경력)
  * Ridge / Elastic Net
  * Random Forest / Gradient Boosting
  * 모든 모델은 Baseline과 반드시 비교한다.
* Validation은 temporal validation을 사용한다.
  * Train: 2014~2023
  * Validation: 2024
  * Test: 2025 (holdout)
  * Application: 2026

---

## 4. 작업 방식

* Phase 단위로 진행하고, 매 Phase마다 커밋/push 후 diff와 결과를 검토받는다. 검토 없이 다음 Phase로 넘어가지 않는다.
* 데이터 확인 전에 최종 변수/모델 구조를 임의로 확정하지 않는다.
* 중요한 선택지(분석 기간, 변수 포함 여부, 모델 종류 등)는 후보와 장단점을 먼저 제시하고 함께 결정한다.
* 데이터가 없는 부분을 가정으로 채우지 않는다.

---

## 5. Phase 로드맵

### Phase 1: Feasibility Test (진행 중)

모델링 전에 데이터 확보 가능성부터 검증한다.

1. 2014~2025 KBO 신규 외국인 투수 마스터 명단 재구축 (신규/재계약 구분)
2. MLB Stats API로 샘플 선수 5명의 MLB/AAA 시즌 데이터가 정상적으로 조회되는지 테스트
3. 위 5명의 STATIZ 페이지에서 KBO 첫 시즌 WAR/FIP/IP 결합 테스트
4. 결측률과 데이터 커버리지 확인
5. 결과를 정리해서 보고 — 이후 분석 시작연도, AA 확장 여부, 최종 변수 수, 모델 복잡도를 함께 결정한다

### 이후 Phase

Phase 1 결과 검토 후 확정한다.

---

## 6. 데이터 공개 원칙

* 원본 스크래핑 데이터(STATIZ HTML 등)는 재배포 조건이 불명확하므로 GitHub에 원본 그대로 업로드하지 않는다.
* 공개 저장소에는 가공된 파생 데이터셋과 분석 코드, 결과물만 포함한다.
* 데이터 출처와 수집 방식을 README에 명시한다.
