# 모델 선택 (Baseline 단계) — 방법론 및 결정 기록

## 1. 데이터 분할

Temporal validation (PROJECT_GUIDELINES.md §3 원칙대로):

| 구분 | 조건 | n |
|---|---|---|
| Train | 2014~2023 KBO 입성 | 130 |
| Validation | 2024 KBO 입성 | 18 |
| Test (holdout, 미열람) | 2025 KBO 입성 | 17 |
| 제외 (타깃 결측) | `kbo_no_appearance=1` (파커 마켈, 에니 로메로) | 2 |

Test(2025)는 이번 모델링 단계 전체에서 한 번도 열람하지 않았다.

## 2. Baseline: 단일변수 선형회귀

`mlb_fip_last`(없으면 `aaa_fip_last`로 대체) 하나만 쓰는 가장 단순한 모델.

- 학습(n=127; MLB·AAA 기록이 둘 다 없는 3명은 변수 자체가 없어 제외):
  `WAR = 2.402 + (-0.0091) × baseline_fip`
- **Validation(2024): MAE=1.666, RMSE=1.900, R²=-0.002**
- "항상 train 평균만 예측"하는 무정보 모델과 사실상 구별 불가능(R²≈0) — `mlb_fip_last`의 WAR 상관 자체가 EDA-3에서 이미 r=-0.042로 거의 0이었으므로 예상된 결과.

## 3. Ridge 회귀

### 변수 (8개, `aaa_fip_last` 제외 — 아래 4번 참고)

`mlb_fip_last`, `mlb_fip_minus_career`, `aaa_hr9_last`, `aaa_bb9_3yr`, `age_at_kbo_entry`, `n_pitch_types_recorded`, `has_mlb_record`(더미), `has_aaa_record`(더미)

### 결측 처리

EDA-1에서 `mlb_*` 결측(27명, AAA 전용 선수)이 MCAR이 아니라 타깃과 유의미하게 연관됨을 이미 확인했으므로(Welch t-test p=0.0002), 단순 평균대치 대신:

1. `has_mlb_record`/`has_aaa_record` 더미(0/1) 추가 — "기록이 없다"는 사실 자체를 모델이 학습
2. 레벨별 수치 변수는 **해당 레벨 기록이 있는 선수들만의 train 평균**으로 대치 (더미가 있으므로 대치값 자체는 "정보 없음"을 표시하는 자리 표시자 역할)
3. MLB·AAA 기록이 모두 없는 3명(알렉산드로 마에스트리, 보 다카하시, 로니 윌리엄스)도 이 방식으로 포함 가능 — 두 더미 모두 0, 레벨 변수는 전부 대치값 → Baseline은 구조적으로 다룰 수 없었던 케이스를 Ridge는 커버함

### 다중공선성 처리

`aaa_fip_last`와 `aaa_hr9_last`의 train 상관이 r=0.933으로 사실상 같은 정보였고, Ridge에서 `aaa_fip_last` 계수 부호가 상식과 반대(+0.035)로 나와 다중공선성 증상으로 확인. 제외 전후 Validation 성능 차이가 미미(R² 0.074→0.068)해서 **`aaa_fip_last`를 최종 변수에서 제외**하고 8개 변수로 확정.

(별도로 `mlb_fip_last`/`mlb_fip_3yr`/`mlb_fip_davenport_translated_last` 간에도 r=0.947~0.976의 높은 상관이 확인되어 있음 — 최종 변수 조합에는 `mlb_fip_last` 하나만 포함해 이 문제를 피함.)

### 표준화 및 alpha 선택

StandardScaler로 표준화 후 `RidgeCV`(5-fold CV, `neg_mean_absolute_error` 기준)로 alpha 탐색. alpha 1~46 구간에서 CV MAE가 1.518~1.528로 완만한 평탄 구간을 이루고, alpha=100을 넘으면 뚜렷이 악화됨 — 최종 선택된 **alpha=9.25**는 이 안정 구간 안에 위치.

### 최종 계수 (표준화 변수 기준)

| 변수 | 계수 | 비고 |
|---|---|---|
| aaa_hr9_last | -0.441 | 방향 일치, 가장 크게 작용 |
| n_pitch_types_recorded | +0.383 | EDA-3 최강 신호와 일치 |
| aaa_bb9_3yr | -0.220 | 방향 일치 |
| has_mlb_record | +0.218 | |
| has_aaa_record | +0.185 | |
| mlb_fip_last | -0.112 | 방향 일치, 약함 |
| age_at_kbo_entry | +0.066 | 거의 무의미 |
| mlb_fip_minus_career | -0.007 | 사실상 무의미 |

## 4. Random Forest / Gradient Boosting — 시도했으나 기각

### 1차: 5-fold GridSearchCV (max_depth 2-5, n_estimators 50-200)

| 모델 | Val R² | Train R² | **Train-Val gap** |
|---|---|---|---|
| RF (depth5/n100/leaf3) | -0.072 | 0.611 | **0.683** |
| GB (depth3/n50/leaf10/lr0.05) | -0.065 | 0.499 | **0.564** |

**심각한 과적합.** "보수적으로 좁힌" 그리드 안에서도 CV가 depth=5 같은 과적합 소지가 큰 조합을 최적으로 골랐다 — n=130에 5-fold(fold당 26명)로는 CV 자체의 분산이 너무 커서 진짜 일반화되는 설정과 우연히 CV 폴드에 잘 맞은 설정을 구분하지 못했다.

### 2차: 극단 단순화 (그리드서치 없이 depth=2, n=20, leaf=10 고정)

Validation: MAE=1.468, RMSE=1.793, **R²=0.108** — 이 시점엔 Ridge(0.068)보다 좋아 보였다.

### 3차: 검증 — Repeated 3-fold × 5seed CV (15회 평가)로 재확인

좁힌 그리드(depth∈{2,3}, n_estimators∈{20,50,100}, min_samples_leaf∈{10,15,20})를 반복 CV로 다시 훑은 결과:

| 모델 | **CV R² (15회 평균)** | Val R² | **CV-Val 방향** |
|---|---|---|---|
| RF (극단단순, depth2/n20/leaf10) | **-0.005** | +0.108 | ❌ 불일치 |
| RF (재탐색 최선, depth3/n50/leaf10) | -0.005 | +0.087 | ❌ 불일치 |
| GB (재탐색 최선, depth3/n50/leaf10/lr0.05) | +0.009 | -0.065 | ❌ 불일치 |
| **Ridge (alpha=9.25)** | **+0.049** | +0.068 | **✅ 일치** |

**2차에서 봤던 RF의 Val R²=0.108은 반복 CV로 재현되지 않았다**(진짜 CV 성능은 -0.005, 사실상 무신호). Validation이 n=18로 매우 작다 보니, 유연한 트리 모델이 우연히 이 18명에 잘 맞아떨어질 가능성이 실재했고, 이번이 정확히 그 사례였다. **이건 이 프로젝트에 남기는 방법론적 교훈이다: 표본이 작을 때는 단일 Validation 성능만으로 모델을 비교/선택하면 안 되고, 반드시 반복 CV로 "우연히 잘 나온 것"과 "안정적으로 잘 나오는 것"을 구분해야 한다.**

RF/GB 세 후보 모두 CV 방향과 Val 방향이 어긋나 최종 후보에서 제외.

## 5. 최종 채택: Ridge (alpha=9.25)

CV(+0.049)와 Validation(+0.068)이 유일하게 같은 방향(양의 R²)을 가리키는 모델. 8개 변수, 결측 더미 처리 포함. 아티팩트: `reports/modeling/ridge_final.pkl` (모델+스케일러+대치값 전체, `scripts/modeling/06_train_final_ridge.py`로 재현 가능), `ridge_final.json`(사람이 읽을 수 있는 계수 요약).

## 6. 한계 — 그리고 이게 왜 이 프로젝트에 중요한 발견인가

R²가 0.05~0.07 수준으로 **절대적으로 낮다.** 해외 기록(MLB/AAA raw stat, FIP, 구종 데이터 존재 여부 등)만으로 KBO 첫 시즌 WAR을 설명하는 능력이 뚜렷한 한계가 있다는 뜻이다.

이건 실패가 아니라 **프로젝트 핵심 질문에 대한 유의미한 답**이다. PROJECT_GUIDELINES.md §3에 이미 명시했듯, 이 프로젝트는 "적응 성공을 예측"하는 게 아니라 "해외 기록으로 설명 가능한 KBO 기대성과"를 추정하고, 실제 성과와의 차이를 **KBO Translation Gap**으로 정의하는 문제로 설계했다. 이번 결과는 그 설계가 왜 필요한지를 데이터로 확인해준다 — 해외 성적만으로는 KBO 성과의 대부분이 설명되지 않으므로, 남는 부분(Translation Gap)이 오히려 분석의 핵심 대상이 된다. 낮은 R² 자체가 "해외 기록 → KBO 성과"라는 단순한 이식 가정이 성립하지 않는다는 증거다.

## 다음 단계 (차기 세션)

- 2025 holdout test 평가 (지금까지 한 번도 열람하지 않음)
- Translation Gap(실제 WAR − Ridge 예측 WAR) 계산 및 사례분석
- 스카우팅 리포트 작성
