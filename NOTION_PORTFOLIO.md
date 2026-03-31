# 🏥 유방암 전이 진단 기간 예측 (WiDS Datathon 2024)

> **이진 분류**: 유방암 환자가 전이성 암 진단까지 90일 이내인지 예측

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **대회** | WiDS Datathon 2024 Challenge 1 — Equity in Healthcare |
| **기간** | 2025년 |
| **역할** | 개인 프로젝트 (데이터 분석, 특성 공학, 모델링 전담) |
| **데이터** | 환자 정보 + 우편번호 단위 인구통계 + 환경 데이터 |
| **타겟** | `DiagPeriodL90D` (1: 진단 ≤90일, 0: 진단 >90일) |
| **평가지표** | AUC-ROC |

---

## 🎯 문제 정의

유방암 환자가 전이성 암(Metastatic Cancer) 진단을 **90일 이내**에 받는지 예측.
조기 진단은 치료 효과에 결정적이며, 의료 접근성 격차(인종·지역·경제)가 진단 기간에 미치는 영향을 분석하는 것이 핵심 과제.

---

## 📂 데이터 구성

- **학습 데이터**: 12,906행 × 83컬럼
- **테스트 데이터**: 3,226행 × 82컬럼
- **주요 피처 그룹**
  - 환자 정보: 나이, 성별, 인종, 주(state), 우편번호, BMI, 보험 유형
  - 진단 코드: 유방암 ICD 코드/설명, 전이암 코드
  - 지역 인구통계: 소득, 교육 수준, 인종 구성, 집값, 실업률 등
  - 환경 데이터: NO₂, PM2.5, Ozone, 통근시간

---

## 🔧 특성 공학 (Feature Engineering)

### 1. 환경 오염 복합 지표 (`N02xPM25xcommute_time`)
- NO₂, PM2.5, 통근시간을 MinMax 스케일링 후 곱
- 대기오염 + 통근 부담의 복합적 건강 영향을 단일 피처로 압축

### 2. 결측치 보완 (NO₂, PM2.5, 통근시간)
- `patient_zip3` 앞 2자리(`zip2`) 기준 그룹 평균으로 대체
- 지역 수준에서의 환경 동질성을 가정

### 3. ICD 코드 버전 분류 (`ICD_version`)
- `C`로 시작 → ICD-10 / 그 외 → ICD-9
- 코딩 체계 차이가 진단 기간에 영향을 줄 수 있음

### 4. 종양 위치 추출 (`tumor_site`)
- 진단 설명(desc)에서 키워드 추출
- 5개 범주: Central / Inner / Outer / Overlapping / Unspecified
- 순서형 인코딩: Central(4) > Overlapping(3) > Inner(2) > Outer(1) > Unspecified(0)

### 5. 인종 결측치 보완 — Naive Bayes
- **전략**: 사후확률 P(race | tumor_site, zip3) ∝ P(tumor_site | race) × P(zip3 | race) × P(race)
  - `P(tumor_site | race)`: 논문 기반 인종별 종양 위치 확률
  - `P(zip3 | race)`: 데이터에서 계산한 지역별 인종 분포
  - `P(race)`: 데이터 사전 확률
- 사후확률 **≥ 0.8**인 경우만 대체 → 불확실한 추론 배제

### 6. 전이암 코드 파생 피처
- `clust`: 전이암 코드 길이 = 4 여부 (세분화 코드)
- `meta_code4`: 코드 앞 4자리 (부위별 그룹화)
- `is_female`: 진단 설명의 'female' 포함 여부

### 7. zip3 내 중복 피처 제거
- zip3 + 복합오염지표 그룹 내 분산 = 0인 컬럼 자동 제거

---

## 🤖 모델링

### 앙상블 아키텍처: Stacking

```
Level-1 Base Models                Level-2 Meta Model
┌──────────────┐
│  CatBoost    │ ──┐
├──────────────┤   │               ┌──────────────────────┐
│ RandomForest │ ──┼──► [pred1~4] ─► Logistic Regression │──► 최종 예측
├──────────────┤   │               └──────────────────────┘
│   XGBoost    │ ──┤
├──────────────┤   │
│  LightGBM    │ ──┘
└──────────────┘
```

### 모델 하이퍼파라미터

| 모델 | 주요 설정 |
|------|-----------|
| **CatBoost** | iterations=500, lr=0.05, depth=10, eval=AUC |
| **RandomForest** | n_estimators=200, max_depth=None |
| **XGBoost** | n_estimators=100, max_depth=6, subsample=0.9 |
| **LightGBM** | n_estimators=200, subsample=0.8, colsample_bytree=0.8 |
| **Meta (LR)** | 기본 설정 |

### 검증 전략
- **10-Fold Stratified K-Fold**: 클래스 불균형 대응
- 각 폴드에서 base model 학습 → test 예측 누적 → 폴드별 평균으로 최종 제출

---

## 📊 주요 인사이트

1. **환경 변수의 중요성**: NO₂, PM2.5가 높고 통근시간이 긴 지역의 환자일수록 진단 지연 경향
2. **인종별 종양 위치 차이**: 논문 데이터 기반, 인종마다 종양 위치 분포가 상이 → 결측 인종 추론에 활용
3. **ICD 코드 버전**: ICD-9 환자 비율이 낮아 별도 피처로 구분 시 신호 포착
4. **전이암 코드 세분화**: 5자리 코드(더 세분화된 진단) vs 4자리가 진단 기간에 영향

---

## 🛠️ 기술 스택

`Python` `Pandas` `NumPy` `Scikit-learn` `CatBoost` `XGBoost` `LightGBM` `Matplotlib` `Seaborn`

---

## 📁 프로젝트 구조

```
WiDS2024-Metastatic-Cancer/
├── WiDS2024_Metastatic_Cancer_Prediction.ipynb  # 최종 코드
├── requirements.txt                              # 의존성
├── .gitignore
└── 데이터/
    ├── training.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## 💡 회고

**잘한 점**
- 단순 결측치 평균 대체가 아닌, 도메인 지식(논문) + 베이즈 통계를 결합한 인종 결측치 보완
- 환경 오염 3가지 변수를 독립적으로 사용하지 않고 복합 지표로 압축

**개선 방향**
- 하이퍼파라미터 튜닝 (Optuna 등) 적용 시 성능 향상 가능
- 외부 인구통계 데이터 추가 결합
- SHAP 기반 모델 해석 추가
