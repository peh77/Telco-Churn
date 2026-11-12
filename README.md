# Telco Customer Churn: 특성 공학 및 모델 성능 비교

Telco Customer Churn 데이터 7,043건을 대상으로 결측치 처리, 인코딩, 스케일링, 파생 변수 및 변수 선택에 따른 분류 성능을 비교한다. Logistic Regression과 Random Forest를 사용하며, 동일 모델 기준선 비교와 중첩 교차검증을 수행한다.

## 실행 환경

- Windows, Python 3.14.5
- scikit-learn 1.9.1
- 상세 패키지 버전: `requirements.txt`
- 데이터 출처: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 실행 방법

프로젝트 루트에서 다음 순서로 실행한다.

```powershell
python -m pip install -r requirements.txt
python rerun_notebooks.py
python complete_assignment.py
python robust_evaluation.py
```

입력 파일은 `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`에 배치한다. 노트북 자동 실행에는 `python3` 커널이 필요하다. 커널이 등록되어 있지 않은 경우 다음 명령을 사용한다.

```powershell
python -m ipykernel install --user --name python3
```

## 분석 구성

| 파일 | 분석 내용 |
|---|---|
| `Telco_Churn_STEP01.ipynb` | 데이터 구조, 컬럼 설명, 타겟 정의 |
| `Telco_Churn_STEP02_EDA.ipynb` | 결측치·중복·이상치·분포·상관관계·범주별 이탈률 |
| `Telco_Churn_STEP03_Feature_Engineering.ipynb` | 전처리 개별 비교 및 파생 변수 추가 전후 비교 |
| `Telco_Churn_STEP04_Feature_Selection.ipynb` | RF 중요도와 전체·Top 5·10·15 특성 비교 |
| `Telco_Churn_STEP05_Model_Training_Evaluation.ipynb` | 두 모델의 GridSearchCV 및 탐색적 홀드아웃 평가 |
| `complete_assignment.py` | 지정 전처리 조합, RF Top 10 선택, 모델별 스케일링 CV |
| `robust_evaluation.py` | 동일 모델 기준선 비교 및 5×3 중첩 교차검증 |
| `rerun_notebooks.py` | 노트북 순차 실행과 출력·그래프 저장 |
| `validate_robust_results.py` | OOF 예측, fold별 지표, 테스트 분리 및 보고서 수치 검증 |

## 실험 및 검증 방법

### 개별 전처리 비교

결측치 대체 3종, 인코딩 2종, 스케일링 3종과 파생 변수 3개를 비교한다. 개별 비교에서는 나머지 조건을 고정한다. 입력 특성의 Label 방식은 `OrdinalEncoder`로 구현한다. `SeniorCitizen`은 의미상 이진 범주형이지만 수치형 전처리 경로에 포함한다. `AvgChargeRatio`의 분모가 0인 경우에는 NaN으로 유지한 후 학습 데이터의 통계량으로 대체한다.

### 기준선과 전처리 조합

- 무전처리 Base: `DummyClassifier(strategy='prior')`. 모든 원본 입력을 유지하고 전처리와 변수 선택을 생략하는 다수 클래스 기준선이다.
- Base-minimal: 같은 학습 모델에 Median+One-Hot만 적용한다. 스케일링·파생 변수·변수 선택은 생략한다.
- Exp-1: Mean+One-Hot+Standard, 파생 변수 사용, 변수 선택 없음.
- Exp-2: Median+Ordinal+MinMax, 파생 변수 사용, RF Top 10 선택.
- Exp-3: Most Frequent+One-Hot+Robust, 파생 변수 사용, RF Top 10 선택.

DummyClassifier와 학습 모델의 차이를 전처리의 순수 효과로 해석하지 않는다. 동일 모델 내 비교에는 Base-minimal을 사용한다. 여러 설정이 함께 달라지는 조합 비교와 단일 전처리의 통제 비교도 구분한다.

### 중첩 교차검증

기존 20% 테스트셋 1,409명은 제외하고 원래 학습 데이터 5,634명만 사용한다. 바깥 Stratified 5-Fold와 안쪽 Stratified 3-Fold를 구성한다. 바깥 난수 시드는 2026, 안쪽 난수 시드는 42다. 각 학습 부분 내부에서 전처리와 변수 선택을 적합하며, 안쪽 CV F1로 조합과 모델을 선택한 후 바깥 fold에서 평가한다.

Logistic Regression은 `C=0.1/1.0`, Random Forest는 `min_samples_leaf=1/3`을 비교한다. 모델당 조합 4개와 하이퍼파라미터 2개를 결합한 8개 후보를 탐색한다. 무스케일 Logistic Regression의 수렴을 위해 보완 검증 전체에 `solver='newton-cholesky'`, `max_iter=1000`을 사용한다. `ConvergenceWarning`은 오류로 처리한다. 수렴 경고가 발생했던 lbfgs 예비 결과는 `preliminary_lbfgs_convergence_warnings`에 별도로 보관하며 최종 결과로 사용하지 않는다.

주 평가 대상은 각 바깥 fold에서 안쪽 점수로 모델까지 선택하는 `Inner-selected procedure`다. 바깥 점수가 가장 높은 모델을 사후에 다시 선택한 결과와 구분한다. fold 표준편차는 신뢰구간이 아니다.

## 결과 파일

- `results/rerun/`: 실행 출력이 포함된 노트북과 그래프
- `results/assignment_completion/`: 지정 조합 및 모델별 스케일링 결과
- `results/robust_evaluation/fixed_model_comparison.csv`: 동일 모델 통제 비교
- `results/robust_evaluation/nested_fold_results.csv`: 바깥 fold별 성능과 선택 설정
- `results/robust_evaluation/nested_summary.csv`: 중첩 CV 요약
- `results/robust_evaluation/nested_oof_predictions.csv`: 바깥 fold OOF 예측
- `results/robust_evaluation/protocol.json`: 평가 설정
- `results/robust_evaluation/validation.json`: 산출물 검증 결과

## 해석 범위

STEP03과 STEP05는 동일 테스트 분할을 사용한 탐색 결과다. 중첩 CV는 기존 테스트셋을 제외하고 설정 선택과 평가를 분리하지만, 선행 EDA와 후보 설계가 동일 데이터셋을 참고했다는 한계는 남는다. 일반화 성능의 추가 확인에는 신규 외부 데이터가 필요하다. 중요도 순위는 인과적 영향력이나 변수 추가의 성능 개선을 보장하지 않는다.

## 문서 생성 도구

`build_robust_report.py`는 분석 결과를 Word에 반영한다. `export_report_pdf.py`와 `review_robust_pdf.py`는 PDF 변환 및 검토에 사용한다. 문서 도구는 분석 코드와 구분하며, Word 변환에는 Windows와 Microsoft Word가 필요하다.
