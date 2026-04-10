#  IMBK_Bank_Customer_Churn_ML

1. 프로젝트명 : 고객 이탈 분류 ML 및 인사이트 분석

2. 기간 : 2026/04/10

3. 기술스택 
  - 데이터 전처리: Pandas, NumPy, Scikit-learn

  - 모델링/최적화: PyCaret, Optuna, LightGBM, CatBoost, AdaBoost

  - 시각화/해석: SHAP, Matplotlib 

5. 데이터 출처 : 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

6. 데이터 전처리
   - 불필요한 피처 제거 : customer_id 
   - 범주형 데이터 인코딩 : country와 gender
   - 데이터 분할 : X = df.drop('churn', axis=1), y = df['churn']
                   train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["churn"])
   - 피처 스케일링 : StandardScaler

7. EDA 및 해석
     고객들의 이탈률에는 상품의 보유 여부와 관련이 있을 것이라고 생각하여 ' 카드를 보유한 고객은 이탈률이 낮을 것이다.' 라는 가설을 들고 출발하였습니다. 그래서 신용카드 보유자와 신용카드 미보유자의 유지 및 이탈률을 확인을 해보았습니다.
     <img width="580" height="455" alt="cardnotget" src="https://github.com/user-attachments/assets/a002335a-7d73-43a4-93ac-aee171c08d19" />
 <img width="580" height="455" alt="cardget" src="https://github.com/user-attachments/assets/374b038f-2b1c-438f-affc-19529103d430" />
      데이터 내에서 신용카드 보유자의 수는 7,055명이고 유지 고객은 5,631명, 이탈고객은 1,424명이었고 신용카드 미보유자 수는 2,945명에 유지고객은 2,332명이고  이탈고객은 613명이었습니다.
      신용 카드의 보유 여부와는 관계 없이 이 두 집단의 이탈률을 25%와 26%로 차이는 미미 했으며 검증 결과 신용카드의 보유 여부는 고객의 이탈률과는 관계가 없다는 결론을 내렸습니다.
   

9. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

10. 인사이트 제안

11. Reference
