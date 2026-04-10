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
   

9. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

10. 인사이트 제안

11. Reference
