# =========================================================
# BANKING CUSTOMER RISK SCORING USING ELASTIC NET
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("banking_risk_dataset.csv")

print(data.head())

# =========================
# FEATURES AND TARGET
# =========================

X = data[['Age',
          'Income',
          'Loan_Amount',
          'Credit_Score',
          'Account_Balance',
          'EMI',
          'Missed_Payments']]

y = data['Risk_Score']

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# ELASTIC NET MODEL
# =========================

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)

print("R2 Score:", r2)

# =========================
# RISK SCORE PREDICTION
# =========================

comparison = pd.DataFrame({
    'Actual Risk Score': y_test.values,
    'Predicted Risk Score': y_pred
})

print("\nCustomer Risk Prediction")

print(comparison)

# =========================
# FINANCIAL INDICATOR ANALYSIS
# =========================

print("\nFinancial Indicator Analysis")

print(data.describe())

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(10,5))

plt.plot(y_test.values,
         label='Actual Risk Score',
         marker='o')

plt.plot(y_pred,
         label='Predicted Risk Score',
         marker='x')

plt.title("Banking Customer Risk Scoring")

plt.xlabel("Customers")

plt.ylabel("Risk Score")

plt.legend()

plt.show()

# =========================
# FEATURE SELECTION
# =========================

print("\nFeature Importance")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# =========================
# FUTURE RISK FORECAST
# =========================

future_prediction = model.predict(X_test[:5])

print("\nFuture Customer Risk Scores")

print(future_prediction)

# =========================================================

# the model learns to predict the risk score of banking customers based on various 
# financial indicators and behaviors.
# By analyzing the coefficients, we can understand which features have the most significant
# impact on customer risk and make informed decisions for risk management and customer 
# segmentation in the banking industry
# Overall, this project demonstrates how Elastic Net Regression can be applied for 
# risk scoring in the financial domain, providing valuable insights for banks 
# and financial institutions.