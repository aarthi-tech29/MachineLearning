# =========================================================
# SMART HEALTHCARE COST ANALYTICS USING ELASTIC NET
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

data = pd.read_csv("healthcare_cost_dataset.csv")

print(data.head())

# =========================
# FEATURES AND TARGET
# =========================

X = data[['age', 'bmi', 'children', 'smoker']]

y = data['expenses']

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
# ACTUAL VS PREDICTED
# =========================

comparison = pd.DataFrame({
    'Actual Expense': y_test.values,
    'Predicted Expense': y_pred
})

print("\nExpense Prediction")
print(comparison)

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(10,5))

plt.plot(y_test.values,
         label='Actual Expense',
         marker='o')

plt.plot(y_pred,
         label='Predicted Expense',
         marker='x')

plt.title("Hospital Expense Prediction")

plt.xlabel("Patients")

plt.ylabel("Medical Expense")

plt.legend()

plt.show()

# =========================
# FEATURE OPTIMIZATION
# =========================

print("\nFeature Importance")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# =========================
# FUTURE PREDICTION
# =========================

future_prediction = model.predict(X_test[:5])

print("\nFuture Hospital Expense Prediction")

print(future_prediction)

# =========================================================
# the model learns to predict medical expenses based on patient characteristics such as age,
#  BMI, number of children, and smoking status.
# By analyzing the coefficients, we can understand which features have the most significant 
# impact on medical expenses and make informed decisions for 
# healthcare cost management and personalized treatment plans.
# Overall, this project demonstrates how Elastic Net Regression can be applied for cost 
# prediction in the healthcare domain, providing valuable insights for patients,
#  healthcare providers, and policymakers.