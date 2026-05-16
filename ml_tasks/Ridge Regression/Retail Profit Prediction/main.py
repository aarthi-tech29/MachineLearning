# =========================================================
# RETAIL PROFIT PREDICTION USING RIDGE REGRESSION
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("retail_profit_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Product_Sales',
    'Marketing_Spend',
    'Seasonal_Index',
    'Customer_Visits',
    'Online_Orders',
    'Employee_Count'
]]

y = data['Profit']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# RIDGE REGRESSION MODEL
# =========================================================

# Ridge Regression helps reduce overfitting
model = Ridge(alpha=1.0)

model.fit(X_train_scaled, y_train)

# =========================================================
# PROFIT PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)

# =========================================================
# MODEL EVALUATION
# =========================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# =========================================================
# PROFIT ESTIMATION ENGINE
# =========================================================

new_data = pd.DataFrame({
    'Product_Sales': [150000],
    'Marketing_Spend': [50000],
    'Seasonal_Index': [3.2],
    'Customer_Visits': [3200],
    'Online_Orders': [800],
    'Employee_Count': [35]
})

new_data_scaled = scaler.transform(new_data)

predicted_profit = model.predict(new_data_scaled)

print("\nEstimated Retail Profit:")
print(predicted_profit[0])

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values,
         marker='o',
         label='Actual Profit')

plt.plot(y_pred,
         marker='s',
         label='Predicted Profit')

plt.title("Actual vs Predicted Retail Profit")
plt.xlabel("Test Samples")
plt.ylabel("Profit")

plt.legend()
plt.grid(True)

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients")
print(importance)

# =========================================================
# SEASONAL TREND ANALYSIS
# =========================================================

seasonal_profit = data.groupby('Seasonal_Index')['Profit'].mean()

print("\nSeasonal Trend Analysis")
print(seasonal_profit)

# =========================================================
# OVERFITTING CHECK
# =========================================================

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\nTraining Score:", train_score)
print("Testing Score :", test_score)

if abs(train_score - test_score) < 0.1:
    print("\nOverfitting Controlled Successfully")
else:
    print("\nPossible Overfitting Detected")

# =========================================================
# RETAIL PROFIT PREDICTION COMPLETE
# =========================================================

print("\nRetail Profit Prediction Engine Executed Successfully")

# the model learns to predict retail profit based on various factors such as product sales, 
# marketing spend, seasonal index, customer visits, online orders, and employee count.
# By analyzing the coefficients, we can understand which features have the most significant impact on profit and make informed decisions for business strategy and resource allocation.
# Overall, this project demonstrates how Ridge Regression can be applied to predict retail profit,
#  providing valuable insights for retail managers and stakeholders.