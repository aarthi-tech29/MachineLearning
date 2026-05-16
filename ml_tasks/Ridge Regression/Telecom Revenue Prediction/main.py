# =========================================================
# TELECOM REVENUE PREDICTION USING RIDGE REGRESSION
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

data = pd.read_csv("telecom_revenue_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Monthly_Recharge',
    'Data_Usage_GB',
    'Call_Minutes',
    'SMS_Count',
    'Active_Days',
    'Customer_Support_Calls'
]]

y = data['Monthly_Revenue']

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

# Ridge helps handle correlated features
model = Ridge(alpha=1.0)

model.fit(X_train_scaled, y_train)

# =========================================================
# PREDICTION
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
# REVENUE FORECASTING
# =========================================================

new_customer = pd.DataFrame({
    'Monthly_Recharge': [1200],
    'Data_Usage_GB': [60],
    'Call_Minutes': [950],
    'SMS_Count': [170],
    'Active_Days': [30],
    'Customer_Support_Calls': [5]
})

new_customer_scaled = scaler.transform(new_customer)

predicted_revenue = model.predict(new_customer_scaled)

print("\nPredicted Telecom Revenue:")
print(predicted_revenue[0])

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values,
         marker='o',
         label='Actual Revenue')

plt.plot(y_pred,
         marker='s',
         label='Predicted Revenue')

plt.title("Actual vs Predicted Telecom Revenue")
plt.xlabel("Test Samples")
plt.ylabel("Revenue")

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
# CORRELATED FEATURE HANDLING
# =========================================================

correlation_matrix = X.corr()

print("\nFeature Correlation Matrix")
print(correlation_matrix)

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

# the model learns to predict telecom revenue based on various customer usage patterns and behaviors.
# By analyzing the coefficients, we can understand which features have the most significant impact on revenue and make informed decisions for customer retention and marketing strategies.
# Overall, this project demonstrates how Ridge Regression can be applied to predict telecom revenue, 
# providing valuable insights for telecom companies to optimize their services and enhance customer satisfaction.