# =========================================================
# HOSPITAL RESOURCE PREDICTION USING RIDGE REGRESSION
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

data = pd.read_csv("hospital_resource_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# SELECT FEATURES AND TARGET
# =========================================================

X = data[[
    'Patients',
    'Available_Doctors',
    'Bed_Occupancy',
    'Emergency_Cases',
    'Nurses_Available',
    'Medical_Equipment'
]]

y = data['Resources_Required']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
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
# RESOURCE FORECASTING
# =========================================================

new_data = pd.DataFrame({
    'Patients': [550],
    'Available_Doctors': [65],
    'Bed_Occupancy': [100],
    'Emergency_Cases': [130],
    'Nurses_Available': [130],
    'Medical_Equipment': [95]
})

new_data_scaled = scaler.transform(new_data)

forecast = model.predict(new_data_scaled)

print("\nPredicted Hospital Resources Required:")
print(forecast[0])

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values, label='Actual Resources', marker='o')
plt.plot(y_pred, label='Predicted Resources', marker='s')

plt.title("Actual vs Predicted Hospital Resources")
plt.xlabel("Test Data")
plt.ylabel("Resources Required")
plt.legend()

plt.grid(True)

plt.show()

# =========================================================
# COEFFICIENTS DISPLAY
# =========================================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance")
print(coefficients)

# =========================================================
# OVERFITTING CONTROL CHECK
# =========================================================

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\nTraining Score:", train_score)
print("Testing Score :", test_score)

if abs(train_score - test_score) < 0.1:
    print("\nOverfitting Controlled Successfully using Ridge Regression")
else:
    print("\nModel may be Overfitting")

# the model learns to predict the required hospital resources based on various factors such as patient count, 
# available doctors, bed occupancy, emergency cases, nurses available, and medical equipment.
# By analyzing the coefficients, we can understand which features have the most significant impact on resource requirements
# and make informed decisions for hospital resource management.
# Overall, this project demonstrates how Ridge Regression can be applied to predict hospital resource needs,
# providing valuable insights for healthcare administrators and policymakers to optimize resource allocation and improve patient care.