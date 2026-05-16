# =========================================================
# CONSTRUCTION COST ESTIMATION USING RIDGE REGRESSION
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

data = pd.read_csv("construction_cost_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Material_Cost',
    'Labor_Cost',
    'Project_Duration_Days',
    'Equipment_Cost',
    'Transport_Cost',
    'Site_Area_Sqft'
]]

y = data['Total_Project_Cost']

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

# Ridge Regression handles correlated cost features
model = Ridge(alpha=1.0)

model.fit(X_train_scaled, y_train)

# =========================================================
# COST PREDICTION
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
# COST PREDICTION ENGINE
# =========================================================

new_project = pd.DataFrame({
    'Material_Cost': [250000],
    'Labor_Cost': [130000],
    'Project_Duration_Days': [130],
    'Equipment_Cost': [50000],
    'Transport_Cost': [25000],
    'Site_Area_Sqft': [5200]
})

new_project_scaled = scaler.transform(new_project)

predicted_cost = model.predict(new_project_scaled)

print("\nEstimated Construction Cost:")
print(predicted_cost[0])

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values,
         marker='o',
         label='Actual Cost')

plt.plot(y_pred,
         marker='s',
         label='Predicted Cost')

plt.title("Actual vs Predicted Construction Cost")
plt.xlabel("Test Samples")
plt.ylabel("Project Cost")

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
# COST FACTOR CORRELATION
# =========================================================

correlation_matrix = X.corr()

print("\nCorrelation Matrix")
print(correlation_matrix)

# =========================================================
# OVERFITTING CONTROL
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
# COST PREDICTION ENGINE COMPLETE
# =========================================================

print("\nConstruction Cost Prediction Engine Executed Successfully")

# the model learns to predict the total construction cost based on various cost factors such as material cost, labor cost, project duration, 
# equipment cost, transport cost, and site area.
# By analyzing the coefficients, we can understand which features have the most significant impact on construction costs and 
# make informed decisions for project budgeting and cost management.
# Overall, this project demonstrates how Ridge Regression can be applied to predict construction costs, providing valuable