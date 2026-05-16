# =========================================================
# EMPLOYEE PERFORMANCE FEATURE SELECTION USING LASSO
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("employee_performance_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Experience_Years',
    'Training_Hours',
    'Projects_Completed',
    'Attendance_Percentage',
    'Overtime_Hours',
    'Team_Meetings',
    'Client_Feedback',
    'Salary'
]]

y = data['Productivity_Score']

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
# LASSO REGRESSION MODEL
# =========================================================

# Lasso removes unnecessary features
model = Lasso(alpha=0.1)

model.fit(X_train_scaled, y_train)

# =========================================================
# PRODUCTIVITY PREDICTION
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
# NEW EMPLOYEE PREDICTION
# =========================================================

new_employee = pd.DataFrame({
    'Experience_Years': [12],
    'Training_Hours': [80],
    'Projects_Completed': [14],
    'Attendance_Percentage': [98],
    'Overtime_Hours': [22],
    'Team_Meetings': [15],
    'Client_Feedback': [94],
    'Salary': [78000]
})

new_employee_scaled = scaler.transform(new_employee)

predicted_productivity = model.predict(new_employee_scaled)

print("\nPredicted Productivity Score:")
print(predicted_productivity[0])

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance")
print(importance)

# =========================================================
# REMOVE UNNECESSARY FEATURES
# =========================================================

selected_features = importance[importance['Coefficient'] != 0]

print("\nSelected Important Features")
print(selected_features)

removed_features = importance[importance['Coefficient'] == 0]

print("\nRemoved Unnecessary Features")
print(removed_features)

# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

plt.figure(figsize=(18,9))

plt.bar(importance['Feature'],
        importance['Coefficient'])

plt.title("Feature Importance using Lasso Regression",
          fontsize=22)

plt.xlabel("Features", fontsize=18, labelpad=20)
plt.ylabel("Coefficient Value", fontsize=18)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

# Fix all text cutting
plt.subplots_adjust(left=0.18, bottom=0.40)

plt.grid(True)

plt.show()

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values,
         marker='o',
         label='Actual Productivity')

plt.plot(y_pred,
         marker='s',
         label='Predicted Productivity')

plt.title("Actual vs Predicted Productivity")
plt.xlabel("Test Samples")
plt.ylabel("Productivity Score")

plt.legend()
plt.grid(True)

plt.show()

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
# PROJECT COMPLETE
# =========================================================

print("\nEmployee Performance Prediction System Executed Successfully")

# the model learns to predict employee productivity based on various performance metrics and work-related factors.
# By analyzing the coefficients, we can understand which features have the most significant impact 
# on productivity and make informed decisions for employee development and performance improvement strategies.
# Overall, this project demonstrates how Lasso Regression can be applied for feature selection in employee performance
# prediction, providing valuable insights for HR professionals and organizational leaders.