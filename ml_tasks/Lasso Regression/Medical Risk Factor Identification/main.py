# =========================================================
# MEDICAL RISK FACTOR IDENTIFICATION USING LASSO
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

data = pd.read_csv("medical_risk_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Age',
    'Blood_Pressure',
    'Cholesterol_Level',
    'Blood_Sugar',
    'BMI',
    'Smoking_Habit',
    'Exercise_Hours',
    'Heart_Rate',
    'Family_History'
]]

y = data['Disease_Risk']

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

# Lasso removes less important symptoms/features
model = Lasso(alpha=0.5)

model.fit(X_train_scaled, y_train)

# =========================================================
# DISEASE RISK PREDICTION
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
# NEW PATIENT RISK PREDICTION
# =========================================================

new_patient = pd.DataFrame({
    'Age': [58],
    'Blood_Pressure': [158],
    'Cholesterol_Level': [228],
    'Blood_Sugar': [119],
    'BMI': [34],
    'Smoking_Habit': [1],
    'Exercise_Hours': [1],
    'Heart_Rate': [95],
    'Family_History': [1]
})

new_patient_scaled = scaler.transform(new_patient)

predicted_risk = model.predict(new_patient_scaled)

print("\nPredicted Disease Risk:")
print(predicted_risk[0])

# =========================================================
# RISK LEVEL IDENTIFICATION
# =========================================================

risk_score = predicted_risk[0]

if risk_score >= 75:
    risk_level = "High Risk"
elif risk_score >= 45:
    risk_level = "Moderate Risk"
else:
    risk_level = "Low Risk"

print("\nDisease Risk Level:")
print(risk_level)

# =========================================================
# IMPORTANT SYMPTOM EXTRACTION
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nImportant Symptom Analysis")
print(importance)

selected_features = importance[importance['Coefficient'] != 0]

print("\nSelected Important Symptoms")
print(selected_features)

removed_features = importance[importance['Coefficient'] == 0]

print("\nRemoved Unnecessary Symptoms")
print(removed_features)

# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

plt.figure(figsize=(18,9))

plt.bar(importance['Feature'],
        importance['Coefficient'])

plt.title("Medical Risk Feature Importance using Lasso",
          fontsize=22)

plt.xlabel("Medical Features", fontsize=18, labelpad=20)
plt.ylabel("Coefficient Value", fontsize=18)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.subplots_adjust(left=0.18, bottom=0.35)

plt.grid(True)

plt.show()

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(y_test.values,
         marker='o',
         label='Actual Disease Risk')

plt.plot(y_pred,
         marker='s',
         label='Predicted Disease Risk')

plt.title("Actual vs Predicted Disease Risk")

plt.xlabel("Test Samples")
plt.ylabel("Disease Risk Score")

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

print("\nMedical Risk Factor Identification Completed Successfully")

# the model learns to predict disease risk based on various medical features and symptoms.
# By analyzing the coefficients, we can identify which symptoms have the most significant
#  impact on disease risk and provide personalized health recommendations for patients.
# Overall, this project demonstrates how Lasso Regression can be applied for feature selection and
# risk prediction in the medical domain, offering valuable insights for healthcare professionals and 
# patients alike.