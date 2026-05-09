# =========================================================
# HOSPITAL EMERGENCY RISK DETECTION SYSTEM
# LOGISTIC REGRESSION PROJECT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from fpdf import FPDF

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("hospital_emergency_dataset.csv")

print("\n========== DATASET ==========\n")

print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "Oxygen_Level",
    "Temperature"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Emergency"]

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
# CREATE LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression()

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")

# =========================================================
# TEST PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

print("\n========== TEST PREDICTIONS ==========\n")

for i in range(len(y_pred)):

    print(f"Actual      : {y_test.iloc[i]}")

    print(f"Predicted   : {y_pred[i]}")

    print("--------------------------------------")

# =========================================================
# ACCURACY METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred) # Correct predictions percentage.

precision = precision_score(y_test, y_pred) # How accurate emergency predictions are.

recall = recall_score(y_test, y_pred) # How many actual emergencies were correctly predicted.

f1 = f1_score(y_test, y_pred) # Balanced score combining precision and recall.

print("\n========== ACCURACY METRICS ==========\n")

print(f"Accuracy Score   : {accuracy:.2f}")

print(f"Precision Score  : {precision:.2f}")

print(f"Recall Score     : {recall:.2f}")

print(f"F1 Score         : {f1:.2f}")

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n========== PATIENT EMERGENCY CHECK ==========\n")

heart_rate = float(
    input("Enter Heart Rate: ")
)

systolic_bp = float(
    input("Enter Systolic BP: ")
)

diastolic_bp = float(
    input("Enter Diastolic BP: ")
)

oxygen = float(
    input("Enter Oxygen Level: ")
)

temperature = float(
    input("Enter Body Temperature: ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    heart_rate,
    systolic_bp,
    diastolic_bp,
    oxygen,
    temperature
]], columns=[
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "Oxygen_Level",
    "Temperature"
])

# =========================================================
# PREDICT EMERGENCY
# =========================================================

prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

emergency_probability = probability[0][1] * 100

normal_probability = probability[0][0] * 100

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:

    print("Prediction : EMERGENCY PATIENT")

else:

    print("Prediction : NORMAL PATIENT")

print(
    f"Emergency Probability : "
    f"{emergency_probability:.2f}%"
)

print(
    f"Normal Probability    : "
    f"{normal_probability:.2f}%"
)

# =========================================================
# DOCTOR NOTIFICATION SYSTEM
# =========================================================

print("\n========== DOCTOR ALERT SYSTEM ==========\n")

if emergency_probability > 70:

    print("ALERT: Immediate doctor attention required")

else:

    print("Patient condition is stable")

# =========================================================
# GRAPH 1
# HEART RATE VS EMERGENCY
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Heart_Rate"],
    df["Emergency"]
)

plt.xlabel("Heart Rate")

plt.ylabel("Emergency")

plt.title("Heart Rate vs Emergency")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# OXYGEN LEVEL VS EMERGENCY
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Oxygen_Level"],
    df["Emergency"]
)

plt.xlabel("Oxygen Level")

plt.ylabel("Emergency")

plt.title("Oxygen Level vs Emergency")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# BLOOD PRESSURE ANALYSIS
# =========================================================

bp_avg = df.groupby(
    "Emergency"
)["Systolic_BP"].mean()

print("\n========== BLOOD PRESSURE ANALYSIS ==========\n")

print(bp_avg)

plt.figure(figsize=(6,5))

bp_avg.plot(kind="bar")

plt.xlabel("Emergency")

plt.ylabel("Average Systolic BP")

plt.title("Emergency vs Blood Pressure")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 4
# RISK PROBABILITY CHART
# =========================================================

risk_chart = pd.DataFrame({

    "Category": [
        "Emergency Risk",
        "Normal Risk"
    ],

    "Probability": [
        emergency_probability,
        normal_probability
    ]
})

plt.figure(figsize=(6,5))

plt.bar(
    risk_chart["Category"],
    risk_chart["Probability"]
)

plt.xlabel("Risk Type")

plt.ylabel("Probability")

plt.title("Emergency Risk Probability")

plt.grid(True)

plt.show()

# =========================================================
# HOSPITAL DASHBOARD
# =========================================================

print("\n========== HOSPITAL DASHBOARD ==========\n")

total_patients = len(df)

emergency_cases = df["Emergency"].sum()

normal_cases = total_patients - emergency_cases

average_heart_rate = df["Heart_Rate"].mean()

emergency_percentage = (
    emergency_cases / total_patients
) * 100

print(f"Total Patients        : {total_patients}")

print(f"Emergency Cases       : {emergency_cases}")

print(f"Normal Cases          : {normal_cases}")

print(
    f"Average Heart Rate    : "
    f"{average_heart_rate:.2f}"
)

print(
    f"Emergency Percentage  : "
    f"{emergency_percentage:.2f}%"
)

# =========================================================
# SAVE DASHBOARD REPORT
# =========================================================

dashboard = pd.DataFrame({

    "Metric": [
        "Total Patients",
        "Emergency Cases",
        "Normal Cases",
        "Average Heart Rate",
        "Emergency Percentage"
    ],

    "Value": [
        total_patients,
        emergency_cases,
        normal_cases,
        average_heart_rate,
        emergency_percentage
    ]
})

dashboard.to_csv(
    "hospital_dashboard_report.csv",
    index=False
)

print("\nDashboard report saved successfully")

# =========================================================
# EXPORT PDF REPORT
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="Hospital Emergency Detection Report",
    ln=True,
    align='C'
)

pdf.ln(10)

pdf.set_font("Arial", size=12)

pdf.cell(
    200,
    10,
    txt=f"Accuracy Score: {accuracy:.2f}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Precision Score: {precision:.2f}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Recall Score: {recall:.2f}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"F1 Score: {f1:.2f}",
    ln=True
)

pdf.ln(10)

if prediction[0] == 1:

    result = "EMERGENCY PATIENT"

else:

    result = "NORMAL PATIENT"

pdf.cell(
    200,
    10,
    txt=f"Prediction: {result}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Emergency Probability: {emergency_probability:.2f}%",
    ln=True
)

pdf.output("Hospital_Emergency_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Hospital_Emergency_Report.pdf")

# =========================================================
# LOGISTIC REGRESSION FORMULA
# =========================================================

print("\n========== LOGISTIC REGRESSION FORMULA ==========\n")

print("P(y=1) = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ... + bnxn))")
# ================================================================
# END OF PROJECT
# ================================================================

# The model learns:
# emergency patterns
# normal patient patterns
# and calculates probabilities based on input features.

# Input features:
# Enter Heart Rate: 140
# Enter Systolic BP: 180
# Enter Diastolic BP: 110
# Enter Oxygen Level: 78
# Enter Body Temperature: 104