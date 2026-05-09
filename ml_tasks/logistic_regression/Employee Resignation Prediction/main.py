# =========================================================
# EMPLOYEE RESIGNATION PREDICTION SYSTEM
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
    f1_score,
    confusion_matrix
)

from fpdf import FPDF

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("employee_resignation_dataset.csv")

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
    "Attendance_Percentage",
    "Salary",
    "Work_Pressure",
    "Overtime_Hours",
    "Satisfaction_Score"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Resigned"]

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
# CREATE MODEL
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

    actual = y_test.iloc[i]

    predicted = y_pred[i]

    print(f"Actual      : {actual}")

    print(f"Predicted   : {predicted}")

    print("--------------------------------------")

# =========================================================
# ACCURACY METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred) # How many predictions are correct.

precision = precision_score(y_test, y_pred) # How accurate resignation predictions are.

recall = recall_score(y_test, y_pred) # How many resigning employees were detected.

f1 = f1_score(y_test, y_pred) # Balanced performance score.

print("\n========== ACCURACY METRICS ==========\n")

print(f"Accuracy Score   : {accuracy:.2f}")

print(f"Precision Score  : {precision:.2f}")

print(f"Recall Score     : {recall:.2f}")

print(f"F1 Score         : {f1:.2f}")

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n========== EMPLOYEE RESIGNATION CHECK ==========\n")

attendance = float(
    input("Enter Attendance Percentage: ")
)

salary = float(
    input("Enter Salary: ")
)

work_pressure = int(
    input("Enter Work Pressure (1-10): ")
)

overtime = float(
    input("Enter Overtime Hours: ")
)

satisfaction = int(
    input("Enter Satisfaction Score (1-10): ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    attendance,
    salary,
    work_pressure,
    overtime,
    satisfaction
]], columns=[
    "Attendance_Percentage",
    "Salary",
    "Work_Pressure",
    "Overtime_Hours",
    "Satisfaction_Score"
])

# =========================================================
# PREDICT RESIGN / STAY
# =========================================================

prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

resign_probability = probability[0][1] * 100

stay_probability = probability[0][0] * 100

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:

    print("Prediction : LIKELY TO RESIGN")

else:

    print("Prediction : LIKELY TO STAY")

print(
    f"Resignation Probability : "
    f"{resign_probability:.2f}%"
)

print(
    f"Stay Probability        : "
    f"{stay_probability:.2f}%"
)

# =========================================================
# HR ALERT SYSTEM
# =========================================================

print("\n========== HR ALERT SYSTEM ==========\n")

if resign_probability > 70:

    print("ALERT: Employee may resign soon")

else:

    print("Employee retention is stable")

# =========================================================
# GRAPH 1
# SALARY VS RESIGNATION
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Salary"],
    df["Resigned"]
)

plt.xlabel("Salary")

plt.ylabel("Resigned")

plt.title("Salary vs Resignation")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# WORK PRESSURE VS RESIGNATION
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Work_Pressure"],
    df["Resigned"]
)

plt.xlabel("Work Pressure")

plt.ylabel("Resigned")

plt.title("Work Pressure vs Resignation")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# SATISFACTION ANALYSIS
# =========================================================

satisfaction_avg = df.groupby(
    "Satisfaction_Score"
)["Resigned"].mean()

print("\n========== SATISFACTION ANALYSIS ==========\n")

print(satisfaction_avg)

plt.figure(figsize=(8,5))

satisfaction_avg.plot(kind="bar")

plt.xlabel("Satisfaction Score")

plt.ylabel("Resignation Rate")

plt.title("Satisfaction vs Resignation")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 4
# OVERTIME ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Overtime_Hours"],
    df["Resigned"]
)

plt.xlabel("Overtime Hours")

plt.ylabel("Resigned")

plt.title("Overtime vs Resignation")

plt.grid(True)

plt.show()

# =========================================================
# HR DASHBOARD
# =========================================================

print("\n========== HR DASHBOARD ==========\n")

total_employees = len(df)

resigned_employees = df["Resigned"].sum()

staying_employees = total_employees - resigned_employees

average_salary = df["Salary"].mean()

resignation_percentage = (
    resigned_employees / total_employees
) * 100

print(f"Total Employees       : {total_employees}")

print(f"Likely Resigned       : {resigned_employees}")

print(f"Likely Staying        : {staying_employees}")

print(f"Average Salary        : {average_salary:.2f}")

print(
    f"Resignation Percentage : "
    f"{resignation_percentage:.2f}%"
)

# =========================================================
# SAVE DASHBOARD REPORT
# =========================================================

dashboard = pd.DataFrame({

    "Metric": [
        "Total Employees",
        "Likely Resigned",
        "Likely Staying",
        "Average Salary",
        "Resignation Percentage"
    ],

    "Value": [
        total_employees,
        resigned_employees,
        staying_employees,
        average_salary,
        resignation_percentage
    ]
})

dashboard.to_csv(
    "employee_hr_dashboard.csv",
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
    txt="Employee Resignation Prediction Report",
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

    result = "LIKELY TO RESIGN"

else:

    result = "LIKELY TO STAY"

pdf.cell(
    200,
    10,
    txt=f"Prediction: {result}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Resignation Probability: {resign_probability:.2f}%",
    ln=True
)

pdf.output("Employee_Resignation_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Employee_Resignation_Report.pdf")

# =========================================================
# LOGISTIC REGRESSION FORMULA
# =========================================================

print("\n========== LOGISTIC REGRESSION FORMULA ==========\n")

print("P(y=1) = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ... + bnxn))")
# =======================================================
# END OF PROJECT
# ========================================================

# P(y=1) is the probability of resignation.
# b0 is the intercept.
# b1, b2, ..., bn are coefficients for each feature.
# x1, x2, ..., xn are the input features.
# Example: P(resignation) = 1 / (1 + e^-(b0 + b1*Attendance + b2*Salary + ... + bn*Satisfaction))
# This formula calculates the probability of an employee resigning based on their features.
# Higher values of features that contribute to resignation will increase the probability, while higher values of features that contribute to staying will decrease it.
# The model learns the coefficients (b1, b2, ..., bn) during training to best fit the data and make accurate predictions.

# The model learns:
# Which factors increase resignation probability
# Which factors decrease resignation probability
# How strongly each factor affects resignation probability
# Example: If b1 (Attendance) is negative, it means higher attendance reduces resignation risk. If b2 (Salary) is positive, it means higher salary increases resignation risk (which may indicate dissatisfaction). The model uses these coefficients to make predictions based on employee data.
# This helps HR identify at-risk employees and take proactive measures to improve retention.
# By analyzing the coefficients, HR can also understand which factors are most influential in employee resignation and focus on improving those areas to enhance employee satisfaction and retention.

# Input Examples:
# Enter Attendance Percentage: 65
# Enter Salary: 30000
# Enter Work Pressure (1-10): 10
# Enter Overtime Hours: 12
# Enter Satisfaction Score (1-10): 1 

# Stay Prediction Input
# Enter Attendance Percentage: 95
# Enter Salary: 70000
# Enter Work Pressure (1-10): 2
# Enter Overtime Hours: 1
# Enter Satisfaction Score (1-10): 9