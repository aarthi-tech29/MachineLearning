# =========================================================
# UNIVERSITY ADMISSION ELIGIBILITY PREDICTOR
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

df = pd.read_csv(
    "university_admission_dataset.csv"
)

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
    "Academic_Marks",
    "Entrance_Score",
    "Reservation_Category",
    "Interview_Marks"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Eligible"]

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

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n========== ACCURACY METRICS ==========\n")

print(f"Accuracy Score   : {accuracy:.2f}")

print(f"Precision Score  : {precision:.2f}")

print(f"Recall Score     : {recall:.2f}")

print(f"F1 Score         : {f1:.2f}")

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n========== ADMISSION ELIGIBILITY CHECK ==========\n")

academic_marks = float(
    input("Enter Academic Marks: ")
)

entrance_score = float(
    input("Enter Entrance Exam Score: ")
)

print("\nReservation Category")
print("0 = General")
print("1 = OBC")
print("2 = SC/ST")

reservation = int(
    input("Enter Reservation Category: ")
)

interview_marks = float(
    input("Enter Interview Marks: ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    academic_marks,
    entrance_score,
    reservation,
    interview_marks
]], columns=[
    "Academic_Marks",
    "Entrance_Score",
    "Reservation_Category",
    "Interview_Marks"
])

# =========================================================
# PREDICT ELIGIBILITY
# =========================================================

prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

eligible_probability = probability[0][1] * 100

not_eligible_probability = (
    probability[0][0] * 100
)

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:

    print("Prediction : ELIGIBLE FOR ADMISSION")

else:

    print("Prediction : NOT ELIGIBLE")

print(
    f"Eligibility Probability : "
    f"{eligible_probability:.2f}%"
)

print(
    f"Non-Eligibility Probability : "
    f"{not_eligible_probability:.2f}%"
)

# =========================================================
# ADMISSION ANALYTICS REPORT
# =========================================================

print("\n========== ADMISSION ANALYTICS ==========\n")

total_students = len(df)

eligible_students = df["Eligible"].sum()

not_eligible_students = (
    total_students - eligible_students
)

average_marks = df[
    "Academic_Marks"
].mean()

eligibility_percentage = (
    eligible_students / total_students
) * 100

print(f"Total Students        : {total_students}")

print(f"Eligible Students     : {eligible_students}")

print(
    f"Not Eligible Students : "
    f"{not_eligible_students}"
)

print(f"Average Marks         : {average_marks:.2f}")

print(
    f"Eligibility Percentage : "
    f"{eligibility_percentage:.2f}%"
)

# =========================================================
# MERIT RANKING DASHBOARD
# =========================================================

df["Merit_Score"] = (
    df["Academic_Marks"] * 0.4 +
    df["Entrance_Score"] * 0.4 +
    df["Interview_Marks"] * 0.2
)

rank_dashboard = df.sort_values(
    by="Merit_Score",
    ascending=False
)

print("\n========== TOP MERIT STUDENTS ==========\n")

print(rank_dashboard.head())

# =========================================================
# GRAPH 1
# MARKS VS ELIGIBILITY
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Academic_Marks"],
    df["Eligible"]
)

plt.xlabel("Academic Marks")

plt.ylabel("Eligibility")

plt.title("Academic Marks vs Eligibility")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# ENTRANCE SCORE VS ELIGIBILITY
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Entrance_Score"],
    df["Eligible"]
)

plt.xlabel("Entrance Score")

plt.ylabel("Eligibility")

plt.title("Entrance Score vs Eligibility")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# RESERVATION ANALYSIS
# =========================================================

reservation_avg = df.groupby(
    "Reservation_Category"
)["Eligible"].mean()

print("\n========== RESERVATION ANALYSIS ==========\n")

print(reservation_avg)

plt.figure(figsize=(6,5))

reservation_avg.plot(kind="bar")

plt.xlabel("Reservation Category")

plt.ylabel("Eligibility Rate")

plt.title("Reservation vs Eligibility")

plt.grid(True)

plt.show()

# =========================================================
# SAVE ANALYTICS REPORT
# =========================================================

analytics = pd.DataFrame({

    "Metric": [
        "Total Students",
        "Eligible Students",
        "Not Eligible Students",
        "Average Marks",
        "Eligibility Percentage"
    ],

    "Value": [
        total_students,
        eligible_students,
        not_eligible_students,
        average_marks,
        eligibility_percentage
    ]
})

analytics.to_csv(
    "admission_analytics_report.csv",
    index=False
)

print("\nAnalytics report saved successfully")

# =========================================================
# EXPORT PDF REPORT
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="University Admission Report",
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

    result = "ELIGIBLE FOR ADMISSION"

else:

    result = "NOT ELIGIBLE"

pdf.cell(
    200,
    10,
    txt=f"Prediction: {result}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Eligibility Probability: {eligible_probability:.2f}%",
    ln=True
)

pdf.output("University_Admission_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: University_Admission_Report.pdf")

# =========================================================
# LOGISTIC REGRESSION FORMULA
# =========================================================

print("\n========== LOGISTIC REGRESSION FORMULA ==========\n")

print("P(y=1) = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ... + bnxn))")
# ===========================================================
# END OF PROJECT
# ===========================================================

# The model learns:
# - Patterns of eligible students
# - Patterns of non-eligible students
# - How different features contribute to eligibility

# Input Example:
# Enter Academic Marks: 92
# Enter Entrance Exam Score: 95
# Enter Reservation Category: 0
# Enter Interview Marks: 18 