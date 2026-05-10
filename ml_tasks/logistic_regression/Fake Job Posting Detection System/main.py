# =========================================================
# FAKE JOB POSTING DETECTION SYSTEM
# LOGISTIC REGRESSION PROJECT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

df = pd.read_csv("fake_job_dataset.csv")

print("\n========== DATASET ==========\n")

print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================================================
# NLP TEXT PREPROCESSING
# =========================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    return text

# Apply cleaning
df["Cleaned_Description"] = df[
    "Job_Description"
].apply(clean_text)

# =========================================================
# TF-IDF VECTORIZATION
# CONVERT TEXT TO NUMBERS
# =========================================================

vectorizer = TfidfVectorizer() 

text_features = vectorizer.fit_transform(
    df["Cleaned_Description"]
)

# =========================================================
# EXTRA NUMERIC FEATURES
# =========================================================

extra_features = df[[
    "Company_Valid",
    "Salary",
    "Remote"
]].values

# =========================================================
# COMBINE TEXT + NUMERIC FEATURES
# =========================================================

from scipy.sparse import hstack

X = hstack([text_features, extra_features])

# Combines:
# text features
# numeric features
# into one dataset.

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Fraudulent"]

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
# Model learns:
# fake patterns
# real job patterns
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

precision = precision_score(y_test, y_pred) # How accurate fake predictions are.

recall = recall_score(y_test, y_pred) # How many actual fake jobs were correctly identified.

f1 = f1_score(y_test, y_pred) # Balanced performance score.

print("\n========== ACCURACY METRICS ==========\n")

print(f"Accuracy Score   : {accuracy:.2f}")

print(f"Precision Score  : {precision:.2f}")

print(f"Recall Score     : {recall:.2f}")

print(f"F1 Score         : {f1:.2f}")

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n========== JOB POST CHECK ==========\n")

job_description = input(
    "Enter Job Description: "
)

print("\nCompany Validation")
print("0 = Invalid Company")
print("1 = Valid Company")

company_valid = int(
    input("Enter Company Validation: ")
)

salary = float(
    input("Enter Salary: ")
)

print("\nRemote Job")
print("0 = No")
print("1 = Yes")

remote = int(
    input("Enter Remote Option: ")
)

# =========================================================
# CLEAN USER INPUT
# =========================================================

cleaned_input = clean_text(
    job_description
)

# =========================================================
# TF-IDF TRANSFORM USER INPUT
# =========================================================

input_text = vectorizer.transform(
    [cleaned_input]
)

# =========================================================
# EXTRA INPUT FEATURES
# =========================================================

extra_input = np.array([[
    company_valid,
    salary,
    remote
]])

# =========================================================
# COMBINE INPUT FEATURES
# =========================================================

new_data = hstack([
    input_text,
    extra_input
])

# =========================================================
# PREDICT FAKE / REAL
# =========================================================

prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

fake_probability = probability[0][1] * 100

real_probability = probability[0][0] * 100

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:

    print("Prediction : FAKE JOB")

else:

    print("Prediction : REAL JOB")

print(
    f"Fake Probability : "
    f"{fake_probability:.2f}%"
)

print(
    f"Real Probability : "
    f"{real_probability:.2f}%"
)

# =========================================================
# GRAPH 1
# FAKE VS REAL COUNT
# =========================================================

fraud_counts = df["Fraudulent"].value_counts()

plt.figure(figsize=(6,5))

fraud_counts.plot(kind="bar")

plt.xlabel("Class")

plt.ylabel("Count")

plt.title("Fake vs Real Job Count")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# SALARY ANALYSIS
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Salary"],
    df["Fraudulent"]
)

plt.xlabel("Salary")

plt.ylabel("Fraudulent")

plt.title("Salary vs Fake Job")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# COMPANY VALIDATION ANALYSIS
# =========================================================

company_avg = df.groupby(
    "Company_Valid"
)["Fraudulent"].mean()

print("\n========== COMPANY VALIDATION ==========\n")

print(company_avg)

plt.figure(figsize=(6,5))

company_avg.plot(kind="bar")

plt.xlabel("Company Validation")

plt.ylabel("Fraud Percentage")

plt.title("Company Validation vs Fraud")

plt.grid(True)

plt.show()

# =========================================================
# ADMIN DASHBOARD
# =========================================================

print("\n========== ADMIN DASHBOARD ==========\n")

total_jobs = len(df)

fake_jobs = df["Fraudulent"].sum()

real_jobs = total_jobs - fake_jobs

avg_salary = df["Salary"].mean()

fraud_percentage = (
    fake_jobs / total_jobs
) * 100

print(f"Total Jobs         : {total_jobs}")

print(f"Fake Jobs          : {fake_jobs}")

print(f"Real Jobs          : {real_jobs}")

print(f"Average Salary     : {avg_salary:.2f}")

print(
    f"Fraud Percentage   : "
    f"{fraud_percentage:.2f}%"
)

# =========================================================
# SAVE DASHBOARD REPORT
# =========================================================

dashboard = pd.DataFrame({

    "Metric": [
        "Total Jobs",
        "Fake Jobs",
        "Real Jobs",
        "Average Salary",
        "Fraud Percentage"
    ],

    "Value": [
        total_jobs,
        fake_jobs,
        real_jobs,
        avg_salary,
        fraud_percentage
    ]
})

dashboard.to_csv(
    "fake_job_dashboard.csv",
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
    txt="Fake Job Detection Report",
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

    result = "FAKE JOB"

else:

    result = "REAL JOB"

pdf.cell(
    200,
    10,
    txt=f"Prediction: {result}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Fake Probability: {fake_probability:.2f}%",
    ln=True
)

pdf.output("Fake_Job_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Fake_Job_Report.pdf")

# =========================================================
# LOGISTIC REGRESSION FORMULA
# =========================================================

print("\n========== LOGISTIC REGRESSION FORMULA ==========\n")

print("P(y=1) = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ... + bnxn))")
# =========================================================
# END OF PROJECT
# ========================================================
# # P(y=1) is the probability that a job posting is FAKE.
# b0 is the intercept (base bias of the model).
# b1, b2, ..., bn are coefficients for each feature (word patterns, salary, company info, etc.).
# x1, x2, ..., xn are input features extracted from job posting data.
# Example:
# P(fake job) = 1 / (1 + e^-(b0 + b1*SalaryPattern + b2*CompanyVerification
#                         + b3*UrgencyWords + b4*ExperienceMismatch + ...))
# This formula calculates the probability that a job advertisement is fake
# based on different job-related signals.
# Higher values of suspicious features (like unrealistic salary, urgent hiring,
# missing company details, spam-like text) increase the probability of FAKE job.
# Higher values of trusted features (like verified company, normal salary range,
# proper job description) decrease the probability of FAKE job.

# The model learns:
# Which words are common in fake job descriptions
# Which words are common in real job descriptions
# How numeric features like salary and company validation affect fraud probability
# How to combine text and numeric data for better predictions
# How to evaluate model performance using accuracy, precision, recall, and F1 score
# How to create visualizations to understand data patterns
# How to build a user input system for real-time predictions
# How to generate reports and dashboards for stakeholders


# Input Examples:
# Enter Job Description:
# Earn money quickly from home no experience needed
# Enter Company Validation: 0
# Company Validation
# 0 = Invalid Company
# 1 = Valid Company
# Enter Salary: 200000
# Enter Remote Option: 1
# Remote Job
# 0 = No
# 1 = Yes

# Real Job Input
# Enter Job Description:
# Python developer required with Django experience
# Enter Company Validation: 1
# Enter Salary: 65000
# Enter Remote Option: 0