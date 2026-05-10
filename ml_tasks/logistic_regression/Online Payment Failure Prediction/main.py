# =========================================================
# ONLINE PAYMENT FAILURE PREDICTION SYSTEM
# LOGISTIC REGRESSION PROJECT (FIXED VERSION)
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

df = pd.read_csv("online_payment_dataset.csv")

print("\n========== DATASET ==========\n")
print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# =========================================================
# LABEL ENCODING (FIX FOR DEVICE + BROWSER)
# =========================================================

device_encoder = LabelEncoder()
browser_encoder = LabelEncoder()

df["Device_Type"] = device_encoder.fit_transform(df["Device_Type"])
df["Browser_Type"] = browser_encoder.fit_transform(df["Browser_Type"])

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Internet_Speed",
    "Transaction_Amount",
    "Device_Type",
    "Browser_Type",
    "Gateway_Response_Time"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Transaction_Status"]

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

model = LogisticRegression(max_iter=1000)

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

print("\n========== PAYMENT FAILURE CHECK ==========\n")

internet_speed = float(input("Enter Internet Speed (Mbps): "))
amount = float(input("Enter Transaction Amount: "))

print("\nDevice Type")
print("0 = Laptop")
print("1 = Mobile")
device = int(input("Enter Device Type: "))

print("\nBrowser Type")
print("0 = Chrome")
print("1 = Edge")
print("2 = Firefox")
browser = int(input("Enter Browser Type: "))

response_time = float(input("Enter Gateway Response Time: "))

# =========================================================
# CREATE INPUT DATA
# =========================================================

new_data = pd.DataFrame([[
    internet_speed,
    amount,
    device,
    browser,
    response_time
]], columns=[
    "Internet_Speed",
    "Transaction_Amount",
    "Device_Type",
    "Browser_Type",
    "Gateway_Response_Time"
])

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

failure_probability = probability[0][1] * 100
success_probability = probability[0][0] * 100

print("\n========== PREDICTION RESULT ==========\n")

if prediction[0] == 1:
    print("Prediction : PAYMENT FAILED")
else:
    print("Prediction : PAYMENT SUCCESSFUL")

print(f"Failure Probability : {failure_probability:.2f}%")
print(f"Success Probability : {success_probability:.2f}%")

# =========================================================
# RISK ANALYTICS DASHBOARD
# =========================================================

print("\n========== RISK ANALYTICS DASHBOARD ==========\n")

total_transactions = len(df)
failed_transactions = df["Transaction_Status"].sum()
successful_transactions = total_transactions - failed_transactions
failure_percentage = (failed_transactions / total_transactions) * 100
avg_amount = df["Transaction_Amount"].mean()

print(f"Total Transactions      : {total_transactions}")
print(f"Failed Transactions     : {failed_transactions}")
print(f"Successful Transactions : {successful_transactions}")
print(f"Average Amount          : {avg_amount:.2f}")
print(f"Failure Percentage      : {failure_percentage:.2f}%")

# =========================================================
# GRAPH 1 - INTERNET SPEED VS FAILURE
# =========================================================

plt.figure(figsize=(8,5))
plt.scatter(df["Internet_Speed"], df["Transaction_Status"])
plt.xlabel("Internet Speed")
plt.ylabel("Transaction Status")
plt.title("Internet Speed vs Failure")
plt.grid(True)
plt.show()

# =========================================================
# GRAPH 2 - AMOUNT VS FAILURE
# =========================================================

plt.figure(figsize=(8,5))
plt.scatter(df["Transaction_Amount"], df["Transaction_Status"])
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Status")
plt.title("Amount vs Failure")
plt.grid(True)
plt.show()

# =========================================================
# GRAPH 3 - DEVICE ANALYSIS
# =========================================================

device_avg = df.groupby("Device_Type")["Transaction_Status"].mean()

plt.figure(figsize=(6,5))
device_avg.plot(kind="bar")
plt.xlabel("Device Type (Encoded)")
plt.ylabel("Failure Rate")
plt.title("Device vs Failure")
plt.grid(True)
plt.show()

# =========================================================
# SAVE DASHBOARD
# =========================================================

dashboard = pd.DataFrame({
    "Metric": [
        "Total Transactions",
        "Failed Transactions",
        "Successful Transactions",
        "Average Amount",
        "Failure Percentage"
    ],
    "Value": [
        total_transactions,
        failed_transactions,
        successful_transactions,
        avg_amount,
        failure_percentage
    ]
})

dashboard.to_csv("payment_dashboard_report.csv", index=False)

print("\nDashboard saved successfully")

# =========================================================
# PDF REPORT
# =========================================================

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=16)

pdf.cell(200, 10, txt="Payment Failure Prediction Report", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.2f}", ln=True)
pdf.cell(200, 10, txt=f"Precision: {precision:.2f}", ln=True)
pdf.cell(200, 10, txt=f"Recall: {recall:.2f}", ln=True)
pdf.cell(200, 10, txt=f"F1 Score: {f1:.2f}", ln=True)

pdf.ln(10)

result = "FAILED" if prediction[0] == 1 else "SUCCESS"

pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
pdf.cell(200, 10, txt=f"Failure Probability: {failure_probability:.2f}%", ln=True)

pdf.output("Payment_Failure_Report.pdf")

print("\nPDF Report Generated Successfully")

# =========================================================
# END OF PROJECT
# =========================================================

# =========================================================
# LOGISTIC REGRESSION FORMULA
# =========================================================

print("\n========== LOGISTIC REGRESSION FORMULA ==========\n")

print("P(y=1) = 1 / (1 + e^-(b0 + b1x1 + b2x2 + ... + bnxn))")
# =========================================================
# END OF PROJECT
# ========================================================
# P(y=1) is the probability that an online transaction will FAIL.
# b0 is the intercept (base failure tendency of the model).
# b1, b2, ..., bn are coefficients for each feature (internet speed, amount, device, etc.).
# x1, x2, ..., xn are the input transaction features.
# Example:
# P(failure) = 1 / (1 + e^-(b0 + b1*InternetSpeed + b2*TransactionAmount
#                         + b3*DeviceType + b4*BrowserType + b5*ResponseTime))
# This formula calculates the probability that a payment transaction will fail
# based on network and system conditions.
# Higher risk features (slow internet, high transaction amount,
# slow gateway response time, suspicious device/browser patterns)
# increase the probability of PAYMENT FAILURE.

# Normal conditions (fast internet, low/medium amount, quick response time,
# trusted device/browser) decrease the probability of FAILURE.

# The model learns:
# patterns of failed transactions
# patterns of successful transactions
# and uses these patterns to predict new transactions.
# The probabilities indicate how confident the model is about its prediction.
# A high failure probability means the model is very confident the transaction will fail, while a high success probability means it is confident the transaction will succeed.

# Input Example:
# Failure
# Enter Internet Speed (Mbps): 5
# Enter Transaction Amount: 10000
# Enter Device Type: 1
# Enter Browser Type: 2
# Enter Gateway Response Time: 10

# Success
# Enter Internet Speed (Mbps): 50
# Enter Transaction Amount: 2000
# Device Type
# 0 = Laptop
# 1 = Mobile
# Enter Device Type: 1
# Browser Type
# 0 = Chrome
# 1 = Edge
# 2 = Firefox
# Enter Browser Type: 0
# Enter Gateway Response Time: 1

# Example 2 (Balanced SAFE transaction)
# Internet Speed: 25
# Transaction Amount: 5000
# Device Type: 0
# Browser Type: 2
# Gateway Response Time: 2

# Example 3 (Medium safe transaction)
# Internet Speed: 30
# Transaction Amount: 8000
# Device Type: 1
# Browser Type: 0
# Gateway Response Time: 3