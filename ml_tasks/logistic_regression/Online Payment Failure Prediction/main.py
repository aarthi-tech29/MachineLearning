# =========================================================
# ONLINE PAYMENT FAILURE PREDICTION SYSTEM
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

df = pd.read_csv("online_payment_dataset.csv")

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

accuracy = accuracy_score(y_test, y_pred) # Correct predictions percentage (how many transactions were correctly classified as SUCCESS or FAILURE)

precision = precision_score(y_test, y_pred) # How many predicted FAILED payments were actually failed (accuracy of failure detection)

recall = recall_score(y_test, y_pred) # # How many actual FAILED transactions were correctly detected by the model (missed failure rate reduction)

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

internet_speed = float(
    input("Enter Internet Speed (Mbps): ")
)

amount = float(
    input("Enter Transaction Amount: ")
)

print("\nDevice Type")
print("0 = Mobile")
print("1 = Laptop")

device = int(
    input("Enter Device Type: ")
)

print("\nBrowser Type")
print("0 = Chrome")
print("1 = Firefox")
print("2 = Edge")

browser = int(
    input("Enter Browser Type: ")
)

response_time = float(
    input("Enter Gateway Response Time: ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
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
# PREDICT TRANSACTION STATUS
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

print(
    f"Failure Probability : "
    f"{failure_probability:.2f}%"
)

print(
    f"Success Probability : "
    f"{success_probability:.2f}%"
)

# =========================================================
# RISK ANALYTICS DASHBOARD
# =========================================================

print("\n========== RISK ANALYTICS DASHBOARD ==========\n")

total_transactions = len(df)

failed_transactions = df[
    "Transaction_Status"
].sum()

successful_transactions = (
    total_transactions - failed_transactions
)

failure_percentage = (
    failed_transactions / total_transactions
) * 100

avg_amount = df[
    "Transaction_Amount"
].mean()

print(f"Total Transactions      : {total_transactions}")

print(f"Failed Transactions     : {failed_transactions}")

print(f"Successful Transactions : {successful_transactions}")

print(f"Average Amount          : {avg_amount:.2f}")

print(
    f"Failure Percentage      : "
    f"{failure_percentage:.2f}%"
)

# =========================================================
# GRAPH 1
# INTERNET SPEED VS FAILURE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Internet_Speed"],
    df["Transaction_Status"]
)

plt.xlabel("Internet Speed")

plt.ylabel("Transaction Status")

plt.title("Internet Speed vs Payment Failure")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# TRANSACTION AMOUNT VS FAILURE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Transaction_Amount"],
    df["Transaction_Status"]
)

plt.xlabel("Transaction Amount")

plt.ylabel("Transaction Status")

plt.title("Amount vs Payment Failure")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# DEVICE ANALYSIS
# =========================================================

device_avg = df.groupby(
    "Device_Type"
)["Transaction_Status"].mean()

print("\n========== DEVICE ANALYSIS ==========\n")

print(device_avg)

plt.figure(figsize=(6,5))

device_avg.plot(kind="bar")

plt.xlabel("Device Type")

plt.ylabel("Failure Rate")

plt.title("Device vs Payment Failure")

plt.grid(True)

plt.show()

# =========================================================
# SAVE DASHBOARD REPORT
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

dashboard.to_csv(
    "payment_dashboard_report.csv",
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
    txt="Online Payment Failure Report",
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

    result = "PAYMENT FAILED"

else:

    result = "PAYMENT SUCCESSFUL"

pdf.cell(
    200,
    10,
    txt=f"Prediction: {result}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"Failure Probability: {failure_probability:.2f}%",
    ln=True
)

pdf.output("Payment_Failure_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Payment_Failure_Report.pdf")

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
# Enter Internet Speed (Mbps): 5
# Enter Transaction Amount: 10000
# Enter Device Type: 1
# Enter Browser Type: 2
# Enter Gateway Response Time: 10