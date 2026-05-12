# =========================================================
# LOAN DEFAULT RISK ANALYTICS
# USING RANDOM FOREST
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("loan_default_risk_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

emi_encoder = LabelEncoder()

df["EMI_Payment_History"] = emi_encoder.fit_transform(
    df["EMI_Payment_History"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Credit_Score",
    "Monthly_Salary",
    "EMI_Payment_History",
    "Monthly_Transactions"
]]

y = df["Loan_Default_Risk"]

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
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print("------------------------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")
print("------------------------------------------------")
print(classification_report(y_test, y_pred))

# =========================================================
# USER INPUT
# =========================================================

print("\nENTER CUSTOMER DETAILS")
print("------------------------------------------------")

credit_score = int(input("Enter Credit Score: "))
salary = float(input("Enter Monthly Salary: "))
emi_history = input("Enter EMI Payment History (Good/Average/Poor): ")
transactions = int(input("Enter Monthly Transactions Count: "))

# =========================================================
# ENCODE INPUT
# =========================================================

emi_encoded = emi_encoder.transform([emi_history])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

customer_data = pd.DataFrame({
    "Credit_Score": [credit_score],
    "Monthly_Salary": [salary],
    "EMI_Payment_History": [emi_encoded],
    "Monthly_Transactions": [transactions]
})

# =========================================================
# DEFAULT RISK PREDICTION
# =========================================================

prediction = model.predict(customer_data)

print("\nLOAN DEFAULT RISK PREDICTION")
print("------------------------------------------------")
print("Predicted Risk Level:", prediction[0])

# =========================================================
# DEFAULT RISK SCORE
# =========================================================

risk_probabilities = model.predict_proba(customer_data)

print("\nDEFAULT RISK SCORE")
print("------------------------------------------------")

for risk, probability in zip(model.classes_, risk_probabilities[0]):
    print(f"{risk} Risk Probability: {round(probability * 100, 2)}%")

# =========================================================
# RISK VISUALIZATION DASHBOARD
# =========================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance in Loan Default Risk")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()
# =========================================================

# The model learns:
# - The random forest model can predict the risk of loan default based on customer financial data and payment history.
# - The feature importance visualization helps identify which factors contribute most to the default risk prediction.
# - The classification report provides insights into the model's performance across different risk categories.
# - The user input section allows for real-time risk assessment based on new customer data, making it a practical tool for financial institutions.
# - The risk score output provides a more detailed understanding of the likelihood of default, which can aid in decision-making for loan approvals and risk management.

# The Input example:
# Risk Level: High
# Enter Credit Score: 510
# Enter Monthly Salary: 22000
# Enter EMI Payment History (Good/Average/Poor): Poor
# Enter Monthly Transactions Count: 30

# Risk Level: Low
# Enter Credit Score: 790
# Enter Monthly Salary: 90000
# Enter EMI Payment History (Good/Average/Poor): Good
# Enter Monthly Transactions Count: 130