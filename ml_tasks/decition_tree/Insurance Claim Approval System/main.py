# =========================================================
# INSURANCE CLAIM APPROVAL SYSTEM USING DECISION TREE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# | Library                | Purpose                     |
# | ---------------------- | --------------------------- |
# | pandas                 | Read and handle dataset     |
# | matplotlib             | Display visualization       |
# | train_test_split       | Split training/testing data |
# | DecisionTreeClassifier | Create decision tree model  |
# | plot_tree              | Visualize decision tree     |
# | accuracy_score         | Calculate model accuracy    |
# | classification_report  | Show performance report     |


# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("insurance_claim_dataset.csv")

print("\nDATASET")
print("--------------------------------")
print(df.head())

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Claim_Amount",
    "Accident_History",
    "Policy_Duration_Years",
    "Fraud_History"
]]

y = df["Claim_Status"]

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
# DECISION TREE MODEL
# =========================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
# Gini is a metric used in Decision Trees to measure:How pure or impure the data is.
# It helps the model decide:
# Which feature should split the data best.
# | Gini Value | Meaning        |
# | ---------- | -------------- |
# | 0          | Perfectly pure |
# | Close to 1 | Mixed / impure |

# | Parameter        | Meaning               |
# | ---------------- | --------------------- |
# | criterion="gini" | Splitting method      |
# | max_depth=4      | Maximum tree depth    |
# | random_state=42  | Same output every run |

# The tree can grow only up to 4 levels deep. This helps prevent overfitting, where the model 
# learns too much from the training data and performs poorly on new data.
# | max_depth  | Result           |
# | ---------- | ---------------- |
# | Very small | Underfitting     |
# | Balanced   | Good performance |
# | Very large | Overfitting      |


model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred) # Displays accuracy percentage.

print("\nMODEL ACCURACY")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")
print("--------------------------------")
print(classification_report(y_test, y_pred)) # Used to evaluate model performance.

# =========================================================
# SAMPLE PREDICTION
# =========================================================

print("\nENTER CLAIM DETAILS")
print("--------------------------------")

claim_amount = float(input("Enter Claim Amount: "))
accident_history = int(input("Enter Accident History Count: "))
policy_duration = int(input("Enter Policy Duration (Years): "))
fraud_history = int(input("Fraud History? (0 = No, 1 = Yes): "))

sample_claim = pd.DataFrame({
    "Claim_Amount": [claim_amount],
    "Accident_History": [accident_history],
    "Policy_Duration_Years": [policy_duration],
    "Fraud_History": [fraud_history]
})

prediction = model.predict(sample_claim)

print("\nCLAIM APPROVAL RESULT")
print("--------------------------------")
print("Prediction:", prediction[0])

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 10))

plot_tree(
    model,
    feature_names=[
        "Claim Amount",
        "Accident History",
        "Policy Duration",
        "Fraud History"
    ],
    class_names=["Approved", "Rejected"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Insurance Claim Approval Decision Tree")

plt.show()
# ========================================================

# The model learns:
# - Higher claim amounts are more likely to be denied.
# - More accidents in history increase denial chances.
# - Longer policy duration can increase approval chances.
# - A history of fraud significantly increases denial chances.
# The decision tree captures these patterns to make informed predictions on claim approvals.
# This system can help insurance companies automate claim processing and reduce fraudulent claims.

# Input Example:
# Approved
# Enter Claim Amount: 25000
# Enter Accident History Count: 2
# Enter Policy Duration (Years): 3
# Fraud History? (0 = No, 1 = Yes): 0

# Enter Claim Amount: 12000
# Enter Accident History Count: 1
# Enter Policy Duration (Years): 5
# Fraud History? (0 = No, 1 = Yes): 0

# Rejected
# Enter Claim Amount: 45000
# Enter Accident History Count: 4
# Enter Policy Duration (Years): 1
# Fraud History? (0 = No, 1 = Yes): 1