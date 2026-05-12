# =========================================================
# CYBER ATTACK RISK CLASSIFICATION
# USING DECISION TREE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("cyber_attack_risk_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

ip_encoder = LabelEncoder()

df["Suspicious_IP"] = ip_encoder.fit_transform(
    df["Suspicious_IP"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Failed_Login_Attempts",
    "Suspicious_IP",
    "Network_Traffic_MB",
    "Access_Time_Hours"
]]

y = df["Threat_Level"]

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

print("\nENTER NETWORK DETAILS")
print("------------------------------------------------")

failed_logins = int(input("Enter Failed Login Attempts: "))
suspicious_ip = input("Suspicious IP? (Yes/No): ")
network_traffic = float(input("Enter Network Traffic (MB): "))
access_time = float(input("Enter Access Time (Hours): "))

# =========================================================
# ENCODE INPUT
# =========================================================

ip_encoded = ip_encoder.transform([suspicious_ip])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

network_data = pd.DataFrame({
    "Failed_Login_Attempts": [failed_logins],
    "Suspicious_IP": [ip_encoded],
    "Network_Traffic_MB": [network_traffic],
    "Access_Time_Hours": [access_time]
})

# =========================================================
# THREAT PREDICTION
# =========================================================

prediction = model.predict(network_data)

print("\nCYBER THREAT PREDICTION")
print("------------------------------------------------")
print("Threat Level:", prediction[0])

# =========================================================
# SECURITY ALERT SYSTEM
# =========================================================

print("\nSECURITY ALERT SYSTEM")
print("------------------------------------------------")

if prediction[0] == "High":
    print("High Risk Alert!")
    print("Immediate security action required.")
elif prediction[0] == "Medium":
    print("Medium Risk Detected.")
    print("Monitor network activity carefully.")
else:
    print("Low Risk.")
    print("System operating normally.")

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 10))

plot_tree(
    model,
    feature_names=[
        "Failed Login Attempts",
        "Suspicious IP",
        "Network Traffic",
        "Access Time"
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Cyber Attack Risk Classification Decision Tree")

plt.show()
# =========================================================

# The model learns:
# - The decision tree model classifies cyber attack risk based on key features such as failed login attempts, suspicious IP activity, network traffic, and access time.
# - The model achieves a certain level of accuracy in predicting threat levels, which can be evaluated using the classification report.
# - The user input allows for real-time threat prediction, enabling proactive security measures.
# - The decision tree visualization helps understand the decision-making process and the key factors influencing cyber attack risk classification.
# - The security alert system provides actionable insights based on the predicted threat level, helping organizations respond effectively to potential cyber threats.

# The Input example:
# Threat Level: High
# Enter Failed Login Attempts: 9
# Suspicious IP? (Yes/No): Yes
# Enter Network Traffic (MB): 1000
# Enter Access Time (Hours): 1

# Threat Level: Low
# Enter Failed Login Attempts: 2
# Suspicious IP? (Yes/No): No
# Enter Network Traffic (MB): 180
# Enter Access Time (Hours): 4