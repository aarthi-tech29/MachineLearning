# =========================================================
# INDUSTRIAL MACHINE FAILURE DETECTION
# USING DECISION TREE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("machine_failure_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Temperature",
    "Vibration",
    "Pressure",
    "Humidity"
]]

y = df["Machine_Status"]

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

print("\nENTER MACHINE SENSOR DETAILS")
print("------------------------------------------------")

temperature = float(input("Enter Temperature: "))
vibration = float(input("Enter Vibration Level: "))
pressure = float(input("Enter Pressure: "))
humidity = float(input("Enter Humidity: "))

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

machine_data = pd.DataFrame({
    "Temperature": [temperature],
    "Vibration": [vibration],
    "Pressure": [pressure],
    "Humidity": [humidity]
})

# =========================================================
# FAILURE PREDICTION
# =========================================================

prediction = model.predict(machine_data)

print("\nMACHINE STATUS PREDICTION")
print("------------------------------------------------")
print("Prediction:", prediction[0])

# =========================================================
# MAINTENANCE ALERT SYSTEM
# =========================================================

if prediction[0] == "Failure":
    print("\nMAINTENANCE ALERT")
    print("------------------------------------------------")
    print("Warning: Machine breakdown risk detected!")
    print("Immediate maintenance required.")
else:
    print("\nMAINTENANCE STATUS")
    print("------------------------------------------------")
    print("Machine operating normally.")

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 10))

plot_tree(
    model,
    feature_names=[
        "Temperature",
        "Vibration",
        "Pressure",
        "Humidity"
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Industrial Machine Failure Detection Decision Tree")

plt.show()
# ========================================================

# The model learns:
# - The decision tree model can predict machine failure based on sensor data with a certain level of accuracy.
# - The classification report provides insights into the model's performance across different classes (e.g.,
#  "Normal" vs "Failure").
# - The user input section allows for real-time predictions based on new sensor readings, making it practical for industrial applications.
# - The maintenance alert system provides actionable insights based on the model's predictions, helping to prevent machine breakdowns and reduce downtime.
# - The decision tree visualization helps understand the decision-making process and the key factors influencing machine failure predictions.

# The Input example:
# Failure:
# Enter Temperature: 88
# Enter Vibration Level: 7.8
# Enter Pressure: 58
# Enter Humidity: 72

# Normal:
# Enter Temperature: 48
# Enter Vibration Level: 2.5
# Enter Pressure: 33
# Enter Humidity: 42
