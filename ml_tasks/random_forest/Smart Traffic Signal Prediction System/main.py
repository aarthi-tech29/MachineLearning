# =========================================================
# SMART TRAFFIC SIGNAL PREDICTION SYSTEM
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

df = pd.read_csv("traffic_signal_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

weather_encoder = LabelEncoder()
peak_encoder = LabelEncoder()

df["Weather_Condition"] = weather_encoder.fit_transform(
    df["Weather_Condition"]
)

df["Peak_Hour"] = peak_encoder.fit_transform(
    df["Peak_Hour"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Vehicle_Count",
    "Weather_Condition",
    "Peak_Hour",
    "Average_Speed"
]]

y = df["Congestion_Level"]

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

# The tree can grow only up to 5 levels deep. This helps prevent overfitting, where the model 
# learns too much from the training data and performs poorly on new data.
# | max_depth  | Result           |
# | ---------- | ---------------- |
# | Very small | Underfitting     |
# | Balanced   | Good performance |
# | Very large | Overfitting      |

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

print("\nENTER TRAFFIC DETAILS")
print("------------------------------------------------")

vehicle_count = int(input("Enter Vehicle Count: "))
weather = input("Enter Weather Condition (Clear/Cloudy/Rainy): ")
peak_hour = input("Peak Hour? (Yes/No): ")
average_speed = float(input("Enter Average Speed: "))

# =========================================================
# ENCODE INPUT
# =========================================================

weather_encoded = weather_encoder.transform([weather])[0]
peak_encoded = peak_encoder.transform([peak_hour])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

traffic_data = pd.DataFrame({
    "Vehicle_Count": [vehicle_count],
    "Weather_Condition": [weather_encoded],
    "Peak_Hour": [peak_encoded],
    "Average_Speed": [average_speed]
})

# =========================================================
# CONGESTION PREDICTION
# =========================================================

prediction = model.predict(traffic_data)

print("\nTRAFFIC CONGESTION PREDICTION")
print("------------------------------------------------")
print("Predicted Congestion Level:", prediction[0])

# =========================================================
# SMART SIGNAL TIMING RECOMMENDATION
# =========================================================

print("\nSMART SIGNAL TIMING RECOMMENDATION")
print("------------------------------------------------")

if prediction[0] == "High":
    print("Increase green signal timing.")
    print("Activate traffic diversion routes.")
elif prediction[0] == "Medium":
    print("Maintain balanced signal timing.")
else:
    print("Normal traffic flow.")
    print("Standard signal timing is sufficient.")

# =========================================================
# FEATURE IMPORTANCE VISUALIZATION
# =========================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance in Traffic Prediction")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()
# ========================================================

# The model learns:
# - The Random Forest model can predict traffic congestion levels based on vehicle count, weather conditions, peak hour status, and average speed.
# - The feature importance visualization shows which factors have the most influence on congestion predictions, helping traffic
# authorities focus on key areas for traffic management and signal timing adjustments.
# - The classification report provides insights into the model's performance across different congestion levels (e.g., "Low", "Medium", "High").
# - The user input allows for real-time congestion prediction, enabling proactive traffic signal adjustments to improve
# traffic flow and reduce congestion in smart cities.
# - The smart signal timing recommendation provides actionable insights based on the predicted congestion level, helping to optimize traffic signal timings and improve overall traffic management.

# The Input example:
# Congestion Level: High
# Enter Vehicle Count: 500
# Enter Weather Condition (Clear/Cloudy/Rainy): Rainy
# Peak Hour? (Yes/No): Yes
# Enter Average Speed: 15

# Congestion Level: Low
# Enter Vehicle Count: 140
# Enter Weather Condition (Clear/Cloudy/Rainy): Clear
# Peak Hour? (Yes/No): No
# Enter Average Speed: 52