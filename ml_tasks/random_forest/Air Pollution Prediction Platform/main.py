# =========================================================
# AIR POLLUTION PREDICTION PLATFORM
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

df = pd.read_csv("air_pollution_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

weather_encoder = LabelEncoder()

df["Weather_Condition"] = weather_encoder.fit_transform(
    df["Weather_Condition"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "AQI",
    "Vehicle_Density",
    "Weather_Condition",
    "Temperature"
]]

y = df["Pollution_Level"]

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

print("\nENTER AIR POLLUTION DETAILS")
print("------------------------------------------------")

aqi = int(input("Enter AQI Value: "))
vehicle_density = int(input("Enter Vehicle Density: "))
weather = input("Enter Weather Condition (Clear/Cloudy/Rainy/Smog): ")
temperature = float(input("Enter Temperature: "))

# =========================================================
# ENCODE INPUT
# =========================================================

weather_encoded = weather_encoder.transform([weather])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

pollution_data = pd.DataFrame({
    "AQI": [aqi],
    "Vehicle_Density": [vehicle_density],
    "Weather_Condition": [weather_encoded],
    "Temperature": [temperature]
})

# =========================================================
# POLLUTION PREDICTION
# =========================================================

prediction = model.predict(pollution_data)

print("\nPOLLUTION LEVEL PREDICTION")
print("------------------------------------------------")
print("Predicted Pollution Level:", prediction[0])

# =========================================================
# CITY-WISE ANALYTICS DASHBOARD
# =========================================================

print("\nCITY-WISE ANALYTICS DASHBOARD")
print("------------------------------------------------")

if prediction[0] == "High":
    print("Air pollution is dangerously high.")
    print("Reduce outdoor activities.")
elif prediction[0] == "Moderate":
    print("Moderate pollution detected.")
    print("Sensitive people should take precautions.")
else:
    print("Air quality is good.")
    print("Pollution level is under control.")

# =========================================================
# FEATURE IMPORTANCE VISUALIZATION
# =========================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance in Air Pollution Prediction")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()
# ========================================================

# The model learns:
# - The model can predict the pollution level based on AQI, vehicle density, weather conditions, and temperature.
# - The feature importance visualization shows which factors have the most influence on pollution level predictions, helping city planners focus on key areas for improvement.
# - The user input section allows individuals to get real-time predictions and recommendations based on current air pollution conditions, promoting awareness and health safety.
# - The classification report provides insights into the model's performance across different pollution levels (e.g., "Low", "Moderate", "High"), helping to evaluate and improve the model further.
# - The city-wise analytics dashboard provides actionable insights based on the predicted pollution level, guiding residents on how to respond to varying air quality conditions.
# - The random forest model's ability to handle complex interactions between features makes it a powerful tool for predicting air pollution levels and providing valuable insights for both individuals and city planners.

# The Input example:
# Pollution Level: High
# Enter AQI Value: 320
# Enter Vehicle Density: 850
# Enter Weather Condition (Clear/Cloudy/Rainy/Smog): Smog
# Enter Temperature: 40

# Pollution Level: Low
# Enter AQI Value: 70
# Enter Vehicle Density: 180
# Enter Weather Condition (Clear/Cloudy/Rainy/Smog): Rainy
# Enter Temperature: 28