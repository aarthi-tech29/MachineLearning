# =========================================================
# CROP DISEASE DETECTION SYSTEM
# USING RANDOM FOREST
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("crop_disease_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Soil_Moisture",
    "Soil_pH",
    "Temperature",
    "Humidity"
]]

y = df["Disease_Status"]

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

print("\nENTER CROP ENVIRONMENT DETAILS")
print("------------------------------------------------")

soil_moisture = float(input("Enter Soil Moisture: "))
soil_ph = float(input("Enter Soil pH: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

crop_data = pd.DataFrame({
    "Soil_Moisture": [soil_moisture],
    "Soil_pH": [soil_ph],
    "Temperature": [temperature],
    "Humidity": [humidity]
})

# =========================================================
# DISEASE PREDICTION
# =========================================================

prediction = model.predict(crop_data)

print("\nCROP DISEASE PREDICTION")
print("------------------------------------------------")
print("Prediction:", prediction[0])

# =========================================================
# FARMER RECOMMENDATION MODULE
# =========================================================

print("\nFARMER RECOMMENDATION")
print("------------------------------------------------")

if prediction[0] == "Disease":
    print("Disease detected in crop.")
    print("Recommended Actions:")
    print("- Apply suitable pesticide.")
    print("- Improve soil condition.")
    print("- Monitor humidity levels.")
else:
    print("Crop is healthy.")
    print("Maintain current farming conditions.")

# =========================================================
# FEATURE IMPORTANCE VISUALIZATION
# =========================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance in Crop Disease Detection")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()
# =========================================================
# The model learns:
# - The Random Forest model can predict crop disease status based on soil moisture, soil pH, temperature, and humidity.
# - The feature importance visualization helps identify which environmental factors have the most influence on disease prediction, guiding farmers in monitoring and managing crop health effectively.
# - The user input allows farmers to get real-time predictions and actionable recommendations to prevent or manage crop diseases, improving agricultural productivity and sustainability.
# - The classification report provides insights into the model's performance across different disease classes, helping to understand the strengths and weaknesses of the model in predicting crop health.
# - The accuracy score gives an overall measure of how well the model is performing in classifying healthy vs diseased crops, which is crucial for building trust in the system among farmers and agricultural stakeholders.
# - The farmer recommendation module provides practical advice based on the model's predictions, making it a valuable tool for decision-making in crop management and disease prevention.

# The Input example:
# Disease:
# Enter Soil Moisture: 18
# Enter Soil pH: 5.1
# Enter Temperature: 39
# Enter Humidity: 88

# Healthy
# Enter Soil Moisture: 45
# Enter Soil pH: 6.7
# Enter Temperature: 29
# Enter Humidity: 64