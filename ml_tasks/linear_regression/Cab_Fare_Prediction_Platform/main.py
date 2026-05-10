# =========================================================
# CAB FARE PREDICTION SYSTEM 
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from flask import Flask, request, jsonify
from fpdf import FPDF

# =========================================================
# GEOLOCATION
# =========================================================

geolocator = Nominatim(user_agent="cab_fare_app")

def get_coordinates(place):
    try:
        location = geolocator.geocode(place)
        if location is None:
            return None, None
        return location.latitude, location.longitude
    except:
        return None, None

# =========================================================
# DISTANCE CALCULATION (HAVERSINE)
# =========================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("cab_fare_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(df.head())

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df[[
    "Distance_km",
    "Traffic_Level",
    "Weather",
    "Peak_Hour",
    "Trip_Time_Minutes",
    "Base_Fare"
]]

y = df["Total_Fare"]

# =========================================================
# TRAIN MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# =========================================================
# PERFORMANCE
# =========================================================

y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))

# =========================================================
# 📊 DASHBOARD (MATPLOTLIB)
# =========================================================

# Fare distribution
plt.figure()
plt.hist(df["Total_Fare"], bins=10)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# Distance vs Fare
plt.figure()
plt.scatter(df["Distance_km"], df["Total_Fare"])
plt.title("Distance vs Fare")
plt.xlabel("Distance (km)")
plt.ylabel("Fare")
plt.show()

# Traffic vs Fare
plt.figure()
plt.scatter(df["Traffic_Level"], df["Total_Fare"])
plt.title("Traffic vs Fare")
plt.xlabel("Traffic Level")
plt.ylabel("Fare")
plt.show()

# =========================================================
# USER INPUT
# =========================================================

pickup = input("Enter Pickup Location: ")
drop = input("Enter Drop Location: ")

lat1, lon1 = get_coordinates(pickup)
lat2, lon2 = get_coordinates(drop)

if lat1 is None or lat2 is None:
    print("Invalid location entered")
    exit()

distance = calculate_distance(lat1, lon1, lat2, lon2)

traffic = int(input("Traffic (0-Low,1-Med,2-High): "))
weather = int(input("Weather (0-Clear,1-Fog,2-Rain): "))
peak = int(input("Peak Hour (0/1): "))
trip_time = float(input("Trip Time (min): "))
base_fare = float(input("Base Fare: "))

# =========================================================
# PREDICTION
# =========================================================

input_data = pd.DataFrame([[ 
    distance,
    traffic,
    weather,
    peak,
    trip_time,
    base_fare
]], columns=X.columns)

prediction = model.predict(input_data)[0]
prediction = max(0, prediction)

print("\n===== FINAL FARE =====")
print("Predicted Fare: Rs.", round(prediction, 2))

# =========================================================
# DASHBOARD SUMMARY
# =========================================================

print("\n===== FARE SUMMARY =====")
print("Min Fare:", df["Total_Fare"].min())
print("Max Fare:", df["Total_Fare"].max())
print("Avg Fare:", df["Total_Fare"].mean())

# =========================================================
# PDF REPORT
# =========================================================

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, "Cab Fare Prediction Report", ln=True)
pdf.ln(5)

pdf.cell(200, 10, f"Pickup: {pickup}", ln=True)
pdf.cell(200, 10, f"Drop: {drop}", ln=True)
pdf.cell(200, 10, f"Distance: {round(distance,2)} km", ln=True)
pdf.cell(200, 10, f"Fare: Rs.{round(prediction,2)}", ln=True)

pdf.output("Cab_Fare_Report.pdf")

print("\nPDF Generated Successfully")

# =========================================================
# FLASK API
# =========================================================

app = Flask(__name__)

@app.route('/')
def home():
    return "Cab Fare Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    lat1, lon1 = get_coordinates(data['pickup'])
    lat2, lon2 = get_coordinates(data['drop'])

    if lat1 is None or lat2 is None:
        return jsonify({"error": "Invalid location"})

    dist = calculate_distance(lat1, lon1, lat2, lon2)

    input_df = pd.DataFrame([[ 
        dist,
        data['traffic'],
        data['weather'],
        data['peak_hour'],
        data['trip_time'],
        data['base_fare']
    ]], columns=X.columns)

    result = model.predict(input_df)[0]
    result = max(0, result)

    return jsonify({
        "distance_km": round(dist, 2),
        "predicted_fare": round(result, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

# =========================================================

# The model learns:
# - How to convert location names into coordinates using geopy.
# - How to calculate distance between two locations using Haversine formula.
# - How to train a linear regression model to predict cab fares.
# - How to evaluate model performance using MAE, RMSE, and R² metrics.
# - How to take user input for pickup and drop locations, traffic, weather, etc.
# - How to generate a PDF report with the prediction results.
# - How to create a Flask API for real-time fare prediction based on user input.
# - How to handle invalid location inputs gracefully without crashing the app.

# Input Example:
# Example 1 (Normal ride)
# {
#   "pickup": "Bangalore",
#   "drop": "Chennai",
#   "traffic": 2,
#   "weather": 2,
#   "peak_hour": 1,
#   "trip_time": 150,
#   "base_fare": 100
# }
# Example 2 (Short ride)
# {
#   "pickup": "Mumbai",
#   "drop": "Pune",
#   "traffic": 1,
#   "weather": 0,
#   "peak_hour": 0,
#   "trip_time": 60,
#   "base_fare": 50
# }
# Example 3 (Low traffic city ride)
# {
#   "pickup": "Chennai",
#   "drop": "Bangalore",
#   "traffic": 0,
#   "weather": 0,
#   "peak_hour": 0,
#   "trip_time": 180,
#   "base_fare": 80
# }
# Example 4 (Rain + peak hour)
# {
#   "pickup": "Delhi",
#   "drop": "Noida",
#   "traffic": 2,
#   "weather": 2,
#   "peak_hour": 1,
#   "trip_time": 45,
#   "base_fare": 40
# }
# ===========================================================