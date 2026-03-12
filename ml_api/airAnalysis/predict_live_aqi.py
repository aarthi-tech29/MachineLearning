import requests
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# 1. Load trained model and scaler
# -------------------------
model = joblib.load("aqi_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# 2. Fetch live AQI data from AQICN
# -------------------------
API_TOKEN = os.getenv("API_TOKEN")  # Set your AQICN API token in environment variable
city = input("Enter city name: ").strip()

url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
response = requests.get(url).json()

if response["status"] != "ok":
    print("City not found or API error.")
    exit()

data = response["data"]
iaqi = data.get("iaqi", {})

# -------------------------
# 3. Prepare features for prediction (only index_value)
# -------------------------
features = pd.DataFrame([[data.get("aqi", 0)]], columns=['index_value'])
features_scaled = scaler.transform(features)

# -------------------------
# 4. Predict ML health risk
# -------------------------
risk_prediction = model.predict(features_scaled)[0]

# -------------------------
# 5. Determine AQI category
# -------------------------
aqi = data.get("aqi", 0)
if aqi <= 50:
    aqi_risk = "Low"
elif aqi <= 100:
    aqi_risk = "Moderate"
elif aqi <= 150:
    aqi_risk = "High"
elif aqi <= 200:
    aqi_risk = "Very High"
else:
    aqi_risk = "Hazardous"

def health_advice(aqi_val):
    if aqi_val <= 50:
        return "Safe to go outside."
    elif aqi_val <= 100:
        return "Moderate air quality. Sensitive people should limit outdoor activity."
    elif aqi_val <= 150:
        return "Unhealthy for sensitive groups. Limit outdoor activity."
    elif aqi_val <= 200:
        return "Very unhealthy. Avoid prolonged outdoor exposure."
    else:
        return "Hazardous. Stay indoors."

# -------------------------
# 6. Print formatted results
# -------------------------
pollutants = {
    "PM2.5": iaqi.get("pm25", {}).get("v", 0),
    "PM10": iaqi.get("pm10", {}).get("v", 0),
    "NO2": iaqi.get("no2", {}).get("v", 0),
    "SO2": iaqi.get("so2", {}).get("v", 0),
    "CO": iaqi.get("co", {}).get("v", 0),
    "O3": iaqi.get("o3", {}).get("v", 0),
}

print("\n========== Live AQI Report ==========")
print(f"City: {city}")
print(f"AQI: {aqi} ({aqi_risk})")
print(f"ML Predicted Health Risk: {risk_prediction}")
print("Health Advice:", health_advice(aqi))
print("\nPollutant Concentrations (µg/m³):")
for k, v in pollutants.items():
    print(f"  {k}: {v}")
print("=====================================")