
# ===================================================================================
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style
import os
from dotenv import load_dotenv
load_dotenv()

# ===============================
# AQICN API KEY
# ===============================
API_TOKEN = os.getenv("API_TOKEN")

# ===============================
# FETCH AIR QUALITY DATA
# ===============================
def fetch_air_quality(city):
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    response = requests.get(url).json()

    if response["status"] == "ok":
        data = response["data"]
        iaqi = data.get("iaqi", {})

        pollutants = {
            "PM2.5": iaqi.get("pm25", {}).get("v", 0),
            "PM10": iaqi.get("pm10", {}).get("v", 0),
            "NO2": iaqi.get("no2", {}).get("v", 0),
            "SO2": iaqi.get("so2", {}).get("v", 0),
            "CO": iaqi.get("co", {}).get("v", 0),
            "O3": iaqi.get("o3", {}).get("v", 0),
            "AQI": data.get("aqi", 0)
        }

        return pollutants
    else:
        return None

# ===============================
# SAMPLE HISTORICAL DATA FOR ML
# ===============================
sample_data = [
    [12, 20, 15, 5, 0.3, 10, 40, "Low"],
    [35, 60, 40, 10, 1.0, 30, 90, "Moderate"],
    [80, 150, 80, 20, 2.5, 60, 160, "High"],
    [55, 100, 60, 15, 1.5, 50, 120, "High"],
    [20, 40, 20, 8, 0.5, 15, 60, "Moderate"],
    [10, 15, 10, 4, 0.2, 8, 30, "Low"]
]

columns = ["PM2.5","PM10","NO2","SO2","CO","O3","AQI","risk_level"]
df = pd.DataFrame(sample_data, columns=columns)

# Features and target
X = df[["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]]
y = df["risk_level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ===============================
# HELPER FUNCTIONS
# ===============================
def color_risk(risk):
    if risk == "Low":
        return Fore.GREEN + risk + Style.RESET_ALL
    elif risk == "Moderate":
        return Fore.YELLOW + risk + Style.RESET_ALL
    elif risk == "High":
        return Fore.MAGENTA + risk + Style.RESET_ALL
    elif risk == "Very High":
        return Fore.RED + risk + Style.RESET_ALL
    else:
        return Fore.LIGHTMAGENTA_EX + risk + Style.RESET_ALL

def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. Safe to go outside."
    elif aqi <= 100:
        return "Air quality is moderate. Sensitive people should reduce prolonged outdoor activity."
    elif aqi <= 150:
        return "Air quality is unhealthy for sensitive groups. Limit outdoor activities."
    elif aqi <= 200:
        return "Air quality is very unhealthy. Avoid prolonged outdoor exposure."
    else:
        return "Air quality is hazardous. Stay indoors and keep windows closed."

def map_aqi_to_risk(aqi):
    if aqi <= 50:
        return "Low"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "High"
    elif aqi <= 200:
        return "Very High"
    else:
        return "Hazardous"

# ===============================
# MAIN PROGRAM
# ===============================
city = input("Enter city name: ")
data = fetch_air_quality(city)

if data:
    print(f"\nAir Quality Data for {city}")
    print("---------------------------")
    for key, value in data.items():
        print(f"{key}: {value}")

    # ML features as DataFrame
    features = pd.DataFrame([[
        data["PM2.5"], data["PM10"], data["NO2"], data["SO2"],
        data["CO"], data["O3"], data["AQI"]
    ]], columns=X.columns)

    features_scaled = scaler.transform(features)
    ml_risk = model.predict(features_scaled)[0]

    aqi = data["AQI"]
    aqi_risk = map_aqi_to_risk(aqi)
    advice = health_advice(aqi)

    print("\nML Predicted Health Risk Level:", color_risk(ml_risk))
    print("AQI-based Standard Risk Level:", color_risk(aqi_risk))
    print("Health Advice:", Fore.CYAN + advice + Style.RESET_ALL)

else:
    print("City not found or API error.")

# Color-coded output:
# Green → Low
# Yellow → Moderate
# Magenta → High
# Red → Very High
# Purple → Hazardous
# ===============================without env====================================================
import requests

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from colorama import Fore, Style
 


API_TOKEN = "8cbafed6720d9ef80efaf7a8e75464623eeb5149"
 
# ===============================

# FETCH AIR QUALITY DATA

# ===============================

def fetch_air_quality(city):

    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"

    response = requests.get(url).json()
 
    if response["status"] == "ok":

        data = response["data"]

        iaqi = data.get("iaqi", {})
 
        pollutants = {

            "PM2.5": iaqi.get("pm25", {}).get("v", 0),

            "PM10": iaqi.get("pm10", {}).get("v", 0),

            "NO2": iaqi.get("no2", {}).get("v", 0),

            "SO2": iaqi.get("so2", {}).get("v", 0),

            "CO": iaqi.get("co", {}).get("v", 0),

            "O3": iaqi.get("o3", {}).get("v", 0),

            "AQI": data.get("aqi", 0)

        }
 
        return pollutants

    else:

        return None
 
 


sample_data = [

    [12, 20, 15, 5, 0.3, 10, 40, "Low"],

    [35, 60, 40, 10, 1.0, 30, 90, "Moderate"],

    [80, 150, 80, 20, 2.5, 60, 160, "High"],

    [55, 100, 60, 15, 1.5, 50, 120, "High"],

    [20, 40, 20, 8, 0.5, 15, 60, "Moderate"],

    [10, 15, 10, 4, 0.2, 8, 30, "Low"]

]
 
columns = ["PM2.5","PM10","NO2","SO2","CO","O3","AQI","risk_level"]

df = pd.DataFrame(sample_data, columns=columns)
 


X = df[["PM2.5","PM10","NO2","SO2","CO","O3","AQI"]]

y = df["risk_level"]
 
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
 
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_scaled, y)
 
 


def color_risk(risk):

    if risk == "Low":

        return Fore.GREEN + risk + Style.RESET_ALL

    elif risk == "Moderate":

        return Fore.YELLOW + risk + Style.RESET_ALL

    elif risk == "High":

        return Fore.MAGENTA + risk + Style.RESET_ALL

    elif risk == "Very High":

        return Fore.RED + risk + Style.RESET_ALL

    else:

        return Fore.LIGHTMAGENTA_EX + risk + Style.RESET_ALL
 
 
def health_advice(aqi):

    if aqi <= 50:

        return "Air quality is good. Safe to go outside."

    elif aqi <= 100:

        return "Air quality is moderate. Sensitive people should reduce prolonged outdoor activity."

    elif aqi <= 150:

        return "Air quality is unhealthy for sensitive groups. Limit outdoor activities."

    elif aqi <= 200:

        return "Air quality is very unhealthy. Avoid prolonged outdoor exposure."

    else:

        return "Air quality is hazardous. Stay indoors and keep windows closed."
 
 
def map_aqi_to_risk(aqi):

    if aqi <= 50:

        return "Low"

    elif aqi <= 100:

        return "Moderate"

    elif aqi <= 150:

        return "High"

    elif aqi <= 200:

        return "Very High"

    else:

        return "Hazardous"
 
 


city = input("Enter city name: ")
 
data = fetch_air_quality(city)
 
if data:

    print(f"\nAir Quality Data for {city}")

    print("---------------------------")
 
    for key, value in data.items():

        print(f"{key}: {value}")
 
    features = pd.DataFrame([[

        data["PM2.5"], data["PM10"], data["NO2"], data["SO2"],

        data["CO"], data["O3"], data["AQI"]

    ]], columns=X.columns)
 
    features_scaled = scaler.transform(features)

    ml_risk = model.predict(features_scaled)[0]
 
    aqi = data["AQI"]

    aqi_risk = map_aqi_to_risk(aqi)

    advice = health_advice(aqi)
 
    print("\nML Predicted Health Risk Level:", color_risk(ml_risk))

    print("AQI-based Standard Risk Level:", color_risk(aqi_risk))

    print("Health Advice:", Fore.CYAN + advice + Style.RESET_ALL)
 
else:

    print("City not found or API error.")
 


