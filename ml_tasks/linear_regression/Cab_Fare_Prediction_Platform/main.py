# =========================================================
# CAB FARE PREDICTION PLATFORM
# LINEAR REGRESSION PROJECT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder # Convert text to numbers

from sklearn.metrics import (
    mean_absolute_error, # Average error
    mean_squared_error, # Prediction error magnitude
    r2_score # Model performance
)

from fpdf import FPDF

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("cab_fare_dataset.csv")

print("\n========== DATASET ==========\n")
print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# =========================================================
# LABEL ENCODING
# CONVERT TEXT TO NUMBERS
# =========================================================

traffic_encoder = LabelEncoder()
weather_encoder = LabelEncoder()
peak_encoder = LabelEncoder()

df["Traffic_Level"] = traffic_encoder.fit_transform(
    df["Traffic_Level"]
)
# Converts traffic text column into numeric values.
df["Weather"] = weather_encoder.fit_transform(
    df["Weather"]
)

df["Peak_Hour"] = peak_encoder.fit_transform(
    df["Peak_Hour"]
)

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Distance_km",
    "Traffic_Level",
    "Weather",
    "Peak_Hour",
    "Trip_Time_Minutes",
    "Base_Fare"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Total_Fare"]

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
# CREATE MODEL
# =========================================================

model = LinearRegression()

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")

# =========================================================
# TEST PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

print("\n========== TEST PREDICTIONS ==========\n")

for i in range(len(y_pred)):

    print(f"Actual Fare      : ₹{y_test.iloc[i]:.2f}")
    print(f"Predicted Fare   : ₹{y_pred[i]:.2f}")
    print("--------------------------------------")

# =========================================================
# ACCURACY METRICS
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== ACCURACY METRICS ==========\n")

print(f"MAE Score    : {mae:.2f}")
print(f"RMSE Score   : {rmse:.2f}")
print(f"R2 Score     : {r2:.2f}")

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n========== CAB FARE PREDICTION ==========\n")

distance = float(input("Enter Distance (km): "))

print("\nTraffic Level")
print("0 = Low")
print("1 = Medium")
print("2 = High")

traffic = int(input("Enter Traffic Level: "))

print("\nWeather")
print("0 = Clear")
print("1 = Fog")
print("2 = Rainy")

weather = int(input("Enter Weather: "))

print("\nPeak Hour")
print("0 = No")
print("1 = Yes")

peak = int(input("Enter Peak Hour: "))

trip_time = float(input("Enter Trip Time (minutes): "))

base_fare = float(input("Enter Base Fare: "))

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    distance,
    traffic,
    weather,
    peak,
    trip_time,
    base_fare
]], columns=[
    "Distance_km",
    "Traffic_Level",
    "Weather",
    "Peak_Hour",
    "Trip_Time_Minutes",
    "Base_Fare"
])

# =========================================================
# PREDICT FARE
# =========================================================

prediction = model.predict(new_data)

print("\n========== PREDICTION RESULT ==========\n")

print(f"Predicted Cab Fare: Rs.{prediction[0]:.2f}")

# =========================================================
# GRAPH 1
# DISTANCE VS FARE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Distance_km"],
    df["Total_Fare"]
)

plt.xlabel("Distance (km)")
plt.ylabel("Fare Amount")
plt.title("Distance vs Cab Fare")

plt.grid(True)

plt.show()
# Shows relationship between:distance,fare
# =========================================================
# GRAPH 2
# TRIP TIME VS FARE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Trip_Time_Minutes"],
    df["Total_Fare"]
)

plt.xlabel("Trip Time (minutes)")
plt.ylabel("Fare Amount")
plt.title("Trip Time vs Cab Fare")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# TRAFFIC LEVEL VS FARE
# =========================================================

traffic_avg = df.groupby(
    "Traffic_Level"
)["Total_Fare"].mean()

plt.figure(figsize=(8,5))

traffic_avg.plot(kind="bar")

plt.xlabel("Traffic Level")
plt.ylabel("Average Fare")
plt.title("Traffic Level vs Average Fare")

plt.grid(True)

plt.show()
# Calculates average fare for:
# low traffic
# medium traffic
# high traffic

# =========================================================
# WEATHER IMPACT ANALYSIS
# =========================================================

weather_avg = df.groupby(
    "Weather"
)["Total_Fare"].mean()

print("\n========== WEATHER IMPACT ==========\n")

print(weather_avg)
# rainy weather → higher fare
# clear weather → lower fare

# =========================================================
# PEAK HOUR ANALYSIS
# =========================================================

peak_avg = df.groupby(
    "Peak_Hour"
)["Total_Fare"].mean()

print("\n========== PEAK HOUR ANALYSIS ==========\n")

print(peak_avg)
# Compares:
# peak hour fare
# non-peak fare

# =========================================================
# FARE COMPARISON DASHBOARD
# =========================================================

print("\n========== FARE DASHBOARD ==========\n")

print(f"Minimum Fare : Rs.{df['Total_Fare'].min()}")

print(f"Maximum Fare : Rs.{df['Total_Fare'].max()}")

print(f"Average Fare : Rs.{df['Total_Fare'].mean():.2f}")

# =========================================================
# SAVE DASHBOARD REPORT
# =========================================================

dashboard = pd.DataFrame({
    "Metric": [
        "Minimum Fare",
        "Maximum Fare",
        "Average Fare"
    ],

    "Value": [
        df['Total_Fare'].min(),
        df['Total_Fare'].max(),
        df['Total_Fare'].mean()
    ]
})

dashboard.to_csv(
    "fare_dashboard_report.csv",
    index=False
)

print("\nDashboard report saved successfully")

# =========================================================
# EXPORT PDF REPORT
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="Cab Fare Prediction Report",
    ln=True,
    align='C'
)

pdf.ln(10)

pdf.set_font("Arial", size=12)

pdf.cell(
    200,
    10,
    txt=f"MAE Score: {mae:.2f}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"RMSE Score: {rmse:.2f}",
    ln=True
)

pdf.cell(
    200,
    10,
    txt=f"R2 Score: {r2:.2f}",
    ln=True
)

pdf.ln(10)

pdf.cell(
    200,
    10,
    txt=f"Predicted Fare: Rs.{prediction[0]:.2f}",
    ln=True
)

pdf.output("Cab_Fare_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Cab_Fare_Report.pdf")

# =========================================================
# LINEAR REGRESSION FORMULA
# =========================================================

print("\n========== LINEAR REGRESSION FORMULA ==========\n")

print("y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn")
# y= Total Fare
# b0=Intercept
# x1=Distance_km,b1=Coefficient for Distance
# x2=Traffic_Level,b2=Coefficient for Traffic Level
# x3=Weather,b3=Coefficient for Weather
# x4=Peak_Hour,b4=Coefficient for Peak Hour
# x5=Trip_Time_Minutes,b5=Coefficient for Trip Time
# x6=Base_Fare,b6=Coefficient for Base Fare
# These coefficients are automatically learned by the model during training.
# =========================================================
# END OF PROJECT
# =========================================================

# The model learns:
# how much fare increases with distance
# how traffic affects fare
# how rainy weather changes fare
# how peak hour increases price
# and combines everything to predict final fare.

# Input Example: Rainy
# Enter Distance (km): 15
# Enter Traffic Level: 2
# Traffic Level
# 0 = Low
# 1 = Medium
# 2 = High
# Enter Weather: 2
# Weather
# 0 = Clear
# 1 = Fog
# 2 = Rainy
# Enter Peak Hour: 1
# Peak Hour
# 0 = No
# 1 = Yes
# Enter Trip Time (minutes): 40
# Enter Base Fare: 50

# Low Traffic Ride
# Enter Distance (km): 8
# Enter Traffic Level: 0
# Enter Weather: 0
# Enter Peak Hour: 0
# Enter Trip Time (minutes): 15
# Enter Base Fare: 50

# Long Peak-Hour Ride
# Enter Distance (km): 25
# Enter Traffic Level: 2
# Enter Weather: 1
# Enter Peak Hour: 1
# Enter Trip Time (minutes): 60
# Enter Base Fare: 50
# =================================================================
# =========================================================
# REAL-TIME PREDICTION API
# =========================================================

from flask import Flask, request, jsonify



# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)

# Creates Flask web application.
# __name__
# tells Flask:
# where the current file is running
# Flask uses it internally.

# =========================================================
# HOME ROUTE
# =========================================================

@app.route('/') # / means home page 
# This creates a URL route.
def home():

    return "Cab Fare Prediction API Running Successfully"

# =========================================================
# PREDICTION API
# =========================================================

@app.route('/predict', methods=['POST'])

def predict_api():
# Whenever someone sends request to:/predict this function executes.


    try:
        # Try running code safely.If any error happens:app will not crash except block handles error
        # =================================================
        # GET JSON DATA
        # =================================================

        data = request.json

        # Reads JSON data sent from:Postman

        # =================================================
        # READ INPUT VALUES
        # =================================================

        distance = data['distance']

        traffic = data['traffic']

        weather = data['weather']

        peak_hour = data['peak_hour']

        trip_time = data['trip_time']

        base_fare = data['base_fare']
        # The entire JSON data from Postman is stored inside:data
        # All the data sent from Postman is stored in these variables.
        # =================================================
        # CREATE DATAFRAME
        # =================================================

        new_api_data = pd.DataFrame([[ # Converts input into table format.
            distance,
            traffic,
            weather,
            peak_hour,
            trip_time,
            base_fare
        ]], columns=[
            "Distance_km",
            "Traffic_Level",
            "Weather",
            "Peak_Hour",
            "Trip_Time_Minutes",
            "Base_Fare"
        ])

        # =================================================
        # PREDICT FARE
        # =================================================

        api_prediction = model.predict(new_api_data) # expects dataframe with column names.

        predicted_fare = round(
            float(api_prediction[0]),
            2
        )

        # =================================================
        # RETURN JSON RESPONSE
        # =================================================

        return jsonify({ # Sends result back as JSON.

            "Predicted_Fare": predicted_fare,

            "Status": "Success"

        })

    except Exception as e:

        return jsonify({

            "Error": str(e),

            "Status": "Failed"

        })
    # If any error happens:
    # catch error 
    # return error message
    # instead of crashing app.
# =========================================================
# RUN FLASK SERVER
# =========================================================

if __name__ == '__main__':
# Runs server only when current file is executed directly.
    app.run(debug=True)
    # Starts Flask web server.