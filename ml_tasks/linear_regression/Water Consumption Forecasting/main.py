# =========================================================
# WATER CONSUMPTION FORECASTING SYSTEM
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
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from fpdf import FPDF

# =========================================================
# BUILDING MANAGEMENT LOGIN
# =========================================================

print("\n========== BUILDING MANAGEMENT LOGIN ==========\n")

username = input("Enter Username: ")

password = input("Enter Password: ")

if username == "admin" and password == "1234":

    print("\nLogin Successful")

else:

    print("\nInvalid Username or Password")

    exit()

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("water_usage_dataset.csv")

print("\n========== DATASET ==========\n")

print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================================================
# LABEL ENCODING
# CONVERT SEASON TEXT TO NUMBERS
# =========================================================

season_encoder = LabelEncoder()

df["Season"] = season_encoder.fit_transform(
    df["Season"]
)

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Month",
    "Occupancy_Count",
    "Temperature",
    "Season",
    "Previous_Usage"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Water_Usage_Liters"]

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

    print(f"Actual Usage      : {y_test.iloc[i]:.2f} Liters")

    print(f"Predicted Usage   : {y_pred[i]:.2f} Liters")

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

print("\n========== WATER USAGE PREDICTION ==========\n")

month = int(input("Enter Month Number (1-12): "))

occupancy = int(input("Enter Occupancy Count: "))

temperature = float(input("Enter Temperature: "))

print("\nSeason")
print("0 = Rainy")
print("1 = Summer")
print("2 = Winter")

season = int(input("Enter Season: "))

previous_usage = float(
    input("Enter Previous Water Usage (Liters): ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    month,
    occupancy,
    temperature,
    season,
    previous_usage
]], columns=[
    "Month",
    "Occupancy_Count",
    "Temperature",
    "Season",
    "Previous_Usage"
])

# =========================================================
# PREDICT WATER USAGE
# =========================================================

prediction = model.predict(new_data)

predicted_usage = prediction[0]

print("\n========== PREDICTION RESULT ==========\n")

print(
    f"Predicted Water Usage: "
    f"{predicted_usage:.2f} Liters"
)

# =========================================================
# DAILY FORECAST
# =========================================================

daily_forecast = predicted_usage / 30

print("\n========== DAILY FORECAST ==========\n")

print(f"Daily Usage Forecast: {daily_forecast:.2f} Liters")

# =========================================================
# WEEKLY FORECAST
# =========================================================

weekly_forecast = daily_forecast * 7

print("\n========== WEEKLY FORECAST ==========\n")

print(f"Weekly Usage Forecast: {weekly_forecast:.2f} Liters")

# =========================================================
# MONTHLY FORECAST
# =========================================================

print("\n========== MONTHLY FORECAST ==========\n")

print(f"Monthly Usage Forecast: {predicted_usage:.2f} Liters")

# =========================================================
# ALERT SYSTEM
# =========================================================

limit = 5500

print("\n========== ALERT SYSTEM ==========\n")

if predicted_usage > limit:

    print("WARNING: Predicted usage exceeds limit")

else:

    print("Water usage is under control")

# =========================================================
# GRAPH 1
# OCCUPANCY VS WATER USAGE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Occupancy_Count"],
    df["Water_Usage_Liters"]
)

plt.xlabel("Occupancy Count")

plt.ylabel("Water Usage")

plt.title("Occupancy vs Water Usage")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# TEMPERATURE VS WATER USAGE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Temperature"],
    df["Water_Usage_Liters"]
)

plt.xlabel("Temperature")

plt.ylabel("Water Usage")

plt.title("Temperature vs Water Usage")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# SEASONAL ANALYSIS
# =========================================================

season_avg = df.groupby(
    "Season"
)["Water_Usage_Liters"].mean()

plt.figure(figsize=(8,5))

season_avg.plot(kind="bar")

plt.xlabel("Season")

plt.ylabel("Average Water Usage")

plt.title("Seasonal Water Usage Analysis")

plt.grid(True)

plt.show()

# =========================================================
# SEASONAL ANALYSIS OUTPUT
# =========================================================

print("\n========== SEASONAL ANALYSIS ==========\n")

print(season_avg)

# =========================================================
# FORECAST GRAPH
# =========================================================

forecast_values = [
    daily_forecast,
    weekly_forecast,
    predicted_usage
]

forecast_labels = [
    "Daily",
    "Weekly",
    "Monthly"
]

plt.figure(figsize=(8,5))

plt.plot(
    forecast_labels,
    forecast_values,
    marker='o'
)

plt.xlabel("Forecast Type")

plt.ylabel("Water Usage")

plt.title("Water Consumption Forecast")

plt.grid(True)

plt.show()

# =========================================================
# SAVE FORECAST REPORT
# =========================================================

forecast_report = pd.DataFrame({

    "Forecast_Type": [
        "Daily",
        "Weekly",
        "Monthly"
    ],

    "Predicted_Usage": [
        daily_forecast,
        weekly_forecast,
        predicted_usage
    ]
})

forecast_report.to_csv(
    "water_forecast_report.csv",
    index=False
)

print("\nForecast report saved successfully")

# =========================================================
# EXPORT PDF REPORT
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="Water Consumption Forecast Report",
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
    txt=f"Predicted Usage: {predicted_usage:.2f} Liters",
    ln=True
)

pdf.output("Water_Consumption_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Water_Consumption_Report.pdf")

# =========================================================
# END OF PROJECT
# =========================================================
# Water Usage=b0​+b1​(Occupancy)+b2​(Temperature)+b3​(Season)+b4​(Previous Usage)
# The model learns:
# Which factors increase water usage
# Which factors reduce water usage
# How strongly each factor affects water consumption

# Input Examples:
# Login Input
# Enter Username: admin
# Enter Password: 1234
# Enter Month Number (1-12): 5
# Enter Occupancy Count: 25
# Enter Temperature: 36
# Enter Season: 1
# Season
# 0 = Rainy
# 1 = Summer
# 2 = Winter
# Enter Previous Water Usage (Liters): 5600