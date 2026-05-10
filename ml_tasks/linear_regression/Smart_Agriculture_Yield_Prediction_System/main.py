# =========================================================
# SMART AGRICULTURE YIELD PREDICTION SYSTEM
# LINEAR REGRESSION PROJECT
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fpdf import FPDF

# =========================================================
# 1. FARMER REGISTRATION MODULE
# =========================================================

FARMER_FILE = "farmers.csv"

if not os.path.exists(FARMER_FILE):
    pd.DataFrame(columns=["Farmer_ID", "Name", "Location"]).to_csv(FARMER_FILE, index=False)

print("\n========== FARMER REGISTRATION ==========\n")

farmer_id = input("Enter Farmer ID: ")
name = input("Enter Farmer Name: ")
location = input("Enter Location: ")

farmers = pd.read_csv(FARMER_FILE)

new_farmer = pd.DataFrame([[farmer_id, name, location]],
                          columns=["Farmer_ID", "Name", "Location"])

farmers = pd.concat([farmers, new_farmer], ignore_index=True)
farmers.to_csv(FARMER_FILE, index=False)

print("✔ Farmer Registered Successfully!")

# =========================================================
# 2. ADMIN DATASET UPLOAD MODULE
# =========================================================

print("\n========== ADMIN DATASET UPLOAD ==========\n")

file_path = input("Enter dataset CSV file name: ")

df = pd.read_csv(file_path)

print("\n✔ Dataset Loaded Successfully!\n")
print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[[
    "Rainfall_mm",
    "Soil_Quality",
    "Temperature_C",
    "Fertilizer_kg",
    "Humidity_%"
]]

y = df["Crop_Yield_Tons"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# LINEAR REGRESSION MODEL
# =========================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✔ Model Training Completed!")

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY METRICS
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# =========================================================
# USER INPUT PREDICTION
# =========================================================

print("\n========== NEW CROP PREDICTION ==========\n")

rainfall = float(input("Rainfall (mm): "))
soil = float(input("Soil Quality (1-10): "))
temp = float(input("Temperature (C): "))
fertilizer = float(input("Fertilizer (kg): "))
humidity = float(input("Humidity (%): "))

new_data = pd.DataFrame([[
    rainfall, soil, temp, fertilizer, humidity
]], columns=[
    "Rainfall_mm",
    "Soil_Quality",
    "Temperature_C",
    "Fertilizer_kg",
    "Humidity_%"
])

prediction = model.predict(new_data)

print(f"\nPredicted Crop Yield: {prediction[0]:.2f} tons")

# =========================================================
# GRAPH VISUALIZATION
# =========================================================

plt.figure()
plt.scatter(df["Rainfall_mm"], df["Crop_Yield_Tons"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Crop Yield")
plt.grid()
plt.show()

plt.figure()
plt.scatter(df["Temperature_C"], df["Crop_Yield_Tons"])
plt.xlabel("Temperature")
plt.ylabel("Yield")
plt.title("Temperature vs Crop Yield")
plt.grid()
plt.show()

plt.figure()
plt.scatter(df["Fertilizer_kg"], df["Crop_Yield_Tons"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Crop Yield")
plt.grid()
plt.show()

# =========================================================
# MONTHLY PREDICTION REPORT (ML BASED)
# =========================================================

print("\n========== MONTHLY PREDICTION REPORT ==========\n")

months = np.arange(1, 13)

monthly_data = pd.DataFrame({
    "Rainfall_mm": df["Rainfall_mm"].mean() + np.sin(months/12 * 2*np.pi) * 40,
    "Soil_Quality": df["Soil_Quality"].mean(),
    "Temperature_C": df["Temperature_C"].mean() + np.cos(months/12 * 2*np.pi) * 5,
    "Fertilizer_kg": df["Fertilizer_kg"].mean(),
    "Humidity_%": df["Humidity_%"].mean()
})

monthly_prediction = model.predict(monthly_data)

report = pd.DataFrame({
    "Month": [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ],
    "Predicted_Yield_Tons": monthly_prediction
})

print(report)

report.to_csv("monthly_prediction_report.csv", index=False)

print("\n✔ Monthly Report Saved")

# =========================================================
# PDF EXPORT
# =========================================================

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)

pdf.cell(200, 10, "Smart Agriculture Yield Report", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", size=14)

pdf.cell(200, 10, f"MAE: {mae:.2f}", ln=True)
pdf.cell(200, 10, f"RMSE: {rmse:.2f}", ln=True)
pdf.cell(200, 10, f"R² Score: {r2:.2f}", ln=True)
pdf.cell(200, 10, f"Predicted Yield: {prediction[0]:.2f}", ln=True)

pdf.output("Crop_Yield_Report.pdf")

print("\nPDF Report Generated Successfully")

# =========================================================
# LINEAR REGRESSION EQUATION
# =========================================================

print("\n========== LINEAR REGRESSION FORMULA ==========\n")

print("y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn")

# y = crop yield
# b0= Base Yield
# x1= rainfall, b1= Rainfall Importance
# x2= soil quality, b2= Soil Quality Importance
# x3= temperature, b3= Temperature Importance
# x4= fertilizer, b4= Fertilizer Importance
# x5= humidity, b5= Humidity Importance
# =========================================================
# END OF PROJECT
# =========================================================
# The model learns:
# - How each factor (rainfall, soil quality, temperature, fertilizer, humidity) impacts crop yield.
# - How to make predictions based on new input data.
# - How to evaluate model performance using MAE, RMSE, and R² metrics.
# - How to visualize relationships between features and target variable.
# - How to generate reports and export them in CSV and PDF formats.
# - How to build a simple farmer registration system to manage user data.
# - How to create a monthly prediction report based on seasonal patterns in the data.
# - How to interpret the linear regression formula and understand the importance of each feature in predicting crop yield.


# Input Examples:

# Farmer Registration
# Enter Farmer ID: F101
# Enter Farmer Name: Arjun
# Enter Location: Tamil Nadu
# Dataset File Name
# crop_yield_dataset.csv
# Prediction Inputs (GOOD YIELD EXAMPLE)
# Rainfall (mm): 850
# Soil Quality (1-10): 8
# Temperature (C): 28
# Fertilizer (kg): 140
# Humidity (%): 72

# Low Yield Case
# Rainfall: 500
# Soil Quality: 5
# Temperature: 35
# Fertilizer: 80
# Humidity: 55

# High Yield Case
# Rainfall: 1000
# Soil Quality: 10
# Temperature: 25
# Fertilizer: 170
# Humidity: 85
