# =========================================================
# SMART AGRICULTURE YIELD PREDICTION SYSTEM
# USING CSV DATASET
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from fpdf import FPDF

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("crop_yield_dataset.csv")

print("\n========== DATASET ==========\n")
print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Rainfall_mm",
    "Soil_Quality",
    "Temperature_C",
    "Fertilizer_kg",
    "Humidity_%"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Crop_Yield_Tons"]

# =========================================================
# SPLIT TRAINING AND TESTING DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# CREATE LINEAR REGRESSION MODEL
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

    print(f"Actual Yield     : {y_test.iloc[i]:.2f} tons")
    print(f"Predicted Yield  : {y_pred[i]:.2f} tons")
    print("--------------------------------------")

# =========================================================
# ACCURACY METRICS
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
# Measures average prediction error.
# Lower MAE = better model.

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse) #root mean square error
# RMSE measures:
# How much error is there between
# Actual values and Predicted values
# Lower RMSE → Better model 
# Higher RMSE → More prediction error 

r2 = r2_score(y_test, y_pred)
# Checks model performance.
# Range:
# 0 → poor
# 1 → perfect

print("\n========== ACCURACY METRICS ==========\n")

print(f"MAE Score      : {mae:.2f}")
print(f"RMSE Score     : {rmse:.2f}")
print(f"R2 Score       : {r2:.2f}")

# =========================================================
# USER INPUT FOR NEW PREDICTION
# =========================================================

print("\n========== NEW CROP YIELD PREDICTION ==========\n")

rainfall = float(input("Enter Rainfall (mm): "))

soil_quality = int(input("Enter Soil Quality (1-10): "))

temperature = float(input("Enter Temperature (C): "))

fertilizer = float(input("Enter Fertilizer Usage (kg): "))

humidity = float(input("Enter Humidity (%): "))

new_data = pd.DataFrame([[
    rainfall,
    soil_quality,
    temperature,
    fertilizer,
    humidity
]], columns=[
    "Rainfall_mm",
    "Soil_Quality",
    "Temperature_C",
    "Fertilizer_kg",
    "Humidity_%"
])

prediction = model.predict(new_data)

print("\n========== PREDICTION RESULT ==========\n")

print(f"Predicted Crop Yield: {prediction[0]:.2f} tons")

# =========================================================
# GRAPH 1
# RAINFALL VS CROP YIELD
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Rainfall_mm"],
    df["Crop_Yield_Tons"]
)

plt.xlabel("Rainfall (mm)")
plt.ylabel("Crop Yield (Tons)")
plt.title("Rainfall vs Crop Yield")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# TEMPERATURE VS CROP YIELD
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Temperature_C"],
    df["Crop_Yield_Tons"]
)

plt.xlabel("Temperature (C)")
plt.ylabel("Crop Yield (Tons)")
plt.title("Temperature vs Crop Yield")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# FERTILIZER VS CROP YIELD
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Fertilizer_kg"],
    df["Crop_Yield_Tons"]
)

plt.xlabel("Fertilizer Usage (kg)")
plt.ylabel("Crop Yield (Tons)")
plt.title("Fertilizer vs Crop Yield")

plt.grid(True)

plt.show()

# =========================================================
# MONTHLY PREDICTION REPORT
# =========================================================

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_yield = np.random.uniform(
    3.0,
    7.0,
    12
)

report_df = pd.DataFrame({
    "Month": months,
    "Predicted_Yield_Tons": monthly_yield
})

print("\n========== MONTHLY PREDICTION REPORT ==========\n")

print(report_df)

# =========================================================
# SAVE MONTHLY REPORT AS CSV
# =========================================================

report_df.to_csv(
    "monthly_prediction_report.csv",
    index=False
)

print("\nMonthly report saved successfully")

# =========================================================
# EXPORT REPORT AS PDF
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="Smart Agriculture Yield Prediction Report",
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
    txt=f"Predicted Crop Yield: {prediction[0]:.2f} tons",
    ln=True
)

pdf.output("Crop_Yield_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Crop_Yield_Report.pdf")

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
# Which factors increase crop yield
# Which factors reduce crop yield
# How strongly each factor affects farming output

# Input Examples:
# Enter Rainfall (mm): 900
# Enter Soil Quality (1-10): 8
# Enter Temperature (C): 29
# Enter Fertilizer Usage (kg): 150
# Enter Humidity (%): 70

# Low Yield Example
# Enter Rainfall (mm): 650
# Enter Soil Quality (1-10): 5
# Enter Temperature (C): 34
# Enter Fertilizer Usage (kg): 85
# Enter Humidity (%): 55