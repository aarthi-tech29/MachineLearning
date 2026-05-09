# =========================================================
# ELECTRICITY DEMAND FORECASTING SYSTEM
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
# LOAD DATASET
# =========================================================

df = pd.read_csv("electricity_demand_dataset.csv")

print("\n========== DATASET ==========\n")

print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================================================
# LABEL ENCODING
# CONVERT FESTIVAL SEASON TEXT TO NUMBERS
# =========================================================

festival_encoder = LabelEncoder()

df["Festival_Season"] = festival_encoder.fit_transform(
    df["Festival_Season"]
)

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Historical_Consumption",
    "Temperature",
    "Festival_Season",
    "Hour",
    "Department_Consumption"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Electricity_Demand"]

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

    print(f"Actual Demand      : {y_test.iloc[i]:.2f} Units")

    print(f"Predicted Demand   : {y_pred[i]:.2f} Units")

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

print("\n========== ELECTRICITY DEMAND PREDICTION ==========\n")

historical_consumption = float(
    input("Enter Historical Consumption: ")
)

temperature = float(
    input("Enter Temperature: ")
)

print("\nFestival Season")
print("0 = No")
print("1 = Yes")

festival = int(
    input("Enter Festival Season: ")
)

hour = int(
    input("Enter Hour (0-23): ")
)

department_consumption = float(
    input("Enter Department Consumption: ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    historical_consumption,
    temperature,
    festival,
    hour,
    department_consumption
]], columns=[
    "Historical_Consumption",
    "Temperature",
    "Festival_Season",
    "Hour",
    "Department_Consumption"
])

# =========================================================
# PREDICT ELECTRICITY DEMAND
# =========================================================

prediction = model.predict(new_data)

predicted_demand = prediction[0]

print("\n========== PREDICTION RESULT ==========\n")

print(
    f"Predicted Electricity Demand: "
    f"{predicted_demand:.2f} Units"
)

# =========================================================
# HOURLY FORECAST SYSTEM
# =========================================================

print("\n========== HOURLY FORECAST ==========\n")

hourly_forecast = []

for h in range(24):

    hourly_value = predicted_demand + np.random.randint(-20, 20)

    hourly_forecast.append(hourly_value)

    print(f"Hour {h}: {hourly_value:.2f} Units")

# =========================================================
# GRAPH 1
# TEMPERATURE VS DEMAND
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Temperature"],
    df["Electricity_Demand"]
)

plt.xlabel("Temperature")

plt.ylabel("Electricity Demand")

plt.title("Temperature vs Electricity Demand")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# HOURLY DEMAND FORECAST
# =========================================================

plt.figure(figsize=(10,5))

plt.plot(
    range(24),
    hourly_forecast,
    marker='o'
)

plt.xlabel("Hour")

plt.ylabel("Electricity Demand")

plt.title("Hourly Electricity Demand Forecast")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# FESTIVAL ANALYSIS
# =========================================================

festival_avg = df.groupby(
    "Festival_Season"
)["Electricity_Demand"].mean()

print("\n========== FESTIVAL ANALYSIS ==========\n")

print(festival_avg)

plt.figure(figsize=(8,5))

festival_avg.plot(kind="bar")

plt.xlabel("Festival Season")

plt.ylabel("Average Demand")

plt.title("Festival Season vs Electricity Demand")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 4
# DEPARTMENT CONSUMPTION CHART
# =========================================================

plt.figure(figsize=(8,5))

plt.bar(
    df.index,
    df["Department_Consumption"]
)

plt.xlabel("Department Index")

plt.ylabel("Department Consumption")

plt.title("Department-wise Consumption Chart")

plt.grid(True)

plt.show()

# =========================================================
# ENERGY USAGE DASHBOARD
# =========================================================

print("\n========== ENERGY USAGE DASHBOARD ==========\n")

print(
    f"Minimum Demand : "
    f"{df['Electricity_Demand'].min()} Units"
)

print(
    f"Maximum Demand : "
    f"{df['Electricity_Demand'].max()} Units"
)

print(
    f"Average Demand : "
    f"{df['Electricity_Demand'].mean():.2f} Units"
)

# =========================================================
# SAVE DEPARTMENT REPORT
# =========================================================

department_report = pd.DataFrame({

    "Department_Index": df.index,

    "Department_Consumption": df[
        "Department_Consumption"
    ],

    "Electricity_Demand": df[
        "Electricity_Demand"
    ]
})

department_report.to_csv(
    "department_consumption_report.csv",
    index=False
)

print("\nDepartment report saved successfully")

# =========================================================
# EXPORT PDF REPORT
# =========================================================

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=16)

pdf.cell(
    200,
    10,
    txt="Electricity Demand Forecast Report",
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
    txt=f"Predicted Demand: {predicted_demand:.2f} Units",
    ln=True
)

pdf.output("Electricity_Demand_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Electricity_Demand_Report.pdf")

# =========================================================
# LINEAR REGRESSION FORMULA
# =========================================================

print("\n========== LINEAR REGRESSION FORMULA ==========\n")

print("y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn")
# =========================================================
# END OF PROJECT
# =========================================================

# Demand=b0+b1(Historical Consumption)+b2(Temperature)+b3(Festival Season)+b4(Hourly Usage)

# The model learns:
# Which factors increase electricity demand
# Which factors reduce electricity demand
# How strongly each factor affects electricity demand
# The model can be used by energy providers to:
# 1. Forecast demand for better resource allocation
# 2. Plan for peak hours and festival seasons
# 3. Optimize energy distribution and reduce wastage
# 4. Provide insights for energy conservation strategies

# Input Examples:
# Enter Historical Consumption: 600
# Enter Temperature: 39
# Enter Festival Season: 1
# Festival Season
# 0 = No
# 1 = Yes
# Enter Hour (0-23): 19
# Enter Department Consumption: 280