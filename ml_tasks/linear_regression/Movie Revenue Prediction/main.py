# =========================================================
# MOVIE REVENUE PREDICTION SYSTEM
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

df = pd.read_csv("movie_revenue_dataset.csv")

print("\n========== DATASET ==========\n")

print(df.head())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

# =========================================================
# LABEL ENCODING
# CONVERT GENRE TEXT TO NUMBERS
# =========================================================

genre_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(
    df["Genre"]
)

# =========================================================
# INPUT FEATURES
# =========================================================

X = df[[
    "Budget_Million",
    "Actor_Popularity",
    "Genre",
    "Trailer_Views_Million",
    "Social_Media_Engagement"
]]

# =========================================================
# TARGET VARIABLE
# =========================================================

y = df["Revenue_Million"]

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

    print(f"Actual Revenue      : {y_test.iloc[i]:.2f} Million")

    print(f"Predicted Revenue   : {y_pred[i]:.2f} Million")

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

print("\n========== MOVIE REVENUE PREDICTION ==========\n")

budget = float(input("Enter Movie Budget (Million): "))

actor_popularity = int(
    input("Enter Actor Popularity Score (1-100): ")
)

print("\nGenre")
print("0 = Action")
print("1 = Comedy")
print("2 = Drama")
print("3 = Thriller")

genre = int(input("Enter Genre: "))

trailer_views = float(
    input("Enter Trailer Views (Million): ")
)

social_media = int(
    input("Enter Social Media Engagement Score: ")
)

# =========================================================
# CREATE DATAFRAME FOR INPUT
# =========================================================

new_data = pd.DataFrame([[
    budget,
    actor_popularity,
    genre,
    trailer_views,
    social_media
]], columns=[
    "Budget_Million",
    "Actor_Popularity",
    "Genre",
    "Trailer_Views_Million",
    "Social_Media_Engagement"
])

# =========================================================
# PREDICT REVENUE
# =========================================================

prediction = model.predict(new_data)

predicted_revenue = prediction[0]

print("\n========== PREDICTION RESULT ==========\n")

print(
    f"Predicted Movie Revenue: "
    f"{predicted_revenue:.2f} Million"
)

# =========================================================
# GRAPH 1
# BUDGET VS REVENUE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Budget_Million"],
    df["Revenue_Million"]
)

plt.xlabel("Movie Budget (Million)")

plt.ylabel("Revenue (Million)")

plt.title("Budget vs Revenue")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 2
# TRAILER VIEWS VS REVENUE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Trailer_Views_Million"],
    df["Revenue_Million"]
)

plt.xlabel("Trailer Views (Million)")

plt.ylabel("Revenue (Million)")

plt.title("Trailer Views vs Revenue")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH 3
# SOCIAL MEDIA VS REVENUE
# =========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["Social_Media_Engagement"],
    df["Revenue_Million"]
)

plt.xlabel("Social Media Engagement")

plt.ylabel("Revenue (Million)")

plt.title("Social Media vs Revenue")

plt.grid(True)

plt.show()

# =========================================================
# GENRE ANALYSIS
# =========================================================

genre_avg = df.groupby(
    "Genre"
)["Revenue_Million"].mean()

print("\n========== GENRE ANALYSIS ==========\n")

print(genre_avg)

# =========================================================
# GENRE ANALYSIS GRAPH
# =========================================================

plt.figure(figsize=(8,5))

genre_avg.plot(kind="bar")

plt.xlabel("Genre")

plt.ylabel("Average Revenue")

plt.title("Genre vs Average Revenue")

plt.grid(True)

plt.show()

# =========================================================
# VISUALIZATION DASHBOARD
# =========================================================

print("\n========== MOVIE DASHBOARD ==========\n")

print(f"Minimum Revenue : {df['Revenue_Million'].min()} Million")

print(f"Maximum Revenue : {df['Revenue_Million'].max()} Million")

print(
    f"Average Revenue : "
    f"{df['Revenue_Million'].mean():.2f} Million"
)

# =========================================================
# SAVE DASHBOARD REPORT
# =========================================================

dashboard = pd.DataFrame({

    "Metric": [
        "Minimum Revenue",
        "Maximum Revenue",
        "Average Revenue"
    ],

    "Value": [
        df['Revenue_Million'].min(),
        df['Revenue_Million'].max(),
        df['Revenue_Million'].mean()
    ]
})

dashboard.to_csv(
    "movie_dashboard_report.csv",
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
    txt="Movie Revenue Prediction Report",
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
    txt=f"Predicted Revenue: {predicted_revenue:.2f} Million",
    ln=True
)

pdf.output("Movie_Revenue_Report.pdf")

print("\nPDF Report Generated Successfully")

print("Saved File: Movie_Revenue_Report.pdf")
# ========================================================
# END OF PROJECT
# =========================================================

# Revenue=b0+b1(Budget)+b2(Actor Popularity)+b3(Genre)+b4(Trailer Views)+b5(Social Media Engagement)

# The model learns:
# Which factors increase movie revenue
# Which factors reduce movie revenue
# How strongly each factor affects revenue

# Input Examples:
# Enter Movie Budget (Million): 120
# Enter Actor Popularity Score (1-100): 95 
# Enter Genre: 0
# Genre
# 0 = Action
# 1 = Comedy
# 2 = Drama
# 3 = Thriller
# Enter Trailer Views (Million): 70
# Enter Social Media Engagement Score: 92