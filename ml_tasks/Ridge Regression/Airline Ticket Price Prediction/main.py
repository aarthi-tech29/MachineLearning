# =========================================================
# AIRLINE TICKET PRICE PREDICTION USING RIDGE REGRESSION
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("airline_ticket_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Travel_Season',
    'Flight_Distance_KM',
    'Airline_Category',
    'Passenger_Demand',
    'Baggage_Weight',
    'Fuel_Price'
]]

y = data['Ticket_Price']

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
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# RIDGE REGRESSION MODEL
# =========================================================

# Ridge Regression controls overfitting
model = Ridge(alpha=1.0)

model.fit(X_train_scaled, y_train)

# =========================================================
# TICKET FARE PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)

# =========================================================
# MODEL EVALUATION
# =========================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# =========================================================
# TICKET FARE PREDICTION ENGINE
# =========================================================

new_flight = pd.DataFrame({
    'Travel_Season': [4],
    'Flight_Distance_KM': [4500],
    'Airline_Category': [3],
    'Passenger_Demand': [110],
    'Baggage_Weight': [40],
    'Fuel_Price': [132]
})

new_flight_scaled = scaler.transform(new_flight)

predicted_price = model.predict(new_flight_scaled)

print("\nPredicted Airline Ticket Price:")
print(predicted_price[0])

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(y_test.values,
         marker='o',
         label='Actual Ticket Price')

plt.plot(y_pred,
         marker='s',
         label='Predicted Ticket Price')

plt.title("Actual vs Predicted Ticket Price")
plt.xlabel("Test Samples")
plt.ylabel("Ticket Price")

plt.legend()
plt.grid(True)

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients")
print(importance)

# =========================================================
# TRAVEL SEASON ANALYSIS
# =========================================================

season_analysis = data.groupby('Travel_Season')['Ticket_Price'].mean()

print("\nTravel Season Analysis")
print(season_analysis)

# =========================================================
# OVERFITTING CHECK
# =========================================================

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\nTraining Score:", train_score)
print("Testing Score :", test_score)

if abs(train_score - test_score) < 0.1:
    print("\nOverfitting Controlled Successfully")
else:
    print("\nPossible Overfitting Detected")

# =========================================================
# AIRLINE TICKET PRICE PREDICTION COMPLETE
# =========================================================

print("\nAirline Ticket Price Prediction System Executed Successfully")

# the model learns to predict airline ticket prices based on various factors such as travel season,
#  flight distance, airline category, passenger demand, baggage weight, and fuel price.
# By analyzing the coefficients, we can understand which features have the most significant impact on ticket prices and make informed decisions for pricing strategies and customer targeting.
# Overall, this project demonstrates how Ridge Regression can be applied to predict airline ticket prices, providing valuable insights for airlines and travelers alike.