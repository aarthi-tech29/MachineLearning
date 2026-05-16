# =========================================================
# REAL ESTATE IMPORTANT FACTOR ANALYSIS USING LASSO
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("real_estate_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Area_Sqft',
    'Bedrooms',
    'Bathrooms',
    'Parking_Spaces',
    'Age_Of_Property',
    'Distance_To_City',
    'Nearby_Schools',
    'Crime_Rate'
]]

y = data['Property_Price']

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
# LASSO REGRESSION MODEL
# =========================================================

# Lasso removes unnecessary features
model = Lasso(alpha=1000)

model.fit(X_train_scaled, y_train)

# =========================================================
# PRICE PREDICTION
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
# NEW PROPERTY PREDICTION
# =========================================================

new_property = pd.DataFrame({
    'Area_Sqft': [2800],
    'Bedrooms': [9],
    'Bathrooms': [8],
    'Parking_Spaces': [7],
    'Age_Of_Property': [1],
    'Distance_To_City': [1],
    'Nearby_Schools': [10],
    'Crime_Rate': [1]
})

new_property_scaled = scaler.transform(new_property)

predicted_price = model.predict(new_property_scaled)

print("\nPredicted Property Price:")
print(predicted_price[0])

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance")
print(importance)

# =========================================================
# FEATURE REDUCTION
# =========================================================

selected_features = importance[importance['Coefficient'] != 0]

print("\nImportant Selected Features")
print(selected_features)

removed_features = importance[importance['Coefficient'] == 0]

print("\nRemoved Unnecessary Features")
print(removed_features)

# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

plt.figure(figsize=(18,9))

plt.bar(importance['Feature'],
        importance['Coefficient'])

plt.title("Real Estate Feature Importance using Lasso",
          fontsize=22)

plt.xlabel("Features", fontsize=18, labelpad=20)
plt.ylabel("Coefficient Value", fontsize=18)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.subplots_adjust(left=0.18, bottom=0.35)

plt.grid(True)

plt.show()

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(y_test.values,
         marker='o',
         label='Actual Price')

plt.plot(y_pred,
         marker='s',
         label='Predicted Price')

plt.title("Actual vs Predicted Property Price")

plt.xlabel("Test Samples")
plt.ylabel("Property Price")

plt.legend()
plt.grid(True)

plt.show()

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
# PROJECT COMPLETE
# =========================================================

print("\nReal Estate Important Factor Analysis Completed Successfully")

# the model learns to predict property prices based on various features such as area,
#  number of bedrooms, bathrooms, parking spaces, age of the property, distance to city center,
#  nearby schools, and crime rate.
# By analyzing the coefficients, we can understand which features have the most significant impact on
#  property prices and make informed decisions for real estate investments and pricing strategies.
# Overall, this project demonstrates how Lasso Regression can be applied for feature selection and 
# price prediction in the real estate domain, providing valuable insights for buyers, sellers, 
# and investors alike.