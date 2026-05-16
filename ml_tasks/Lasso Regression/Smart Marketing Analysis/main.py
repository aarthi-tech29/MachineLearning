# =========================================================
# SMART MARKETING ANALYSIS USING LASSO REGRESSION
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

data = pd.read_csv("smart_marketing_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Ad_Spend',
    'Social_Media_Clicks',
    'Email_Clicks',
    'Website_Visits',
    'Customer_Age',
    'Time_Spent_Minutes',
    'Previous_Purchases',
    'Discount_Offered'
]]

y = data['Conversion_Rate']

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
model = Lasso(alpha=0.5)

model.fit(X_train_scaled, y_train)

# =========================================================
# CONVERSION PREDICTION
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
# NEW CAMPAIGN PREDICTION
# =========================================================

new_campaign = pd.DataFrame({
    'Ad_Spend': [45000],
    'Social_Media_Clicks': [7200],
    'Email_Clicks': [1300],
    'Website_Visits': [12500],
    'Customer_Age': [62],
    'Time_Spent_Minutes': [65],
    'Previous_Purchases': [12],
    'Discount_Offered': [38]
})

new_campaign_scaled = scaler.transform(new_campaign)

predicted_conversion = model.predict(new_campaign_scaled)

print("\nPredicted Conversion Rate:")
print(predicted_conversion[0])

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
# IMPORTANT FACTOR SELECTION
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

plt.title("Marketing Feature Importance using Lasso",
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
         label='Actual Conversion')

plt.plot(y_pred,
         marker='s',
         label='Predicted Conversion')

plt.title("Actual vs Predicted Conversion Rate")

plt.xlabel("Test Samples")
plt.ylabel("Conversion Rate")

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

print("\nSmart Marketing Analysis Completed Successfully")

# the model learns to predict conversion rates based on various marketing factors such as ad spend, social media clicks, 
# email clicks, website visits, customer demographics, and previous purchase behavior.
# By analyzing the coefficients, we can understand which features have the most significant
#  impact on conversion rates and make informed decisions for marketing strategies and 
# resource allocation.
# Overall, this project demonstrates how Lasso Regression can be applied for feature selection
#  and conversion prediction in the marketing domain, providing valuable insights for 
# marketers and business stakeholders.