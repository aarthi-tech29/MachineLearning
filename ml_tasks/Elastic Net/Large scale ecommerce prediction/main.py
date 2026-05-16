# =========================================================
# E-COMMERCE REVENUE PREDICTION USING ELASTIC NET
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("ecommerce_dataset.csv")

print("\nDataset Preview:")
print(data.head())

# =========================
# FEATURES AND TARGET
# =========================
X = data[['Product_Price',
          'Discount',
          'Page_Views',
          'Cart_Additions',
          'Purchases',
          'User_Rating']]

y = data['Revenue']

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL (ELASTIC NET)
# =========================
model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# =========================
# USER BEHAVIOR ANALYSIS
# =========================
print("\nUser Behavior Summary:")
print(data[['Page_Views', 'Cart_Additions', 'Purchases']].describe())

# =========================
# FEATURE IMPORTANCE
# =========================
print("\nFeature Importance (Elastic Net):")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# =========================
# ACTUAL VS PREDICTED DATAFRAME
# =========================
comparison = pd.DataFrame({
    "Actual_Revenue": y_test.values,
    "Predicted_Revenue": y_pred
})

print("\nPrediction Table:")
print(comparison)

# =========================
# FIXED VISUALIZATION (NO SHIFT ISSUE)
# =========================
# comparison = comparison.reset_index(drop=True)
plt.figure(figsize=(10,5))

x = np.arange(len(y_test))

plt.plot(x, y_test.values, marker='o', label="Actual Revenue")
plt.plot(x, y_pred, marker='x', label="Predicted Revenue")

plt.title("E-commerce Revenue Prediction (Elastic Net)")
plt.xlabel("Test Samples")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)

plt.show()

# =========================
# FUTURE PREDICTION
# =========================
future_prediction = model.predict(X_test[:5])

print("\nFuture Revenue Prediction:")
print(future_prediction)

# ========================================================
# the model learns to predict e-commerce revenue based on various features such as
#  product price, discount, page views, cart additions, purchases, and user ratings.
# By analyzing the coefficients, we can understand which features have the most significant
#  impact on revenue predictions and make informed decisions for marketing strategies, 
# pricing, and inventory management.
# Overall, this project demonstrates how Elastic Net can be applied for regression tasks
#  in the e-commerce domain, providing valuable insights for business growth and customer
#  engagement strategies.