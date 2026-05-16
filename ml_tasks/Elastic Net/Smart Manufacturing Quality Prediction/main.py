# =========================================================
# SMART MANUFACTURING QUALITY PREDICTION USING ELASTIC NET
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("manufacturing_quality_dataset.csv")

print(data.head())

# =========================
# FEATURES AND TARGET
# =========================

X = data[['Temperature',
          'Pressure',
          'Humidity',
          'Vibration',
          'Machine_Speed',
          'Defect_Rate']]

y = data['Quality_Score']

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
# ELASTIC NET MODEL
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
# PRODUCTION QUALITY PREDICTION
# =========================

comparison = pd.DataFrame({
    'Actual Quality': y_test.values,
    'Predicted Quality': y_pred
})

print("\nProduction Quality Prediction")

print(comparison)

# =========================
# SENSOR DATA ANALYSIS
# =========================

print("\nSensor Dataset Analysis")

print(data.describe())

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(10,5))

plt.plot(y_test.values,
         label='Actual Quality',
         marker='o')

plt.plot(y_pred,
         label='Predicted Quality',
         marker='x')

plt.title("Smart Manufacturing Quality Prediction")

plt.xlabel("Production Units")

plt.ylabel("Quality Score")

plt.legend()

plt.show()

# =========================
# FEATURE OPTIMIZATION
# =========================

print("\nFeature Importance")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)

# =========================
# FUTURE QUALITY FORECAST
# =========================

future_prediction = model.predict(X_test[:5])

print("\nFuture Production Quality Prediction")

print(future_prediction)

# =========================================================
# the model learns to predict the quality score of manufactured products based on various
#  sensor readings and machine parameters.
# By analyzing the coefficients, we can understand which features have the most significant
#  impact on product quality and make informed decisions for process optimization and
# quality control in the manufacturing environment.
# Overall, this project demonstrates how Elastic Net Regression can be applied for quality prediction
# in smart manufacturing, providing valuable insights for production managers and engineers.