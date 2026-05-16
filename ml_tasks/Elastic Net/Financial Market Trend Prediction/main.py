# =========================================================
# FINANCIAL MARKET TREND PREDICTION USING ELASTIC NET
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

data = pd.read_csv("financial_market_dataset.csv")

print(data.head())

# =========================
# FEATURES AND TARGET
# =========================

X = data[['open', 'high', 'low',
          'volume', 'MA_5',
          'MA_10', 'Daily_Return',
          'Prev_Close']]

y = data['close']

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
# VISUALIZATION
# =========================

plt.figure(figsize=(10,5))

plt.plot(y_test.values,
         label='Actual Price',
         marker='o')

plt.plot(y_pred,
         label='Predicted Price',
         marker='x')

plt.title("Financial Market Trend Prediction")
plt.xlabel("Test Data")
plt.ylabel("Stock Price")
plt.legend()

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

feature_names = X.columns

for feature, coef in zip(feature_names, model.coef_):
    print(feature, ":", coef)

# =========================
# FUTURE FORECAST
# =========================

future_prediction = model.predict(X_test[:5])

print("\nFuture Trend Forecast:")
print(future_prediction)

# =========================================================
# the model learns to predict stock prices based on various features such as opening price, high, 
# low, volume, moving averages, daily returns, and previous closing price.
# By analyzing the coefficients, we can understand which features have the most significant 
# impact on stock price predictions.
# Overall, this project demonstrates how Elastic Net Regression can be applied for financial market trend prediction,
# providing valuable insights for investors and traders.