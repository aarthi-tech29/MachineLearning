# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("ecommerce_sales.csv")

#  FIX: Remove commas and convert to numeric
df = df.replace({',': ''}, regex=True)
df = df.apply(pd.to_numeric)

# Check data types
print("Data Types:\n", df.dtypes)

# 3. Dataset preview
print("\nDataset Preview:")
print(df.head())

# 4. Select features (X) and target (y)
X = df[["WebsiteVisitors", "AdSpend", "DiscountPercentage", "PreviousDaySales"]]
y = df["Sales"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Compare actual vs predicted
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison)

# 9. Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 10. Visualization
plt.scatter(df["AdSpend"], df["Sales"])
plt.xlabel("Ad Spend")
plt.ylabel("Sales")
plt.title("AdSpend vs Sales")
plt.show()

# 11. User input
print("\nEnter values to predict sales:")

visitors = float(input("Website Visitors: "))
ad_spend = float(input("Ad Spend: "))
discount = float(input("Discount Percentage: "))
prev_sales = float(input("Previous Day Sales: "))

# Convert to array
user_data = pd.DataFrame([[visitors, ad_spend, discount, prev_sales]],
                         columns=["WebsiteVisitors", "AdSpend", "DiscountPercentage", "PreviousDaySales"])

# 12. Prediction
predicted_sales = model.predict(user_data)

# 13. Output
print("\nPredicted Sales:", predicted_sales[0])

# Example input
# Website Visitors: 1600
# Ad Spend: 700
# Discount: 15
# Previous Sales: 5000