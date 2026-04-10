# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv("delivery_time.csv")

# 3. Preview
print("Dataset Preview:")
print(df.head())

# 4. Features & Target
X = df[["Distance", "TrafficLevel", "OrderPreparationTime"]]
y = df["DeliveryTime"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Compare
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison)

# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 10. User input (FIXED - no warning)
print("\nEnter delivery details:")

distance = float(input("Distance (km): "))
traffic = int(input("Traffic Level (1-5): "))
prep_time = float(input("Preparation Time: "))

user_data = pd.DataFrame([[distance, traffic, prep_time]],
                         columns=["Distance", "TrafficLevel", "OrderPreparationTime"])

# 11. Prediction
predicted_time = model.predict(user_data)

# 12. Output
print("\nEstimated Delivery Time:", int(predicted_time[0]), "minutes")

# Example input:
# Distance: 5
# Traffic: 3
# Preparation Time: 15