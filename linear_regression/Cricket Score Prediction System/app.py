# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("cricket_score.csv")

# 3. Dataset preview
print("Dataset Preview:")
print(df.head())

# 4. Select features (X) and target (y)
X = df[["OversPlayed", "WicketsLost", "RunRate"]]
y = df["TotalRuns"]

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

# 10. Graph (Overs vs Runs)
plt.scatter(df["OversPlayed"], df["TotalRuns"])
plt.xlabel("Overs Played")
plt.ylabel("Total Runs")
plt.title("Overs vs Total Runs")
plt.show()

# 11. Live match input
print("\nEnter current match details:")

overs = float(input("Overs Played: "))
wickets = float(input("Wickets Lost: "))
runrate = float(input("Run Rate: "))

# Convert to array
user_data = pd.DataFrame([[overs, wickets, runrate]],
                         columns=["OversPlayed", "WicketsLost", "RunRate"])

# 12. Prediction
predicted_score = model.predict(user_data)

# 13. Output
print("\nPredicted Total Score:", int(predicted_score[0]))

# Example input:
# Overs Played: 25
# Wickets Lost: 3
# Run Rate: 8.2