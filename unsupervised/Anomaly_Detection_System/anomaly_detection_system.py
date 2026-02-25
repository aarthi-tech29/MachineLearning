import pandas as pd
from sklearn.ensemble import IsolationForest

# Step 1: Create dataset
data = {
    "Transaction_Amount": [100, 120, 130, 115, 105, 110, 5000, 7000],
    "Transaction_Time": [10, 11, 9, 10, 12, 11, 2, 3]
}

# Small amounts → Normal
# 5000, 7000 → Unusual

df = pd.DataFrame(data)

# Step 2: Train Isolation Forest model
model = IsolationForest(contamination=0.25, random_state=42)
model.fit(df)

# IsolationForest-Special algorithm for detecting unusual data.
# contamination=0.25 → 25% of the data is expected to be anomalies.
# have 8 rows:
# 25% of 8 = 2
# So model will mark 2 points as anomaly.

# Isolation Forest Works:
# Randomly splits data
# Normal points need many splits
# Anomalies get isolated quickly
# So they are marked as -1

# Step 3: Predict anomalies
df["Anomaly"] = model.predict(df)

# Interpretation:
# -1 = Anomaly
#  1 = Normal

print(df)