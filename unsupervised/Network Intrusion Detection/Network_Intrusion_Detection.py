import pandas as pd
from sklearn.ensemble import IsolationForest

# Step 1: Create network traffic dataset
data = {
    "Packet_Size": [500, 520, 510, 530, 505, 515, 2000, 2500],
    "Connection_Duration": [30, 35, 32, 28, 33, 31, 5, 3],
    "Failed_Login_Attempts": [1, 0, 1, 2, 1, 0, 10, 15]
}

# Packet_Size → size of data packets
# Connection_Duration → how long the session lasts
# Failed_Login_Attempts → login failures

df = pd.DataFrame(data)

# Step 2: Train Isolation Forest model
model = IsolationForest(contamination=0.25, random_state=42)
model.fit(df)

# Step 3: Predict anomalies
df["Intrusion"] = model.predict(df)

# Interpretation:
#  1  → Normal traffic
# -1  → Intrusion (Anomaly)

print(df)