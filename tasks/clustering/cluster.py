import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# STEP 1: CREATE DATASET
# ======================================================

np.random.seed(42)

data = {
    "Age": np.random.randint(18, 65, 200),
    "Income": np.random.randint(20000, 150000, 200),
    "Freq": np.random.randint(1, 30, 200),
    "Visits": np.random.randint(5, 100, 200),
}

df = pd.DataFrame(data)

df["Score"] = (
    df["Income"] * 0.0003 +
    df["Freq"] * 2 +
    df["Visits"] * 0.5 +
    np.random.randint(1, 30, 200)
)

# ======================================================
# STEP 2: KMEANS CLUSTERING
# ======================================================

X = df[["Income", "Freq", "Visits", "Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ======================================================
# STEP 3: SAFE CLUSTER LABELING
# ======================================================

order = df.groupby("Cluster")["Score"].mean().sort_values()

low_cluster = order.index[0]
med_cluster = order.index[1]
high_cluster = order.index[2]

df["Seg"] = df["Cluster"].map({
    low_cluster: "Low",
    med_cluster: "Med",
    high_cluster: "High"
})

# ======================================================
# STEP 4: ONE-HOT STYLE COLUMNS
# ======================================================

df["High"] = (df["Seg"] == "High").astype(int)
df["Low"] = (df["Seg"] == "Low").astype(int)
df["Med"] = (df["Seg"] == "Med").astype(int)

# ======================================================
# STEP 5: FINAL TABLE OUTPUT
# ======================================================

final_df = df[[
    "Age",
    "Income",
    "Freq",
    "Visits",
    "Score",
    "High",
    "Low",
    "Med"
]]

print(final_df.head(5).to_string(index=False))