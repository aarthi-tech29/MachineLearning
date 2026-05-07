import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# ======================================================
# STEP 1: CREATE CUSTOMER DATASET
# ======================================================
np.random.seed(42)

data = {
    "Age": np.random.randint(18, 65, 200),
    "Annual_Income": np.random.randint(20000, 150000, 200),
    "Purchase_Frequency": np.random.randint(1, 30, 200),
    "Website_Visits": np.random.randint(5, 100, 200),
}

df = pd.DataFrame(data)

df["Spending_Score"] = (
    df["Annual_Income"] * 0.0003 +
    df["Purchase_Frequency"] * 2 +
    df["Website_Visits"] * 0.5 +
    np.random.randint(1, 30, 200)
)

# ======================================================
# STEP 2: K-MEANS CLUSTERING
# ======================================================
# Selecting features for clustering
X = df[["Annual_Income", "Purchase_Frequency", "Website_Visits", "Spending_Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying KMeans to create 3 segments (Low, Medium, High)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ======================================================
# STEP 3: MAPPING CLUSTERS TO VALUE LABELS
# ======================================================
# Sort clusters by their average Spending_Score to identify Low/Med/High correctly
order = df.groupby("Cluster")["Spending_Score"].mean().sort_values().index

df["Low_Value_Customer"] = (df["Cluster"] == order[0]).astype(int)
df["Medium_Value_Customer"] = (df["Cluster"] == order[1]).astype(int)
df["High_Value_Customer"] = (df["Cluster"] == order[2]).astype(int)

# ======================================================
# STEP 4: FINAL OUTPUT
# ======================================================
# Selecting only the columns shown in your picture
final_df = df[[
    "Age", 
    "Annual_Income", 
    "Purchase_Frequency", 
    "Website_Visits", 
    "Spending_Score", 
    "Low_Value_Customer", 
    "Medium_Value_Customer", 
    "High_Value_Customer"
]]

print("Dataset:")
print(final_df.head(5).to_string(index=True))