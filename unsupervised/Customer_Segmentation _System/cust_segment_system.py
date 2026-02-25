import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create dataset
data = {
    "Annual_Income": [20000, 25000, 30000, 80000, 85000, 90000],
    "Spending_Score": [20, 25, 30, 75, 80, 85]
}

df = pd.DataFrame(data)

# Step 2: Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# StandardScaler is used to scale (normalize) data so that all features are on the same level.
# Z=(X−Mean)​/Standard Deviation
# Income values are much bigger.
# If we don’t scale:
# KMeans will think income is more important
# Spending score effect becomes very small
# So clustering becomes biased.
# After scaling - examples Income - -1.2, -0.8, Spending score - -1.1, -0.9 - Now both features are balanced

# Step 3: Apply KMeans
model = KMeans(n_clusters=2, random_state=42)
df["Customer_Segment"] = model.fit_predict(scaled_data)

# Step 4: Print result
print(df)

# 0 - Low Income + Low Spending
# 1 - High Income + High Spending