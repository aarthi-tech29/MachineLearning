import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create user–product interaction data
# (Example: average ratings / interactions)
data = {
    "Avg_Rating": [2.0, 2.5, 3.0, 4.5, 4.7, 5.0],
    "Purchases_Per_Month": [1, 2, 2, 10, 12, 15],
    "Time_Spent_Hours": [0.5, 1.0, 1.2, 4.5, 5.0, 6.0]
}

df = pd.DataFrame(data)

# Step 2: Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df["User_Cluster"] = kmeans.fit_predict(scaled_data)

# Step 4: View clustered users
print(df)