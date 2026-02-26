import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create user behavior dataset
data = {
    "Posts_Per_Month": [5, 8, 6, 40, 45, 50],
    "Likes_Per_Day": [20, 25, 18, 200, 220, 250],
    "Time_Spent_Hours": [1, 1.5, 2, 5, 6, 7]
}

df = pd.DataFrame(data)

# Step 2: Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply KMeans
model = KMeans(n_clusters=2, random_state=42)
df["User_Group"] = model.fit_predict(scaled_data)

# Step 4: Print result
print(df)