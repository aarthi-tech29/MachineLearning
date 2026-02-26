import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create employee performance dataset
data = {
    "Experience_Years": [1, 2, 3, 8, 9, 10],
    "Performance_Score": [60, 65, 70, 85, 90, 95],
    "Projects_Completed": [2, 3, 4, 10, 12, 15]
}

df = pd.DataFrame(data)

# Step 2: Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply KMeans clustering
model = KMeans(n_clusters=2, random_state=42)
df["Performance_Group"] = model.fit_predict(scaled_data)

# Step 4: Print result
print(df)