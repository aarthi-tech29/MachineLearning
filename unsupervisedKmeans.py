from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Create dataset
data = {
    "Math_Marks": [40, 45, 50, 85, 90, 95],
    "Science_Marks": [42, 48, 55, 88, 92, 96]
}

df = pd.DataFrame(data)

# Step 2: Apply KMeans
model = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = model.fit_predict(df)

# Step 3: Print result
print(df)

from sklearn.cluster import KMeans
import pandas as pd

data = {
    "Salary": [20000, 25000, 30000, 80000, 85000, 90000]
}

df = pd.DataFrame(data)

model = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = model.fit_predict(df)

print(df)