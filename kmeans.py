import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
 
 
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
 
df["High_Value_Customer"] = np.where(df["Spending_Score"] > df["Spending_Score"].median(), 1, 0)
 
print("Dataset:")
print(df.head())
 
 
# ======================================================
# STEP 2: FEATURE SCALING
# ======================================================
 
features = ["Age", "Annual_Income", "Purchase_Frequency", "Website_Visits"]
 
X = df[features]
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
 
# ======================================================
# STEP 3: K-MEANS CLUSTERING
# ======================================================
 
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
 
print("\nK-Means Cluster Output:")
print(df[["Age", "Annual_Income", "Spending_Score", "KMeans_Cluster"]].head())

# ======================================================
# STEP 4: ELBOW METHOD
# ======================================================
 
wcss = []
 
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
 
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
 
# ======================================================
# STEP 5: HIERARCHICAL CLUSTERING
# ======================================================
 
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)
 
print("\nHierarchical Cluster Output:")
print(df[["Age", "Annual_Income", "Hierarchical_Cluster"]].head())
 
 
# Dendrogram
linked = linkage(X_scaled[:30], method="ward")
 
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()
 
 
# ======================================================
# STEP 6: DBSCAN CLUSTERING
# ======================================================
 
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
 
print("\nDBSCAN Cluster Output:")
print(df[["Age", "Annual_Income", "DBSCAN_Cluster"]].head())
 
print("\nDBSCAN Cluster Counts:")
print(df["DBSCAN_Cluster"].value_counts())