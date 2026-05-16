# ============================================
# ATM FRAUD ZONE DETECTION USING DBSCAN
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("atm_fraud_dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ============================================
# SELECT FEATURES
# ============================================

# Using ATM transaction locations
X = data[['Latitude', 'Longitude']]

# ============================================
# DATA SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# APPLY DBSCAN ALGORITHM
# ============================================

# eps = neighborhood distance
# min_samples = minimum points required

dbscan = DBSCAN(eps=0.5, min_samples=3)

# Train model
data['Cluster'] = dbscan.fit_predict(X_scaled)

# ============================================
# DISPLAY CLUSTERS
# ============================================

print("\nCluster Results:\n")
print(data[['Transaction_ID', 'Latitude',
            'Longitude', 'Cluster']])

# ============================================
# IDENTIFY OUTLIERS
# ============================================

# DBSCAN marks outliers as -1

outliers = data[data['Cluster'] == -1]

print("\nOutlier Transactions:\n")
print(outliers)

# ============================================
# FRAUD HOTSPOT CLUSTERING
# ============================================

fraud_clusters = data[data['Cluster'] != -1]

print("\nFraud Hotspot Clusters:\n")
print(fraud_clusters)

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(5,7))

# Scatter plot
scatter = plt.scatter(
    data['Longitude'],
    data['Latitude'],
    c=data['Cluster'],
    cmap='rainbow',
    s=100
)

# Labels
plt.title("ATM Fraud Zone Detection using DBSCAN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Color bar
plt.colorbar(scatter, label='Cluster')

# Show transaction IDs
for i in range(len(data)):
    plt.text(
        data['Longitude'][i],
        data['Latitude'][i],
        str(data['Transaction_ID'][i]),
        fontsize=8
    )

plt.show()

# ============================================
# FRAUD ANALYSIS SUMMARY
# ============================================

total_clusters = len(set(data['Cluster'])) - (1 if -1 in data['Cluster'].values else 0)
total_outliers = len(outliers)

print("\n===================================")
print("ATM FRAUD ZONE DETECTION SUMMARY")
print("===================================")

print("Total Clusters Found :", total_clusters)
print("Total Outliers Found :", total_outliers)

print("\nDBSCAN Successfully Detected:")
print("✔ Fraud Hotspot Clustering")
print("✔ Suspicious ATM Zones")
print("✔ Outlier Transactions")
print("✔ Transaction Location Analysis")

# the model learns to identify clusters of ATM transactions based on their geographical locations, allowing us to detect potential fraud zones and outliers in the transaction data. 
# By analyzing the clusters and outliers, we can gain insights into suspicious activities and enhance fraud detection strategies for ATM transactions. 
# Overall, this project demonstrates how DBSCAN can be effectively applied to detect fraud zones in ATM transaction data, providing valuable 
# insights for financial institutions and law enforcement agencies.
# DBSCAN is a powerful clustering algorithm that can identify clusters of varying shapes and sizes, making it well-suited for detecting fraud zones in ATM transaction data. By analyzing the clusters and outliers, we can gain insights into suspicious activities and enhance fraud detection strategies for ATM transactions. Overall, this project demonstrates how DBSCAN can be effectively applied to detect fraud zones in ATM transaction data, providing valuable insights for financial institutions and law enforcement agencies.
