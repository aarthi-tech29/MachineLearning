# ============================================
# SHIPPING ROUTE ANOMALY DETECTION
# USING DBSCAN
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("shipping_route_dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ============================================
# SELECT FEATURES
# ============================================

# GPS Tracking Coordinates
X = data[['Latitude', 'Longitude']]

# ============================================
# DATA SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# APPLY DBSCAN ALGORITHM
# ============================================

dbscan = DBSCAN(eps=0.5, min_samples=3)

# Train model
data['Cluster'] = dbscan.fit_predict(X_scaled)

# ============================================
# DISPLAY CLUSTER RESULTS
# ============================================

print("\nCluster Results:\n")
print(data[['Ship_ID',
            'Latitude',
            'Longitude',
            'Cluster']])

# ============================================
# ROUTE CLUSTERING
# ============================================

normal_routes = data[data['Cluster'] != -1]

print("\nNormal Shipping Route Clusters:\n")
print(normal_routes)

# ============================================
# SUSPICIOUS ROUTE DETECTION
# ============================================

suspicious_routes = data[data['Cluster'] == -1]

print("\nSuspicious / Anomalous Routes:\n")
print(suspicious_routes)

# ============================================
# ALERT SYSTEM
# ============================================

print("\n===================================")
print("SHIPPING SECURITY ALERT SYSTEM")
print("===================================")

if len(suspicious_routes) > 0:
    print("⚠ Suspicious Shipping Routes Detected!")
    print("⚠ Maritime Security Alert Activated!")
else:
    print("No Suspicious Routes Found")

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    data['Longitude'],
    data['Latitude'],
    c=data['Cluster'],
    cmap='rainbow',
    s=120
)

# Labels
plt.title("Shipping Route Anomaly Detection using DBSCAN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Display Ship IDs
for i in range(len(data)):
    plt.text(
        data['Longitude'][i],
        data['Latitude'][i],
        str(data['Ship_ID'][i]),
        fontsize=8
    )

plt.colorbar(scatter, label='Cluster')

plt.show()

# ============================================
# FINAL SUMMARY
# ============================================

total_clusters = len(set(data['Cluster'])) - (1 if -1 in data['Cluster'].values else 0)
total_suspicious = len(suspicious_routes)

print("\n===================================")
print("SHIPPING ROUTE ANALYSIS SUMMARY")
print("===================================")

print("Total Route Clusters Found :", total_clusters)
print("Total Suspicious Routes :", total_suspicious)

print("\nDBSCAN Successfully Detected:")
print("✔ GPS Tracking Analysis")
print("✔ Route Clustering")
print("✔ Suspicious Route Detection")
print("✔ Maritime Security Monitoring")

# the model learns to identify clusters of shipping routes based on their GPS coordinates, allowing us to detect potential anomalies and suspicious routes in the maritime data.
# By analyzing the clusters and outliers, we can gain insights into shipping route patterns and enhance maritime security measures.
# Overall, this project demonstrates how DBSCAN can be effectively applied to detect anomalies in shipping routes
# using GPS tracking data, providing valuable insights for maritime authorities and shipping companies.
# By analyzing the clusters and outliers, we can gain insights into shipping route patterns and enhance maritime security measures. 
# Overall, this project demonstrates how DBSCAN can be effectively applied to detect anomalies in shipping routes using GPS tracking data, providing valuable insights for maritime authorities and shipping companies.
