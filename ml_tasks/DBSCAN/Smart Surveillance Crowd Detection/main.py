# ============================================
# SMART SURVEILLANCE CROWD DETECTION
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

data = pd.read_csv("crowd_detection_dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ============================================
# SELECT FEATURES
# ============================================

# CCTV Coordinates
X = data[['X_Coordinate', 'Y_Coordinate']]

# ============================================
# DATA SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# APPLY DBSCAN ALGORITHM
# ============================================

dbscan = DBSCAN(eps=0.5, min_samples=3)

# Train Model
data['Cluster'] = dbscan.fit_predict(X_scaled)

# ============================================
# DISPLAY CLUSTER RESULTS
# ============================================

print("\nCluster Results:\n")
print(data[['Person_ID',
            'X_Coordinate',
            'Y_Coordinate',
            'Cluster']])

# ============================================
# CROWD DENSITY DETECTION
# ============================================

crowd_clusters = data[data['Cluster'] != -1]

print("\nDense Crowd Areas:\n")
print(crowd_clusters)

# ============================================
# OUTLIER DETECTION
# ============================================

outliers = data[data['Cluster'] == -1]

print("\nSparse / Suspicious Locations:\n")
print(outliers)

# ============================================
# EMERGENCY ALERT SYSTEM
# ============================================

print("\n===================================")
print("EMERGENCY ALERT SYSTEM")
print("===================================")

if len(crowd_clusters) > 10:
    print("⚠ High Crowd Density Detected!")
    print("⚠ Security Team Alert Activated!")
else:
    print("Crowd Density is Normal")

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    data['X_Coordinate'],
    data['Y_Coordinate'],
    c=data['Cluster'],
    cmap='rainbow',
    s=120
)

# Labels
plt.title("Smart Surveillance Crowd Detection using DBSCAN")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Show Person IDs
for i in range(len(data)):
    plt.text(
        data['X_Coordinate'][i],
        data['Y_Coordinate'][i],
        str(data['Person_ID'][i]),
        fontsize=8
    )

plt.colorbar(scatter, label='Cluster')

plt.show()

# ============================================
# FINAL SUMMARY
# ============================================

total_clusters = len(set(data['Cluster'])) - (1 if -1 in data['Cluster'].values else 0)
total_outliers = len(outliers)

print("\n===================================")
print("SMART SURVEILLANCE SUMMARY")
print("===================================")

print("Total Crowd Clusters Found :", total_clusters)
print("Total Outlier Locations :", total_outliers)

print("\nDBSCAN Successfully Detected:")
print("✔ CCTV Coordinate Analysis")
print("✔ Crowd Density Detection")
print("✔ Emergency Alert System")
print("✔ Suspicious Area Identification")

# the model learns to identify clusters of people based on their coordinates, allowing us to detect areas of high crowd density
#  and potential outliers in the surveillance data. 
# By analyzing the clusters and outliers, we can gain insights into crowd behavior and enhance security measures in public spaces.
#  Overall, this project demonstrates how DBSCAN can be effectively applied to smart surveillance data for crowd detection,
#  providing valuable insights for security personnel and event organizers.