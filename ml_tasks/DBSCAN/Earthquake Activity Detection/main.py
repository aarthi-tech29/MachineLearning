# ============================================
# EARTHQUAKE ACTIVITY DETECTION USING DBSCAN
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("earthquake_dataset.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ============================================
# SELECT FEATURES
# ============================================

# Using earthquake locations
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
print(data[['Earthquake_ID',
            'Latitude',
            'Longitude',
            'Cluster']])

# ============================================
# IDENTIFY RISK AREAS
# ============================================

risk_areas = data[data['Cluster'] != -1]

print("\nDense Earthquake Risk Areas:\n")
print(risk_areas)

# ============================================
# IDENTIFY OUTLIERS
# ============================================

outliers = data[data['Cluster'] == -1]

print("\nOutlier Earthquake Locations:\n")
print(outliers)

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
plt.title("Earthquake Activity Detection using DBSCAN")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Display earthquake IDs
for i in range(len(data)):
    plt.text(
        data['Longitude'][i],
        data['Latitude'][i],
        str(data['Earthquake_ID'][i]),
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
print("EARTHQUAKE ACTIVITY DETECTION")
print("===================================")

print("Total Dense Regions Found :", total_clusters)
print("Total Outlier Locations :", total_outliers)

print("\nDBSCAN Successfully Detected:")
print("✔ Seismic Activity Clusters")
print("✔ Dense Earthquake Regions")
print("✔ Risk Area Identification")
print("✔ Outlier Earthquake Locations")

# the model learns to identify clusters of earthquake activity based on their geographical locations, 
# allowing us to detect potential risk zones and outliers in the earthquake data. 
# By analyzing the clusters and outliers, we can gain insights into seismic activity patterns and enhance earthquake preparedness strategies. 
# Overall, this project demonstrates how DBSCAN can be effectively applied to detect earthquake activity zones in geographical data, 
# providing valuable insights for disaster management and mitigation efforts.