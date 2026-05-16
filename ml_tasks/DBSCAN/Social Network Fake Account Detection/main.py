# ============================================
# SOCIAL NETWORK FAKE ACCOUNT DETECTION
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

data = pd.read_csv("social_network_fake_accounts.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ============================================
# SELECT FEATURES
# ============================================

# User activity analysis
X = data[['Posts_Per_Day',
          'Likes_Per_Day',
          'Followers',
          'Following',
          'Messages_Per_Day']]

# ============================================
# DATA SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# APPLY DBSCAN ALGORITHM
# ============================================

dbscan = DBSCAN(eps=1.2, min_samples=3)

# Train model
data['Cluster'] = dbscan.fit_predict(X_scaled)

# ============================================
# DISPLAY CLUSTER RESULTS
# ============================================

print("\nCluster Results:\n")
print(data[['User_ID', 'Cluster']])

# ============================================
# UNUSUAL BEHAVIOR CLUSTERING
# ============================================

normal_users = data[data['Cluster'] != -1]

print("\nNormal User Activity Clusters:\n")
print(normal_users)

# ============================================
# FAKE PROFILE DETECTION
# ============================================

fake_accounts = data[data['Cluster'] == -1]

print("\nDetected Fake / Suspicious Accounts:\n")
print(fake_accounts)

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    data['Followers'],
    data['Following'],
    c=data['Cluster'],
    cmap='rainbow',
    s=120
)

# Labels
plt.title("Social Network Fake Account Detection using DBSCAN")
plt.xlabel("Followers")
plt.ylabel("Following")

# Show User IDs
for i in range(len(data)):
    plt.text(
        data['Followers'][i],
        data['Following'][i],
        str(data['User_ID'][i]),
        fontsize=8
    )

plt.colorbar(scatter, label='Cluster')

plt.show()

# ============================================
# FAKE ACCOUNT ALERT SYSTEM
# ============================================

print("\n===================================")
print("FAKE ACCOUNT ALERT SYSTEM")
print("===================================")

if len(fake_accounts) > 0:
    print("⚠ Suspicious Fake Accounts Detected!")
    print("⚠ Social Network Security Alert Activated!")
else:
    print("No Fake Accounts Found")

# ============================================
# FINAL SUMMARY
# ============================================

total_clusters = len(set(data['Cluster'])) - (1 if -1 in data['Cluster'].values else 0)
total_fake_accounts = len(fake_accounts)

print("\n===================================")
print("SOCIAL NETWORK ANALYSIS SUMMARY")
print("===================================")

print("Total User Clusters Found :", total_clusters)
print("Total Fake Accounts Found :", total_fake_accounts)

print("\nDBSCAN Successfully Detected:")
print("✔ User Activity Analysis")
print("✔ Unusual Behavior Clustering")
print("✔ Fake Profile Detection")
print("✔ Suspicious User Identification")

# the model learns to identify clusters of user activity based on their behavior patterns, allowing us to detect 
# potential fake accounts and outliers in the social network data.
# By analyzing the clusters and outliers, we can gain insights into user behavior 
# and enhance security measures on the social network platform.
# Overall, this project demonstrates how DBSCAN can be effectively applied to detect 
# fake accounts in social network data, providing valuable insights for social media platforms 
# and users alike.
