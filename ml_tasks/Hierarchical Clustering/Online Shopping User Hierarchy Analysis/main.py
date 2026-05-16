# =====================================================
# ONLINE SHOPPING USER HIERARCHY ANALYSIS
# USING HIERARCHICAL CLUSTERING
# =====================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("shopping_users.csv")

print("\n========== DATASET ==========")
print(data)

# =========================
# USER BEHAVIOR TRACKING
# =========================

features = data[[
    "TimeSpent",
    "ProductsViewed",
    "PurchaseAmount",
    "Electronics",
    "Fashion",
    "Grocery",
    "HomeAppliances"
]]

print("\n========== USER BEHAVIOR FEATURES ==========")
print(features)

# =========================
# DATA SCALING
# =========================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(features)

print("\n========== DATA SCALING COMPLETED ==========")

# =========================
# HIERARCHICAL CLUSTERING
# =========================

linkage_matrix = linkage(scaled_data, method='ward')

print("\n========== HIERARCHICAL CLUSTERING COMPLETED ==========")

# =========================
# PURCHASE HIERARCHY
# =========================

plt.figure(figsize=(5,7))

plt.title("Online Shopping User Hierarchy")

dendrogram(
    linkage_matrix,
    labels=data["UserID"].values,
    leaf_rotation=90
)

plt.xlabel("Users")
plt.ylabel("Distance")

plt.tight_layout()

# =========================
# PRODUCT PREFERENCE GROUPING
# =========================

plt.show()

print("\n========== PRODUCT PREFERENCE GROUPING DISPLAYED ==========")

# =========================
# FINAL OUTPUT
# =========================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")

print("User Behavior Tracking Completed")
print("Purchase Hierarchy Completed")
print("Product Preference Grouping Completed")

# the model learns to group users based on their shopping behavior, creating a hierarchy that reveals patterns in how different users interact with the online shopping platform.
# The dendrogram visually represents these relationships, showing which users are more similar to each other in terms of their shopping habits and preferences. 
# This analysis can help businesses understand their customer base better and tailor marketing strategies accordingly.
# Overall, this project demonstrates how hierarchical clustering can be applied to analyze user behavior in an online shopping context, providing insights into customer segmentation and preferences.