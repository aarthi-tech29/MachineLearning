# =====================================================
# HOSPITAL DEPARTMENT RELATIONSHIP MAPPING
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

data = pd.read_csv("hospital_departments.csv")

print("\n========== HOSPITAL DATASET ==========")
print(data)

# =========================
# DOCTOR SPECIALIZATION ANALYSIS
# =========================

features = data[[
    "Doctors",
    "PatientsPerDay",
    "SurgeryCases",
    "EmergencyCases",
    "CardiologyScore",
    "NeurologyScore",
    "OrthopedicScore"
]]

print("\n========== DOCTOR SPECIALIZATION ANALYSIS ==========")
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
# PATIENT FLOW MAPPING
# =========================

plt.figure(figsize=(5,7))

plt.title("Hospital Department Relationship Mapping")

dendrogram(
    linkage_matrix,
    labels=data["Department"].values,
    leaf_rotation=90
)

plt.xlabel("Departments")
plt.ylabel("Distance")

plt.tight_layout()

# =========================
# DEPARTMENT CLUSTER HIERARCHY
# =========================

plt.show()

print("\n========== DEPARTMENT CLUSTER HIERARCHY DISPLAYED ==========")

# =========================
# FINAL OUTPUT
# =========================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")

print("Doctor Specialization Analysis Completed")
print("Patient Flow Mapping Completed")
print("Department Cluster Hierarchy Completed")

# the model learns to group hospital departments based on their characteristics, creating a hierarchy that reveals relationships between different departments.
# The dendrogram visually represents these relationships, showing which departments are more similar to each other in terms of their doctor specialization and patient flow patterns. 
# This analysis can help hospital administrators understand how different departments are related and optimize resource allocation accordingly.
# Overall, this project demonstrates how hierarchical clustering can be applied to analyze hospital department relationships, providing insights into departmental 
# similarities and helping healthcare professionals make informed decisions about patient care and hospital management.
