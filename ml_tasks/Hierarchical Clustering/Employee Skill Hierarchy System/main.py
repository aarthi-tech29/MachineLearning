# =====================================================
# EMPLOYEE SKILL HIERARCHY SYSTEM
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

data = pd.read_csv("employee_skills.csv")

print("\n========== EMPLOYEE SKILL DATASET ==========")
print(data)

# =========================
# SKILL MATRIX ANALYSIS
# =========================

features = data[[
    "Python",
    "Java",
    "SQL",
    "MachineLearning",
    "Communication",
    "Management",
    "CloudComputing"
]]

print("\n========== SKILL MATRIX ANALYSIS ==========")
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
# EMPLOYEE GROUPING
# =========================

plt.figure(figsize=(5,7))

plt.title("Employee Skill Hierarchy System")

dendrogram(
    linkage_matrix,
    labels=data["EmployeeID"].values,
    leaf_rotation=90
)

plt.xlabel("Employees")
plt.ylabel("Distance")

plt.tight_layout()

# =========================
# SKILL RELATIONSHIP HIERARCHY
# =========================

plt.show()

print("\n========== SKILL RELATIONSHIP HIERARCHY DISPLAYED ==========")

# =========================
# FINAL OUTPUT
# =========================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")

print("Skill Matrix Analysis Completed")
print("Employee Grouping Completed")
print("Skill Relationship Hierarchy Completed")

# the model learns to group employees based on their skill sets, 
# creating a hierarchy that reveals relationships between different employees' skills.
# The dendrogram visually represents these relationships, showing which employees have similar skill profiles and how they are clustered together based on their skill similarities.
# This analysis can help HR professionals and managers understand the skill distribution within their workforce, identify potential skill gaps, and make informed decisions about training and development programs.
# Overall, this project demonstrates how hierarchical clustering can be applied to analyze employee skills, providing insights into skill relationships and helping organizations optimize their talent management strategies.
