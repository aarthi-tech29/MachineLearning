# =====================================================
# MUSIC GENRE EVOLUTION ANALYSIS
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

data = pd.read_csv("music_genres.csv")

print("\n========== MUSIC GENRE DATASET ==========")
print(data)

# =========================
# SONG FEATURE EXTRACTION
# =========================

features = data[[
    "Tempo",
    "Energy",
    "Danceability",
    "Acousticness",
    "Loudness",
    "Popularity"
]]

print("\n========== SONG FEATURES ==========")
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
# GENRE RELATIONSHIP MAPPING
# =========================

plt.figure(figsize=(5,7))

plt.title("Music Genre Evolution Analysis")

dendrogram(
    linkage_matrix,
    labels=data["Genre"].values,
    leaf_rotation=90
)

plt.xlabel("Music Genres")
plt.ylabel("Distance")

plt.tight_layout()

# =========================
# HIERARCHICAL VISUALIZATION
# =========================

plt.show()

print("\n========== HIERARCHICAL VISUALIZATION DISPLAYED ==========")

# =========================
# FINAL OUTPUT
# =========================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")

print("Song Feature Extraction Completed")
print("Genre Relationship Mapping Completed")
print("Hierarchical Visualization Completed")

# The model learns to group music genres based on their characteristics, creating a hierarchy that reveals relationships between different genres.
# The dendrogram visually represents these relationships, showing which genres are more similar to each other in terms of their musical features.
# This analysis can help music producers, artists, and listeners understand the evolution of music genres and
# discover new genres that are similar to their preferences.
# Overall, this project demonstrates how hierarchical clustering can be applied to analyze music genres, providing insights into genre similarities and helping music enthusiasts explore the rich diversity of musical styles.