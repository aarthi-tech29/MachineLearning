import os
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

# os → access files in folder
# numpy → handle arrays (numbers)
# PIL → read and process images safely
# NearestNeighbors → find similar images
# KNeighborsClassifier → predict label

# -------------------------
# 1. Folder Path
# -------------------------
image_folder = "images"

image_data = []
labels = []
image_names = []

# -------------------------
# 2. Load Images (SAFE)
# -------------------------
for file in os.listdir(image_folder):
# Loop through all files in the folder
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')): #Skips non-image files
        continue

    img_path = os.path.join(image_folder, file)
    print("Reading:", img_path)

    try:
        img = Image.open(img_path).convert("L")   # grayscale
        img = img.resize((100, 100))
        img = np.array(img).flatten()

# Convert to grayscale (simplifies data)
# Resize to same size (important for ML)
# Convert image → numbers
# Flatten → 1D vector

    except Exception as e:
        print("Skipped:", file, e)
        continue

    image_data.append(img)
    image_names.append(file)

    # label from filename (cat_1 → cat)
    label = file.split("_")[0]
    labels.append(label)

# -------------------------
# 3. Convert to Array
# -------------------------
if len(image_data) == 0:
    print("No valid images found!")
    exit()

X = np.array(image_data)
y = np.array(labels)

# ML models need data in array format

# -------------------------
# 4. Train Models
# -------------------------
sim_model = NearestNeighbors(n_neighbors=3)
sim_model.fit(X)

clf_model = KNeighborsClassifier(n_neighbors=3)
clf_model.fit(X, y)

# Two models:
# 1. Similarity model
# Finds closest images
# 2. Classification model
# Predicts label (cat/dog)

# -------------------------
# 5. Query Image
# -------------------------
query_path = os.path.join(image_folder, "cat_1.jpg")

try:
    query_img = Image.open(query_path).convert("L")
    query_img = query_img.resize((100, 100))
    query_img = np.array(query_img).flatten().reshape(1, -1)
    # Same processing as training images
except:
    print("Query image error!")
    exit()

# -------------------------
# 6. Predict Label
# -------------------------
prediction = clf_model.predict(query_img)[0]

print("\nPredicted Label:", prediction)

# Output:cat or dog

# -------------------------
# 7. Find Similar Images (Filtered)
# -------------------------
distances, indices = sim_model.kneighbors(query_img) #Finds closest images based on distance

print("\nSimilar Images (Filtered):")

for i in indices[0]:
    file_name = image_names[i]
    label = file_name.split("_")[0]

    # Show only same label
    if label == prediction:
        print(file_name)
# Shows only:
# Same category images (cat only)
# Removes wrong ones (dog)