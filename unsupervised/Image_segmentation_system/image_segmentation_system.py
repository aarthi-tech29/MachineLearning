import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Read image
image = cv2.imread("image.jpg")        # put any medical/satellite image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Reshape image into pixels
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(pixel_values)

# Step 4: Replace pixel values with cluster center values
centers = np.uint8(kmeans.cluster_centers_)
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(image.shape)

# Step 5: Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()