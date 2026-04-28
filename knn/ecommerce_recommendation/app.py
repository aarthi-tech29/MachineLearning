import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# pandas → used to read and handle dataset
# TfidfVectorizer → converts text into numeric vectors
# cosine_similarity → measures similarity between vectors

# Load dataset
products = pd.read_csv("products.csv")

# Combine important fields
products["text"] = (
    products["product_name"] + " " +
    products["category"] + " " +
    products["description"]
)
# # Machine learning model needs one text column, so we combine all useful info.
# Convert text to vectors
vectorizer = TfidfVectorizer()
product_vectors = vectorizer.fit_transform(products["text"])

# Each word becomes a number
# Important words → higher weight
# Common words (like the, for, men) → removed
# Output:
# Each product becomes a vector (matrix of numbers)


# User input (example: "wireless electronics" or "shoes for running")
user_input = input("Enter product preference (category/description): ")

# Convert user input to vector
user_vector = vectorizer.transform([user_input])

# Same process as products
# Now user input is also numeric

# Compute similarity
scores = cosine_similarity(user_vector, product_vectors)[0]

# This compares:
# user input  VS  all products
# A list of similarity scores
# Range: 0 to 1
# Score	Meaning
# 1	    exact match
# 0	    no match

# Add scores to dataframe
products["score"] = scores

# Get top 3 recommendations
result = products.sort_values("score", ascending=False).head(3)

# Display results
print("\nRecommended Products:")
print(result[["product_name", "category", "price", "score"]])


# Input: wireless electronics,
# Cosine Similarity - Measure similarity, Similarity score, Compare with all items, Manual sorting needed
# KNN - Find nearest items(using cosine distance internally), Distance + neighbors, Returns top K directly, Automatic selection 
# Using cosine similarity, we first compute similarity with all products and then select the nearest ones.
# Using KNN, the model directly returns the nearest (most similar) products using cosine distance.
# Cosine → Calculate similarity
# KNN → Give nearest items

# # =================To get clean output ================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# pandas → used to read and handle dataset
# TfidfVectorizer → converts text into numeric vectors
# cosine_similarity → measures similarity between vectors

# -------------------------------
# 1. Load dataset
# -------------------------------
products = pd.read_csv("products.csv")

# -------------------------------
# 2. Combine important columns
# -------------------------------
products["text"] = (
    products["product_name"] + " " +
    products["category"] + " " +
    products["description"]
)
# Machine learning model needs one text column, so we combine all useful info.
# -------------------------------
# 3. TF-IDF Vectorization
# (removes common words like 'men', 'for', etc.)
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(products["text"])

# Each word becomes a number
# Important words → higher weight
# Common words (like the, for, men) → removed
# Output:
# Each product becomes a vector (matrix of numbers)

# -------------------------------
# 4. User Input
# -------------------------------
user_input = input("Enter product preference: ")

# Convert input to vector
user_vector = vectorizer.transform([user_input])

# Same process as products
# Now user input is also numeric

# -------------------------------
# 5. Compute similarity
# -------------------------------
scores = cosine_similarity(user_vector, product_vectors)[0]

# This compares:
# user input  VS  all products
# A list of similarity scores
# Range: 0 to 1
# Score	Meaning
# 1	    exact match
# 0	    no match

# Add scores to dataframe
products["score"] = scores

# Now each product has a similarity score

# -------------------------------
# 6. Filter low scores (IMPORTANT FIX)
# -------------------------------
threshold = 0.2
filtered_products = products[products["score"] > threshold]

# Removes weak matches
# Keeps only relevant products
# Example:
# 0.67 → keep 
# 0.05 → remove 

# -------------------------------
# 7. Sort and get top results
# -------------------------------
result = filtered_products.sort_values("score", ascending=False).head(3)

# Sort by highest similarity
# Take top 3 products

# -------------------------------
# 8. Display output
# -------------------------------
print("\nRecommended Products:")

if len(result) == 0:
    print("No strong matches found. Try different keywords.")
else:
    print(result[["product_name", "category", "price", "score"]])

# ========================= Use knn + cosine======================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# pandas → to handle dataset
# TfidfVectorizer → converts text → numbers
# NearestNeighbors → finds similar items (KNN)

# -------------------------------
# 1. Load dataset
# -------------------------------
products = pd.read_csv("products.csv")

# -------------------------------
# 2. Combine important columns
# -------------------------------
products["text"] = (
    products["product_name"] + " " +
    products["category"] + " " +
    products["description"]
)
# Because model understands only one text column
# -------------------------------
# 3. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(products["text"])

# Each word gets a weight
# Important words → higher value
# Common words (like “the”, “for”) → removed
# Output:
# Each product becomes a vector (numbers)

# -------------------------------
# 4. Build KNN model (Cosine metric)
# -------------------------------
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(product_vectors)

# metric='cosine' → compares similarity using cosine
# algorithm='brute' → checks all data (accurate for small datasets)
# Now model is ready to find similar products

# -------------------------------
# 5. User Input
# -------------------------------
user_input = input("Enter product preference: ")

# Convert input to vector
user_vector = vectorizer.transform([user_input])

# Same process as products
# So now user input is also in numeric form
# -------------------------------
# 6. Get top K similar products
# -------------------------------
k = 3
distances, indices = model.kneighbors(user_vector, n_neighbors=k)

# indices → positions of similar products
# distances → how far they are
# -------------------------------
# 7. Display results
# -------------------------------
print("\nRecommended Products:")

found = False

for i, idx in enumerate(indices[0]):
    distance = distances[0][i]
#  Loop through top 3 products   
    # Filter weak matches (IMPORTANT)
    if distance < 0.8:   # lower distance = better match
        found = True
        print(
            products.iloc[idx]["product_name"],
            "| Category:", products.iloc[idx]["category"],
            "| Price:", products.iloc[idx]["price"],
            "| Score:", round(1 - distance, 2)  # convert to similarity, similarity = 1 - distance, distance = 0.33 → similarity = 0.67
        )
# Distance range: 0 → 1
# 0 = exact match
# 1 = no match
# So:
# < 0.8 → somewhat relevant
# >= 0.8 → ignore

if not found:
    print("No strong matches found. Try different keywords.")
# ====================================================================================

# # Input: wireless electronics, running shoes men, android smartphone, fitness smartwatch, casual tshirt summer