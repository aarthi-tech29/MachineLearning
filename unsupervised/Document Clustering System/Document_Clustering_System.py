import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 1: Create document dataset
documents = [
    "Stock market rises as economy improves",
    "Sensex and Nifty show strong growth in market",
    "Football team wins championship match",
    "Cricket world cup final match highlights",
    "New AI technology improves healthcare sector",
    "Machine learning used in medical diagnosis"
]

df = pd.DataFrame({"Document": documents})

# Step 2: Convert text to numerical form using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # use single + two-word phrases ex: stock market, football team
    max_df=0.85,          # ignore very common words
    min_df=1
)

# stop_words="english"	Removes common useless words
# ngram_range=(1,2)	Uses single + two-word phrases
# max_df=0.85	Ignores words appearing in many docs
# min_df=1	Keeps words appearing at least once

X = vectorizer.fit_transform(df["Document"])

# Step 3: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Step 4: Display results
print("Document Clustering Result:\n")
print(df)

# Step 5: Show top keywords for each cluster
print("\nTop terms per cluster:\n")

terms = vectorizer.get_feature_names_out()

for i in range(3):
    print(f"Cluster {i}:")
    center_terms = kmeans.cluster_centers_[i].argsort()[-5:][::-1] # top 5 terms
    for idx in center_terms:
        print("  -", terms[idx])
    print()