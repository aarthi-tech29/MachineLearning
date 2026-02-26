import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: Create text dataset
documents = [
    "Stock market rises as economy improves",
    "Sensex and Nifty show strong market growth",
    "Football team wins championship match",
    "Cricket world cup final match highlights",
    "New AI technology improves healthcare",
    "Machine learning used in medical diagnosis"
]

df = pd.DataFrame({"Text": documents})

# Step 2: Convert text to numerical format (Bag of Words)
vectorizer = CountVectorizer(stop_words="english") # Converts text → numbers.
X = vectorizer.fit_transform(df["Text"])

# Step 3: Apply LDA Topic Modeling
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# LDA works on word counts

# Step 4: Display topics
words = vectorizer.get_feature_names_out()

for idx, topic in enumerate(lda.components_):
    print(f"\nTopic {idx}:")
    print(" ".join([words[i] for i in topic.argsort()[-5:]]))