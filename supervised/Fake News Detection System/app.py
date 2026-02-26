import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create news dataset
data = {
    "News_Text": [
        "Government announces new economic policy",
        "Breaking: Celebrity found alive on Mars",
        "Scientists discover new vaccine for disease",
        "You won't believe what this politician did",
        "Stock market shows steady growth today",
        "Miracle cure discovered overnight shocking doctors"
    ],
    "Is_Fake": [0, 1, 0, 1, 0, 1]   # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# Step 2: Convert text to numerical form (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["News_Text"])
Y = df["Is_Fake"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict new news article
new_news = ["Shocking secret revealed that doctors don't want you to know"]
new_news_vector = vectorizer.transform(new_news)

# Words like “shocking, miracle, secret, unbelievable”→ usually appear in fake news

prediction = model.predict(new_news_vector)

if prediction[0] == 1:
    print("Fake News")
else:
    print("Real News")