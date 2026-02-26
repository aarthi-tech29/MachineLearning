
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create email dataset
data = {
    "Email_Text": [
        "Win cash prize now click here",
        "Meeting scheduled tomorrow at office",
        "Congratulations you won a lottery",
        "Project discussion at 10 am",
        "Limited offer buy now",
        "Please find the attached report"
    ],
    "Is_Spam": [1, 0, 1, 0, 1, 0]   # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Step 2: Convert text into numerical form (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Email_Text"])

Y = df["Is_Spam"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Step 4: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Real-time prediction
new_email = ["You have won free cash prize"]
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")