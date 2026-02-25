import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset
data = {
    "Resume_Text": [
        "Python SQL Machine Learning Data Analysis",
        "Java Spring Boot Microservices Developer",
        "Project Management Leadership Planning",
        "Deep Learning Python NLP TensorFlow",
        "Team Management Budget Planning"
    ],
    "Job_Role": [
        "Data_Scientist",
        "Developer",
        "Manager",
        "Data_Scientist",
        "Manager"
    ]
}

df = pd.DataFrame(data)

# Step 2: Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Resume_Text"])

Y = df["Job_Role"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict new resume
new_resume = ["Python Machine Learning Data Visualization"]
new_resume_vector = vectorizer.transform(new_resume)

prediction = model.predict(new_resume_vector)

print("Predicted Job Role:", prediction[0])