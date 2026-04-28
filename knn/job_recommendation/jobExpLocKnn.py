import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# -------------------------------
# 1. Load dataset
# -------------------------------
jobs = pd.read_csv("jobExpLocKnn.csv")
# Reads all jobs added by Admin
# Stored in a table (DataFrame)

# -------------------------------
# 2. Handle missing values
# -------------------------------
jobs["required_skills"] = jobs["required_skills"].fillna("")
jobs["job_description"] = jobs["job_description"].fillna("")
jobs["location"] = jobs["location"].fillna("Unknown")
jobs["experience"] = jobs["experience"].fillna(0)

# Skills → empty
# Description → empty
# Location → "Unknown"
# Experience → 0

# Prevents errors
# -------------------------------
# 3. Combine text features
# -------------------------------
jobs["text"] = (
    jobs["required_skills"].str.lower() + " " +
    jobs["job_description"].str.lower()
)
# Makes one single text column
# -------------------------------
# 4. TF-IDF for text
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
text_vectors = vectorizer.fit_transform(jobs["text"])
# Converts words into numbers
# Machine can now understand text
# -------------------------------
# 5. Encode categorical data (location)
# -------------------------------
le = LabelEncoder()
jobs["location_encoded"] = le.fit_transform(jobs["location"])
# Chennai → 0  
# Bangalore → 1  
# Hyderabad → 2
# Converts text → numeric
# -------------------------------
# 6. Numerical features
# -------------------------------
num_features = jobs[["experience", "location_encoded"]]

# Scale numerical data
scaler = StandardScaler()
num_scaled = scaler.fit_transform(num_features)
# Balances values like:
# experience
# location
# So no feature dominates
# -------------------------------
# 7. Combine all features
# -------------------------------
job_vectors = hstack([text_vectors, num_scaled * 2])  # weight numeric features
# Final job data contains:
# [ text data | experience | location ]
# * 2 → gives more importance to experience & location
# -------------------------------
# 8. User input
# -------------------------------
user_skills = "python django mysql"
user_desc = "backend developer"
user_experience = 2
user_location = "Chennai"

# -------------------------------
# 9. Process user text
# -------------------------------
user_text = (user_skills + " " + user_desc).lower()
user_text_vector = vectorizer.transform([user_text])
# Same TF-IDF process
# Now user is also converted to numbers
# -------------------------------
# 10. Handle unknown location safely
# -------------------------------
if user_location in le.classes_:
    user_location_encoded = le.transform([user_location])[0]
else:
    user_location_encoded = 0  # default fallback
# If location exists → encode
#     Else → default 0

# Prevents crash
# -------------------------------
# 11. Process numerical input
# -------------------------------
user_num = scaler.transform(
    np.array([[user_experience, user_location_encoded]])
)
# Converts:
# experience
# location
# Same scaling as jobs
# -------------------------------
# 12. Combine user features
# -------------------------------
user_vector = hstack([user_text_vector, user_num * 2])
# Final user vector:
# [ text | experience | location ]
# -------------------------------
# 13. Apply KNN (FIXED)
# -------------------------------
k = min(5, len(jobs))  # prevents crash if dataset is small
model = NearestNeighbors(n_neighbors=k, metric='cosine') # Model learns all job data
model.fit(job_vectors)

distances, indices = model.kneighbors(user_vector) # Finds closest jobs to user

# -------------------------------
# 14. Output results
# -------------------------------
print("\n Recommended Jobs for You:\n")

for i, idx in enumerate(indices[0]):
    score = (1 - distances[0][i]) * 100 # Converts distance → percentage
    print(f"{i+1}. {jobs.iloc[idx]['job_title']} - {round(score, 2)}% match")

# -------------------------------
# 15. Debug (optional)
# -------------------------------
print("\n--- Debug Info ---\n")

for index, row in jobs.iterrows():
    print("Job:", row["job_title"])
    print("Skills:", row["required_skills"])
    print("Location:", row["location"])
    print("Experience:", row["experience"])
    print("-----------------------------")