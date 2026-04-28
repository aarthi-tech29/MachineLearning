import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# -------------------------------
# 1. Load jobs data
# -------------------------------
jobs = pd.read_csv("job.csv")
 
# -------------------------------
# 2. Clean + combine text
# -------------------------------
jobs["job_text"] = (
    jobs["required_skills"].str.lower() + " " +
    jobs["job_description"].str.lower()
)
 
# -------------------------------
# 3. Convert text → numbers (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(jobs["job_text"])
 
# -------------------------------
# 4. User input
# -------------------------------
user_skills = "python django mysql"
user_description = "I am interested in backend development"
 
user_text = (user_skills + " " + user_description).lower()
 
# -------------------------------
# 5. Convert user text → numbers
# -------------------------------
user_vector = vectorizer.transform([user_text])
 
# -------------------------------
# 6. Calculate similarity
# -------------------------------
scores = cosine_similarity(user_vector, job_vectors)[0]
 
# -------------------------------
# 7. Add scores to dataframe
# -------------------------------
jobs["match_score"] = scores * 100
 
# -------------------------------
# 8. Sort jobs by match score
# -------------------------------
recommended_jobs = jobs.sort_values(by="match_score", ascending=False)
 
# -------------------------------
# 9. Show ALL jobs (not only top 3)
# -------------------------------
print("\nRecommended Jobs:\n")
 
for index, row in recommended_jobs.iterrows():
    print(f"{row['job_title']} - {round(row['match_score'], 2)} % match")
 
# -------------------------------
# 10. Debug (optional)
# -------------------------------
print("\n--- Detailed Calculation ---\n")
 
for index, row in jobs.iterrows():
    print("Job:", row["job_title"])
    print("Text:", row["job_text"])
    print("Score:", round(row["match_score"], 2), "%")
    print("---------------------------")
# =========================================================================================
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
 
# jobs = pd.read_csv("jobs.csv")
 
# jobs["text"] = jobs["job_title"] + " " + jobs["skills"] + " " + jobs["location"]
 
# vectorizer = TfidfVectorizer()
# job_vectors = vectorizer.fit_transform(jobs["text"])
 
# user_input = input("Enter your skills and location: ")
 
# user_vector = vectorizer.transform([user_input])
 
# scores = cosine_similarity(user_vector, job_vectors)[0]
 
# jobs["score"] = scores
 
# result = jobs.sort_values("score", ascending=False).head(3)
 
# print("\nRecommended Jobs:")
# print(result[["job_title", "location", "score"]])

