import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
students = pd.read_csv("student_profile.csv")
courses = pd.read_csv("courses.csv")

# =========================
# FEATURE SELECTION
# =========================
features = [
    "Math", "Programming", "Communication", "ProblemSolving",
    "Interest_AI", "Interest_Web", "Interest_DataScience",
    "CGPA"
]

X = students[features]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN KNN MODEL
# =========================
model = NearestNeighbors(n_neighbors=4, metric="cosine")
model.fit(X_scaled)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend_courses(student_id):

    idx = students[students["StudentID"] == student_id].index[0]

    student_vector = X_scaled[idx].reshape(1, -1)

    distances, indices = model.kneighbors(student_vector)

    similar_students = students.iloc[indices[0]]

    # REMOVE SELF MATCH (IMPORTANT FIX)
    similar_students = similar_students[similar_students["StudentID"] != student_id]

    print("\nSTUDENT COURSE RECOMMENDATION REPORT")
    print("----------------------------------------")
    print("Student ID:", student_id)
    print("CGPA:", students.iloc[idx]["CGPA"])

    print("\nSimilar Students Found:")
    for sid in similar_students["StudentID"]:
        print("-", sid)

    # =========================
    # COURSE RECOMMENDATION LOGIC
    # =========================
    recommended_courses = set()

    for _, student in similar_students.iterrows():

        if student["Interest_AI"] >= 4:
            recommended_courses.add("Machine Learning")
            recommended_courses.add("Deep Learning")

        if student["Interest_Web"] >= 4:
            recommended_courses.add("Web Development")
            recommended_courses.add("React & Node.js")

        if student["Interest_DataScience"] >= 4:
            recommended_courses.add("Data Science")

        if student["Programming"] >= 85:
            recommended_courses.add("Python Programming")

    print("\nRecommended Courses:")
    for course in recommended_courses:
        print("-", course)

# =========================
# RUN SYSTEM
# =========================
student_id = input("Enter Student ID: ")
recommend_courses(student_id)

# ======================================================

# The model learns:
# How to build student profiles based on their skills and interests
# How to use KNN to find similar students and recommend courses
# How to handle edge cases (like self-matching) in KNN recommendations
# How to provide actionable course recommendations based on similar students' profiles

# Input Example:
# StudentID: S1
# | Input | Meaning                   |
# | ----- | ------------------------- |
# | S1    | AI strong student         |
# | S2    | Web development student   |
# | S3    | High performer AI student |
# | S4    | Average web student       |
# | S5    | Balanced student          |
# | S6    | Web + moderate skills     |
# | S7    | Top AI student            |
# | S8    | Weak web-focused student  |
