import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
users = pd.read_csv("fitness_users.csv")
workouts = pd.read_csv("workout_plan.csv")

# =========================
# BMI CALCULATION
# =========================
users["BMI"] = users["Weight_kg"] / ((users["Height_cm"] / 100) ** 2)

# =========================
# ENCODE CATEGORICAL DATA
# =========================
users_encoded = pd.get_dummies(users, columns=["WorkoutPreference", "Goal"])

# =========================
# FEATURE SELECTION
# =========================
features = [
    "BMI", "Age", "WorkoutDays", "HoursPerDay"
] + [col for col in users_encoded.columns if "WorkoutPreference_" in col or "Goal_" in col]

X = users_encoded[features]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN KNN MODEL
# =========================
model = NearestNeighbors(n_neighbors=3, metric="cosine")
model.fit(X_scaled)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def match_fitness_partner(user_id):

    idx = users[users["UserID"] == user_id].index[0]

    user_vector = X_scaled[idx].reshape(1, -1)

    distances, indices = model.kneighbors(user_vector)

    similar_users = users.iloc[indices[0]]

    # REMOVE SELF MATCH 
    similar_users = similar_users[similar_users["UserID"] != user_id]

    print("\nFITNESS PARTNER MATCHING REPORT")
    print("------------------------------------")
    print("User ID:", user_id)
    print("BMI:", round(users.iloc[idx]["BMI"], 2))
    print("Goal:", users.iloc[idx]["Goal"])

    print("\nSimilar Fitness Partners:")
    for u in similar_users["UserID"]:
        print("-", u)

    # =========================
    # WORKOUT RECOMMENDATION
    # =========================
    user_goal = users.iloc[idx]["Goal"]

    workout = workouts[workouts["Goal"] == user_goal]["Workout"].values[0]

    print("\nRecommended Workout Plan:")
    print("-", workout)

# =========================
# RUN SYSTEM
# =========================
uid = input("Enter User ID: ")
match_fitness_partner(uid)
# =========================

# The model learns:
# - Similar users based on BMI, age, workout habits, and preferences.
# - Recommends workout plans aligned with the user's fitness goals.
# - Provides a list of potential fitness partners with similar profiles for motivation and accountability.
# This KNN-based fitness partner matching app can help users find workout buddies and personalized workout plans, enhancing their fitness journey.

# Input Example:
# Enter User ID: U1
