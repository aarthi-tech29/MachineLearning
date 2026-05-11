import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
patients = pd.read_csv("patient_symptoms.csv")
images = pd.read_csv("image_features.csv")

# =========================
# MERGE SYMPTOMS + IMAGE FEATURES
# =========================
data = patients.merge(images, on="ImageID")

# =========================
# FEATURE SELECTION
# =========================
features = [
    "Itching", "Redness", "Swelling", "Blisters",
    "Rash", "Pain", "Age",
    "Texture", "ColorIntensity", "EdgeSharpness", "LesionSize"
]

X = data[features]

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
# DISEASE PREDICTION
# =========================
def predict_disease(patient_id):

    idx = data[data["PatientID"] == patient_id].index[0]

    patient_vector = X_scaled[idx].reshape(1, -1)

    distances, indices = model.kneighbors(patient_vector)

    similar_cases = data.iloc[indices[0]]

    # Majority disease vote
    predicted_disease = similar_cases["Disease"].mode()[0]

    # =========================
    # DOCTOR REPORT
    # =========================
    print("\nMEDICAL REPORT")
    print("------------------------")
    print("Patient ID:", patient_id)
    print("Predicted Condition:", predicted_disease)

    print("\nSimilar Cases Found:")

    for i in indices[0]:
        print("-", data.iloc[i]["PatientID"], "→", data.iloc[i]["Disease"])

    print("\n Note: This is similarity-based prediction, not medical diagnosis.")

# =========================
# RUN SYSTEM
# =========================
pid = input("Enter Patient ID: ")
predict_disease(pid)

# ========================
# The model learns:
# - Patients with similar symptoms and image features tend to have the same skin condition.
# - The KNN model identifies the most similar past cases to predict the likely disease for a new patient based on their symptoms and image features.
# - This system can assist doctors in making informed decisions by providing insights from similar cases, but it should not replace professional medical diagnosis.

# Input example:
# PatientID: P1