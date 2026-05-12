# =========================================================
# SMART SCHOLARSHIP ELIGIBILITY SYSTEM
# USING DECISION TREE
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD CSV DATASET
# =========================================================

df = pd.read_csv("scholarship_eligibility_dataset.csv")

print("\nDATASET")
print("------------------------------------------------")
print(df.head())

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

df["Community_Category"] = label_encoder.fit_transform(
    df["Community_Category"]
)

# =========================================================
# INPUT FEATURES AND TARGET
# =========================================================

X = df[[
    "Family_Income",
    "Attendance_Percentage",
    "Academic_Marks",
    "Community_Category"
]]

y = df["Scholarship_Status"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# DECISION TREE MODEL
# =========================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print("------------------------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")
print("------------------------------------------------")
print(classification_report(y_test, y_pred))

# =========================================================
# USER INPUT
# =========================================================

print("\nENTER STUDENT DETAILS")
print("------------------------------------------------")

family_income = float(input("Enter Family Income: "))
attendance = float(input("Enter Attendance Percentage: "))
marks = float(input("Enter Academic Marks: "))
community = input("Enter Community Category (SC/BC/MBC/OC): ")

# =========================================================
# ENCODE COMMUNITY CATEGORY
# =========================================================

community_encoded = label_encoder.transform([community])[0]

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

student_data = pd.DataFrame({
    "Family_Income": [family_income],
    "Attendance_Percentage": [attendance],
    "Academic_Marks": [marks],
    "Community_Category": [community_encoded]
})

# =========================================================
# ELIGIBILITY PREDICTION
# =========================================================

prediction = model.predict(student_data)

print("\nSCHOLARSHIP ELIGIBILITY RESULT")
print("------------------------------------------------")
print("Prediction:", prediction[0])

# =========================================================
# GOVERNMENT REPORT GENERATION
# =========================================================

report = student_data.copy()

report["Predicted_Status"] = prediction[0]

print("\nGOVERNMENT SCHOLARSHIP REPORT")
print("------------------------------------------------")
print(report)

# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

plt.figure(figsize=(12, 10))

plot_tree(
    model,
    feature_names=[
        "Family Income",
        "Attendance %",
        "Academic Marks",
        "Community Category"
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Smart Scholarship Eligibility Decision Tree")

plt.show()
# ========================================================
# The model learns:
# - How different factors like family income, attendance, marks, and community category affect scholarship eligibility.
# - The decision rules that determine whether a student is eligible for a scholarship or not.
# - The importance of each feature in making the eligibility decision.
# - The model can be used to predict eligibility for new students based on their details and can help generate reports for government use.
# - The decision tree visualization helps understand the decision-making process and the key factors influencing scholarship eligibility.

# The Input example:
# Eligible
# Enter Family Income: 30000
# Enter Attendance Percentage: 92
# Enter Academic Marks: 88
# Enter Community Category (SC/BC/MBC/OC): SC

# Enter Family Income: 45000
# Enter Attendance Percentage: 89
# Enter Academic Marks: 85
# Enter Community Category (SC/BC/MBC/OC): MBC

# Not Eligible
# Enter Family Income: 120000
# Enter Attendance Percentage: 70
# Enter Academic Marks: 65
# Enter Community Category (SC/BC/MBC/OC): OC