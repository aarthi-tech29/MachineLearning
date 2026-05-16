# =========================================================
# STUDENT CAREER PREDICTION USING LASSO REGRESSION
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("student_career_data.csv")

print("Dataset Preview")
print(data.head())

# =========================================================
# FEATURE SELECTION
# =========================================================

X = data[[
    'Programming_Skill',
    'Communication_Skill',
    'Math_Score',
    'Creativity_Score',
    'Leadership_Skill',
    'Technical_Knowledge',
    'Project_Score',
    'Internship_Experience'
]]

y = data['Career_Score']

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
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# LASSO REGRESSION MODEL
# =========================================================

# Lasso removes unnecessary features
model = Lasso(alpha=0.5)

model.fit(X_train_scaled, y_train)

# =========================================================
# CAREER PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)

# =========================================================
# MODEL EVALUATION
# =========================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# =========================================================
# CAREER RECOMMENDATION ENGINE
# =========================================================

new_student = pd.DataFrame({
    'Programming_Skill': [95],
    'Communication_Skill': [88],
    'Math_Score': [92],
    'Creativity_Score': [85],
    'Leadership_Skill': [72],
    'Technical_Knowledge': [96],
    'Project_Score': [94],
    'Internship_Experience': [1]
})

new_student_scaled = scaler.transform(new_student)

predicted_career_score = model.predict(new_student_scaled)

print("\nPredicted Career Score:")
print(predicted_career_score[0])

# =========================================================
# CAREER RECOMMENDATION
# =========================================================

score = predicted_career_score[0]

if score >= 100:
    career = "AI Engineer / Data Scientist"
elif score >= 90:
    career = "Software Developer"
elif score >= 80:
    career = "System Analyst"
elif score >= 70:
    career = "Web Developer"
else:
    career = "Technical Support"

print("\nRecommended Career:")
print(career)

# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================

performance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nPerformance Analysis")
print(performance)

# =========================================================
# SKILL FEATURE SELECTION
# =========================================================

selected_features = performance[performance['Coefficient'] != 0]

print("\nImportant Selected Skills")
print(selected_features)

removed_features = performance[performance['Coefficient'] == 0]

print("\nRemoved Unnecessary Skills")
print(removed_features)

# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

plt.figure(figsize=(18,9))

plt.bar(performance['Feature'],
        performance['Coefficient'])

plt.title("Student Skill Importance using Lasso",
          fontsize=22)

plt.xlabel("Skills", fontsize=18, labelpad=20)
plt.ylabel("Coefficient Value", fontsize=18)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.subplots_adjust(left=0.18, bottom=0.35)

plt.grid(True)

plt.show()

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(y_test.values,
         marker='o',
         label='Actual Career Score')

plt.plot(y_pred,
         marker='s',
         label='Predicted Career Score')

plt.title("Actual vs Predicted Career Score")

plt.xlabel("Test Samples")
plt.ylabel("Career Score")

plt.legend()
plt.grid(True)

plt.show()

# =========================================================
# OVERFITTING CHECK
# =========================================================

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\nTraining Score:", train_score)
print("Testing Score :", test_score)

if abs(train_score - test_score) < 0.1:
    print("\nOverfitting Controlled Successfully")
else:
    print("\nPossible Overfitting Detected")

# =========================================================
# PROJECT COMPLETE
# =========================================================

print("\nStudent Career Prediction System Completed Successfully")

# the model learns to predict a student's career score based on various skills and experiences.
#  By analyzing the coefficients, we can identify which skills have the most significant 
# impact on career success and provide personalized career recommendations for students. 
# The project demonstrates how Lasso Regression can be applied for feature selection and 
# prediction in the context of student career development, offering valuable insights for 
# students, educators, and career counselors.