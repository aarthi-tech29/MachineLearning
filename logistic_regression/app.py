import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report  
 
# Load dataset  
data = pd.read_csv("studentsdata.csv")  
 
# Define features and target  
X = data[['StudyHours', 'Attendance']]  
y = data['Pass']  
 
# Split dataset  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  
 
# Create model  
model = LogisticRegression()  
 
# Train model  
model.fit(X_train, y_train)  
 
# Predictions  
y_pred = model.predict(X_test)  
y_prob = model.predict_proba(X_test)  
 
# Compare actual and predicted  
results = pd.DataFrame({  
    'Actual': y_test.values,  
    'Predicted': y_pred  
})  
 
print("\nResults:\n", results)  
 
# Evaluation  
print("\nAccuracy:", accuracy_score(y_test, y_pred))  
print("Precision:", precision_score(y_test, y_pred))  
print("Recall:", recall_score(y_test, y_pred))  
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  
 
# Coefficients  
print("\nCoefficients:", model.coef_)  
print("Intercept:", model.intercept_)  