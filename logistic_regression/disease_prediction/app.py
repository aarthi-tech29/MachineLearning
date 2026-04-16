import pandas as pd  
import numpy as np  

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report  

# Load dataset  
data = pd.read_csv("healthdata.csv")  

# Symptoms / health indicators
X = data[['Fever', 'Cough', 'Fatigue', 'BloodPressure', 'SugarLevel']]  

# Target: 0 = No Disease, 1 = Disease
y = data['Disease']  

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
print("\nAccuracy:", accuracy_score(y_test, y_pred))  # Overall correctness
print("Precision:", precision_score(y_test, y_pred))  # Out of predicted “Disease”, how many are correct
print("Recall:", recall_score(y_test, y_pred))  # Out of actual “Disease”, how many detected
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))  
# [[TN  FP]
#  [FN  TP]]
# TN → correct No Disease
# TP → correct Disease
# FP → wrong Disease
# FN → missed Disease
print("\nClassification Report:\n", classification_report(y_test, y_pred))  

# Coefficients  
print("\nCoefficients:", model.coef_)  # These values tell how each input affects the prediction, show the impact of each feature on disease prediction
# Feature	    Coefficient	   Meaning
# Fever	        +1.25	       Increases chance of disease
# Cough	        +2.43	       Strongly increases disease 
# Fatigue	    -0.21	       Slightly decreases disease
# BloodPressure	+0.0047	       Very small effect
# SugarLevel	+0.075	       Slight increase
# Positive value - More value → more chance of disease
# Negative value - More value → less chance of disease
print("Intercept:", model.intercept_)  # starting value - model starts with No Disease