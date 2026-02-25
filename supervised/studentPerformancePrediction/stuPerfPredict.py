import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Final_Score": [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Study_Hours"]]  
Y = df["Final_Score"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 5: Predict score for new student
new_student = pd.DataFrame([[9]], columns=["Study_Hours"])
predicted_score = model.predict(new_student)

print("Predicted Score for 9 study hours:", predicted_score[0])