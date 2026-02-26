import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create patient dataset
data = {
    "Fever": [1, 1, 0, 1, 0, 0, 1, 0],          # 1 = Yes, 0 = No
    "Cough": [1, 0, 1, 1, 0, 1, 1, 0],
    "Fatigue": [1, 1, 0, 1, 0, 0, 1, 0],
    "Headache": [0, 1, 1, 1, 0, 0, 1, 0],
    "Disease": [1, 1, 0, 1, 0, 0, 1, 0]        # 1 = Disease, 0 = Healthy
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Fever", "Cough", "Fatigue", "Headache"]]
Y = df["Disease"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Step 4: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict disease for a new patient
new_patient = pd.DataFrame([[1, 1, 1, 0]],
                           columns=["Fever", "Cough", "Fatigue", "Headache"])

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Disease Detected")
else:
    print("No Disease Detected")