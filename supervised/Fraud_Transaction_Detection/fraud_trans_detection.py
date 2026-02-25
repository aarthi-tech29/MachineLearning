import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset
data = {
    "Transaction_Amount": [100, 2000, 150, 5000, 300, 7000, 250, 9000],
    "Transaction_Time": [1, 23, 2, 1, 3, 0, 2, 1],   # Time in hours (0-23)
    "Is_Fraud": [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Not Fraud, 1 = Fraud
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Transaction_Amount", "Transaction_Time"]]
Y = df["Is_Fraud"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict new transaction
new_transaction = pd.DataFrame([[8000, 1]], columns=["Transaction_Amount", "Transaction_Time"])
prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("Transaction is Fraud")
else:
    print("Transaction is Safe")