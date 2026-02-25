import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset
data = {
    "Income": [25000, 50000, 40000, 80000, 30000, 90000, 20000, 70000],
    "Credit_Score": [600, 750, 680, 800, 620, 820, 580, 770],
    "Loan_Amount": [100000, 200000, 150000, 300000, 120000, 350000, 90000, 250000],
    "Loan_Status": [0, 1, 1, 1, 0, 1, 0, 1]  # 0 = Rejected, 1 = Approved
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Income", "Credit_Score", "Loan_Amount"]]
Y = df["Loan_Status"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict new applicant
new_applicant = pd.DataFrame([[60000, 720, 180000]],
                             columns=["Income", "Credit_Score", "Loan_Amount"])

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")