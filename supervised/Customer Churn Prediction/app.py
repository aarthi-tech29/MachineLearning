import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create customer dataset
data = {
    "Monthly_Charges": [500, 600, 450, 900, 850, 400, 1000, 300],
    "Tenure_Months": [24, 18, 36, 6, 8, 40, 4, 48],
    "Customer_Support_Calls": [1, 2, 0, 5, 4, 0, 6, 0],
    "Churn": [0, 0, 0, 1, 1, 0, 1, 0]   # 1 = Left, 0 = Stayed
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Monthly_Charges", "Tenure_Months", "Customer_Support_Calls"]]
Y = df["Churn"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Step 4: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 5: Predict churn for a new customer
new_customer = pd.DataFrame([[950, 5, 4]],
                            columns=["Monthly_Charges", "Tenure_Months", "Customer_Support_Calls"])

# Monthly_Charges	Monthly bill amount
# Tenure_Months	How long customer stayed
# Customer_Support_Calls	Complaints made
# Churn	Target (0/1)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will stay")