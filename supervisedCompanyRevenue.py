import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create dataset
data = {
    "Years_in_Business": [1, 2, 3, 4, 5, 6, 7, 8],
    "Revenue": [50000, 90000, 130000, 180000, 230000, 300000, 370000, 450000]
}

df = pd.DataFrame(data)

# Step 2: Define input (X) and output (Y)
X = df[["Years_in_Business"]]  
Y = df["Revenue"]

# Step 3: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 5: Predict revenue for new company
new_company = pd.DataFrame([[10]], columns=["Years_in_Business"])
predicted_revenue = model.predict(new_company)

print("Predicted Revenue for 10 years business:", predicted_revenue[0])